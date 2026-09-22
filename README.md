# Skin Lesion Lab

Computer vision and deep learning experiments for skin lesion classification.
The task is to distinguish melanoma from benign lesions using HSV histograms
with Gaussian mixture models, a small CNN, and EfficientNet-B0.

Every model loads one frozen development split manifest. Linked lesions,
complete patient IDs when available, and known duplicate clusters stay together.
Training, model selection, calibration and development evaluation use distinct
groups. The [evaluation protocol](docs/evaluation-protocol.md) records data use,
seeds, comparison budgets, endpoints and confirmation limits.

The project is experimental and has not been clinically validated. The S09
development comparison has an audited, run-linked report. A later HIBA evaluation
used the saved models and policies, but its primary interval includes zero and
the original release did not completely freeze the execution code.
[The dated development status](docs/development-status-2026-09-22.md) separates
that accepted result from later diagnostic work and future methods.

## Run

Use uv 0.12.13 and Python 3.14.7. Direct dependencies have exact version pins;
`uv.lock` records the full dependency resolution. Linux installs use the
PyTorch CPU wheel index, so the default environment does not install CUDA.
The checked platforms are macOS arm64 and Linux arm64/x86-64 in CI or Docker.

```sh
uv python install
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Supply local metadata and matching JPEG images. Select `ham10000` for a CSV
with `image_id`, `lesion_id`, and `dx`, or `isic2018_task3` for an ISIC 2018 Task 3
export with `isic_id`, `lesion_id`, and `diagnosis_1` through `diagnosis_3`.
The source selection is explicit. See [the source catalog](catalog/sources.json)
for provenance, terms, and the local metadata checksum. Data is not redistributed.

The development data credit is ViDIR Group, Department of Dermatology, Medical University of Vienna. See Philipp Tschandl, Cliff Rosendahl and Harald Kittler, [the HAM10000 data descriptor](https://doi.org/10.1038/sdata.2018.161) (2018), and Noel Codella et al., [the ISIC 2018 challenge paper](https://arxiv.org/abs/1902.03368).
The local export matches ISIC 2018 Task 3 by fields and counts, but its original
download receipt is unavailable. Its [CC-BY-NC terms](https://challenge.isic-archive.com/data/)
remain applicable. HIBA is credited to Hospital Italiano de Buenos Aires under
CC-BY ([dataset DOI](https://doi.org/10.34970/587329)).

```sh
uv run --locked python freeze_splits.py \
  --source isic2018_task3 --metadata-path /path/to/metadata.csv \
  --images-dir /path/to/images --split-manifest runs/development-v1.json
uv run --locked python train_pipeline.py \
  --source isic2018_task3 --metadata-path /path/to/metadata.csv \
  --images-dir /path/to/images --split-manifest runs/development-v1.json
uv run --locked python train_deep_pipeline.py \
  --source isic2018_task3 --metadata-path /path/to/metadata.csv \
  --images-dir /path/to/images --split-manifest runs/development-v1.json \
  --architecture small_cnn --epochs 5
```

The classical command defaults to HSV/GMM. Use `--model prevalence` for a
training-prevalence reference or `--model logistic` for L2 logistic regression
on HSV histograms. Logistic regression fits its scaler on training rows and
chooses C from 0.01, 0.1, 1 and 10 by selection ROC-AUC, with smaller C
breaking ties. GMM selects its component count by training BIC and records
the covariance setting, regularization and convergence. Set `--seed` to
17, 42 or 73 for the declared comparison runs. Each run reads the current
image files and recomputes histograms; it does not reuse a feature cache.
PCA and embedding variants are not part of these baseline runs.

Data paths default to `data/raw/`. Each training command creates a new ignored
directory under `runs/` with `run.json`, a manifest snapshot, checked predictions,
and its results, model, and array files. Pass `--run-id` to name a run or let the
command generate one. An existing ID fails before training. A failed or interrupted
run can be retried with a new ID and `--resume-from OLD_ID`; no files are reused.
Validate and recompute scores from a completed run without the training images:

```sh
uv run --locked python summarize_results.py --run-dir runs/RUN_ID
```

`run.json` records the configuration, source state, input hashes, environment,
runtime, status, and artifact checksums. Replaying a run requires the recorded
source commit and any saved diff, matching metadata and image contents, the lockfile,
the saved split manifest, and the same configuration and seed. The validator rejects
changed or missing artifacts. [The run contract](docs/run-contract.md) describes
the fields and validation rules. The [deep training guide](docs/deep-training.md)
describes head training, full fine-tuning, loss weighting and checkpoint selection.
Pretrained EfficientNet uses `EfficientNet_B0_Weights.IMAGENET1K_V1` and requires
an explicit `--pretrained` option. An uncached weight file needs download access.
Use `--fine-tune-backbone` when training EfficientNet from scratch; a random
frozen backbone is rejected.
Each pipeline writes `cohort_attrition.json` inside its run directory. For
valid metadata, it records a reason for every row. Rows with unknown diagnoses,
excluded cancers, or missing or corrupt images cannot enter the binary cohort.
Invalid IDs, duplicate image content, and conflicting diagnosis fields stop preparation.
The binary task distinguishes melanoma from selected benign lesions. It does
not screen for every skin cancer.

## Container

```sh
docker build -t skin-lesion-lab:local .
docker run --rm skin-lesion-lab:local
```

The default command prints training options. For an experiment, mount data
read-only at `/app/data/raw` and a writable output directory at `/app/runs`,
then pass `python train_deep_pipeline.py ...`.
The runtime uses UID/GID 10001. GPU passthrough is not configured.
Make host output directories writable by UID 10001 before mounting them.
An accelerator installation needs a separate environment and PyTorch wheel
source, followed by its own tests; it is outside the locked CPU setup.

## Evaluation status

The current cohort has lesion IDs, not verified patient identities. Unknown
patient and near-duplicate links remain a limitation. Optional `patient_id` and
`duplicate_cluster_id` metadata columns preserve known links; partial patient
IDs or mixed-label linked groups stop allocation. Image-level run limits are
not supported because they would change the frozen cohort.

The manifest checks all metadata cells and eligible image bytes on each fresh
preparation. Rerunning the freeze command validates an existing manifest; it does
not replace it. Changed data requires an explicit new protocol and manifest.
Output names use `selection`, `calibration` and `development` to distinguish these
roles from legacy `val` and `test` artifacts. Both pipelines fit calibration and
thresholds after model selection, then reuse saved policies for development
evaluation. Both CLIs reject confirmation-purpose manifests.

Deep training defaults to unweighted loss. Use `--loss-strategy pos_weight` for
the training-count-weighted ablation. Weighted outputs need calibration before a
probability-based cost threshold can be interpreted as an optimal decision rule.
The 10:1 cost ratio is illustrative. The S09 report separates variation across
seeds from grouped resampling intervals.

Six saved-model runs cover 884 eligible HIBA dermoscopy images. The paired
ROC-AUC difference was 0.0481 (95% interval -0.0016 to 0.1091), which includes
zero. Later inspection reproduced the saved predictions and report, but the
original release omitted execution dependencies and did not enforce its protocol
digest. The [external status note](docs/external-status-2026-09-22.md) explains
why reconstruction cannot establish a complete pre-outcome freeze. The manuscript
decision remains no-go, and these results do not support clinical use.
The [draft story](docs/experiment-story.md) connects the development comparison,
GMM diagnosis and external result to those limits.

## Notebooks and measured results

The [grouping example](notebooks/01-groups.ipynb) uses generated data. The
[development notebook](notebooks/02-development-report.ipynb) reads the accepted
S09 report, verifies its SHA-256, plan digest and embedded run registry, then
displays run IDs beside the model estimates. It does not open the 27 fitted run
records or validate their files. The report and real images are not distributed. Obtain
the report from the locally validated S09 run; do not substitute a similarly
named file. Start from the repository root with Python 3.14.7 and uv 0.12.13:

```sh
uv sync --locked --group notebooks
uv run --locked --group notebooks python scripts/check_notebooks.py --mode synthetic
uv run --locked --group notebooks python scripts/check_notebooks.py \
  --mode accepted --report /path/to/benchmark-report.json
```

The first execution uses six generated records and constructed predictions for
the grouping example, plus generated predictions for the model table. It labels
both as illustrative.
The second requires the exact accepted private report. Each run writes executed
copies under ignored `outputs/notebooks/`; the published sources stay clean.

On the 1,395-image development set, unweighted full EfficientNet has mean
ROC-AUC 0.8906 across seeds 17, 42 and 73, versus 0.7552 for HSV logistic
regression. The prespecified paired difference is 0.1354 (95% component
bootstrap interval 0.0950 to 0.1761). Its sensitivity difference interval
crosses zero. The 10:1 error cost is illustrative, and referral retains some
missed melanoma images. Grouping covers lesion and known-duplicate links, not
verified patient identities. The results do not establish clinical safety.

A retrospective check of the saved GMM scores found that fixed clipping before
calibration turned most tail scores into ties. Mean raw-score ROC-AUC was 0.7364,
versus the accepted fitted-probability ROC-AUC of 0.5704. The
[development notebook](notebooks/02-development-report.ipynb) gives the three
seed values and limits. This analysis leaves the frozen endpoint intact; it is
not a new trained result or evidence that raw probabilities are calibrated.
The [v2 score interface](docs/calibration-v2.md) is available for a future,
prospectively specified experiment; no real-data performance claim follows from
its software tests.

Dependabot updates Python dependencies, Actions, and container images.

Probability calibration and saved referral rules are described in
[calibration](docs/calibration.md). Run reports distinguish raw scores, weighting
correction and fitted probabilities, with proper scores and referral counts.
