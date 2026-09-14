# Skin Lesion Lab

Computer vision and deep learning experiments for skin lesion classification.
The task is to distinguish melanoma from benign lesions using HSV histograms
with Gaussian mixture models, a small CNN, and EfficientNet-B0.

Every model loads one frozen development split manifest. Linked lesions,
complete patient IDs when available, and known duplicate clusters stay together.
Training, model selection, calibration and development evaluation use distinct
groups. The [evaluation protocol](docs/evaluation-protocol.md) records data use,
seeds, comparison budgets, endpoints and confirmation limits.

The project is experimental and has not been clinically validated. Earlier
results still need a methodological audit before they can serve as benchmarks.

## Run

Use uv 0.12.11 and Python 3.14.7. Direct dependencies have exact version pins;
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

Data paths default to `data/raw/`. Outputs go to ignored `results/`,
`models/`, and `runs/` directories. Pretrained EfficientNet weights require
an explicit `--pretrained` option and download access. Use
`--fine-tune-backbone` when training EfficientNet from scratch.
Each pipeline writes `cohort_attrition.json` under its output directory. For
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
read-only at `/app/data/raw` and writable output directories at `/app/results`,
`/app/models`, and `/app/runs`, then pass `python train_deep_pipeline.py ...`.
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
roles from legacy `val` and `test` artifacts. Classical calibration is reserved
for later implementation. Deep checkpoint selection and threshold selection use
separate roles. Both CLIs reject confirmation-purpose manifests.

CNN training uses class weights, so its outputs need calibration before a
probability-based cost threshold can be interpreted as an optimal decision rule.
The 10:1 cost ratio is illustrative. Further evaluation should report variation
across seeds and grouped resamples, followed by testing on an untouched external
cohort.

Planned Jupyter notebooks will explain the experiments and their results.
Dependabot updates Python dependencies, Actions, and container images.
