# Skin Lesion Lab

Computer vision and deep learning experiments for skin lesion classification.
The task is to distinguish melanoma from benign lesions using HSV histograms
with Gaussian mixture models, a small CNN, and EfficientNet-B0.

Images of the same lesion stay in the same train, validation, or test split.
The classical pipeline fits standardization on training data. Evaluation compares
decision thresholds under asymmetric costs, and Monte Carlo dropout estimates
how much CNN predictions vary across stochastic passes.

The project is experimental and has not been clinically validated. Earlier
results still need a methodological audit before they can serve as benchmarks.

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

Supply a local ISIC/HAM10000 metadata export with `isic_id`, `lesion_id`, and
diagnosis columns, plus matching `<isic_id>.jpg` files. The original HAM10000
`image_id` schema needs normalization before use. Data is not redistributed.

```sh
uv run --locked python train_pipeline.py \
  --metadata-path /path/to/metadata.csv --images-dir /path/to/images
uv run --locked python train_deep_pipeline.py \
  --metadata-path /path/to/metadata.csv --images-dir /path/to/images \
  --architecture small_cnn --epochs 5
```

Data paths default to `data/raw/`. Outputs go to ignored `results/`,
`models/`, and `runs/` directories. Pretrained EfficientNet weights require
an explicit `--pretrained` option and download access. Use
`--fine-tune-backbone` when training EfficientNet from scratch.

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

The next experiments need stricter diagnosis mapping, checks for duplicate
images and patient overlap, and saved split manifests shared by every model.
Calibration also needs data separate from model selection.

CNN training uses class weights, so its outputs need calibration before a
probability-based cost threshold can be interpreted as an optimal decision rule.
The 10:1 cost ratio is illustrative. Further evaluation should report variation
across seeds and grouped resamples, followed by testing on an untouched external
cohort.

Planned Jupyter notebooks will explain the experiments and their results.
Dependabot updates Python dependencies, Actions, and container images.
