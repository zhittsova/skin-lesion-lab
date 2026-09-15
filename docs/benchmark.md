# Controlled development benchmark

The benchmark uses the fixed split and search settings in
[evaluation protocol v1](evaluation-protocol.md). It contains nine classical
runs and eighteen deep searches. Each deep search compares two learning rates
on selection ROC-AUC before calibrating and evaluating its winner. All three
seeds, 17, 42 and 73, remain in the comparison.

## Freeze the schedule

Review and commit the implementation first. Confirm the training device,
maximum runtime and pretrained-weight download policy with the experiment owner.
Keep the benchmark directory under ignored `runs/` or another excluded path.

```sh
uv run --locked python scripts/benchmark_experiment.py freeze \
  --directory runs/benchmark-v1 \
  --metadata-path data/raw/metadata.csv \
  --images-dir data/raw \
  --split-manifest runs/development-v1.json \
  --source isic2018_task3 --device cpu --max-seconds 3600
```

The runtime above is an example, not an approved budget. The command records the
budget and download policy; it does not launch training or enforce a timeout.
The experiment controller must enforce the approved deadline and download
policy when executing the saved argument lists. Add `--allow-weight-download`
only when the owner has authorized the pretrained download. Otherwise the
controller must verify that the required weights are already cached before
starting EfficientNet.

`plan.json` binds the source state, protocol text and checksum, metadata, split,
lockfile, configurations, seeds and full command argument lists. It cannot be
overwritten by the freeze command. Run each saved command from the repository
root, with one training process per assigned device. Preserve every failed run
and its diagnostics. Do not reuse a failed run ID or select a replacement seed.
A correctness fix needs a reviewed new schedule and reruns of affected settings.

## Account for runs and report

```sh
uv run --locked python scripts/benchmark_experiment.py status \
  --directory runs/benchmark-v1
uv run --locked python scripts/benchmark_experiment.py report \
  --directory runs/benchmark-v1
```

Status accounts for every scheduled job, including jobs that have not started,
failed jobs and invalid artifacts. Reporting requires the complete checked
matrix. Each run must match the frozen source, inputs and configuration and
start after the schedule was frozen. Run validation checks all artifact
checksums, predictions and saved decision policies before comparison.

The report averages image-level metrics across the three seeds. Its 2,000
bootstrap draws sample complete manifest components with replacement within
class, using seed 2026 and the same draw for every model and seed. Percentile
95% intervals condition on these seeds and observed class group counts.
Between-seed sample standard deviation and range are separate fields. The
primary comparison is full, unweighted pretrained EfficientNet minus HSV
logistic regression; other contrasts are exploratory.

Average precision and trapezoidal PR-AUC have separate names. Log loss uses the
existing probability clip of `1e-15`. An undefined precision or bootstrap draw
produces a null result or interval, with an undefined-draw count; the report does
not silently discard draws. Calibration bins and referral summaries come from
the saved predictions and policies. All outputs remain development evidence.
Unknown patient identity and unrecorded duplicates limit the group intervals.
No confirmation or clinical performance claim follows from these checks.
