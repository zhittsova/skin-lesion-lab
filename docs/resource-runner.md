# Resource preflight and GPU pilot

The resource runner separates a resource request from permission to start a full
workload. It probes the current host, measures representative work, checks the
budget, and produces a decision. Execution requires that decision's SHA-256,
an explicit command manifest and a fresh acquisition check.

## Free Colab pilot

Create a Colab notebook, select a free GPU runtime, and run the prepared pilot
notebook. Account sign-in and allocation happen in Colab. These scripts do not
provision cloud machines, purchase credits or verify account billing status.
If Colab cannot allocate a free GPU, stop there.

The pilot uses generated JPEGs, the project models and transforms, batch size 64,
image size 128, and full precision. It measures CNN training, frozen-head and
full-backbone EfficientNet training, evaluation and MC Dropout inference. It
tries the configured loader-worker counts within the available CPU count.
Pretrained weights come from torchvision's declared source.

The repository's Linux lock installs CPU PyTorch. The notebook creates a separate
Python 3.14.7 environment with the same direct model dependency versions and a
GPU-compatible wheel selected by uv 0.12.13. It saves a generated dependency lock
with hashes. Missing compatible wheels stop setup; versions are not silently
downgraded. That GPU environment still needs verification before real experiments.

```sh
python scripts/resource_pilot.py pilot \
  --config configs/resources/colab-free.json \
  --output runs/resource-pilot
```

The example config budgets 15 minutes for probing and 5 hours for the requested
workload, with zero paid budget. Its workload counts describe all eighteen deep
jobs, not the live S09 remainder. Adjust counts to the intended workload before
using the estimated total. Every attempted sample has a log and exit status.
The pilot never starts a full experiment.

## Interpreting the decision

Hardware discovery includes an allocation and synchronized computation, rather
than trusting a device name. Probes run in bounded subprocesses while holding a
device lease. Leases coordinate these runners on one host; they cannot reserve
hardware against unrelated applications or promise continued Colab availability.

The planner checks fresh observations, host memory, device memory, disk, CPU
count and elapsed-time budget. MPS shares the host-memory allowance. CUDA and MPS
are limited to one training job per device in this version. Loader workers are
separate from concurrent training jobs. CPU capacity can exceed one, but the
estimate assumes no unmeasured speedup from that concurrency.

Each timing estimate includes two learning-rate candidates, twenty epochs,
two selection passes per epoch, and thirty inference passes over calibration
and development rows. Startup, hashing, plots and final statistical reporting
are additional. Generated JPEGs do not establish real-data disk throughput.
The memory allowance is 1.5 times the observed footprint; MPS driver memory is
sampled at the end, not measured with a peak counter. The 512-MiB output allowance
per job is a planning allowance, not a measured disk result.

## Execute after preflight

Supply a JSON list of jobs, each with a unique `id`, a measured `workload`, an
`argv` string list, and an absolute `cwd`. Workload counts must match the resource
config. Commands must explicitly specify the measured `--device` and
`--num-workers`. The runner never rewrites those flags to fit available hardware.

```sh
python scripts/resource_runner.py \
  --decision runs/resource-pilot/decision.json \
  --decision-sha256 <sha256-of-that-file> \
  --jobs /absolute/path/to/jobs.json \
  --benchmark-plan /absolute/path/to/plan.json \
  --output runs/resource-execution --execute
```

The executor accepts only canonical deep-training commands from a frozen S09
plan. It verifies the source, inputs, cohort counts, model mode, batch/image size,
epochs, candidate grid and inference passes against the pilot profile. Classical
models need their own measured execution path; this pilot does not cover them.

The executor reacquires the selected device under the lease, rejects stale or
insufficient measurements, and limits concurrency to the calculated capacity.
It checks memory before launching another job, enforces the wall-time and disk
limits, and preserves exit codes and logs. Interruptions stop owned process
groups, including descendants left by an exited leader. Existing output
directories and logs are not overwritten.

This runner manages resources; the training entry points and frozen benchmark
registry still enforce scientific provenance. A resource pilot does not amend a
frozen experiment. Changing an existing run's backend, loader configuration or
source requires a documented execution decision and compatible frozen plan.
Preserve completed runs and the original plan. Never mix incompatible records
to make an incomplete matrix appear complete.
