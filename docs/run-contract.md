# Run contract v1

Both training commands write to `runs/<run-id>/`. The ID is either supplied with
`--run-id` or generated. Creating an existing ID fails. A retry uses a new ID;
`--resume-from` records the earlier failed or incomplete run without copying its
files. The earlier directory remains available for inspection.

`run.json` is the run summary. Its `schema_version` is 1. It starts with
`status: running` and becomes `completed` only after artifact validation and
checksums are written. An exception records `status: failed`, the stage, exception
type, and a path-free reason code. Abrupt termination leaves `running`, which the
validator rejects. `runtime_seconds` measures local wall time for a completed or
handled failed run.

The `config` object holds the model arguments and training seed. Input path
arguments are named placeholders. `config_sha256` hashes that object. `inputs`
contains SHA-256 digests for metadata, the split manifest, and `uv.lock`. The
saved manifest also contains each eligible image digest, the cohort hash, group
membership, and split hash. `source` records the Git commit, the dirty diff hash,
and hashes for untracked Python source under `src/` or `scripts/`. The run stores
that diff and those source files under `inputs/`. The `environment` object records
Python, platform, machine, package versions, and the selected device. No exported
field needs a local absolute data path.

The development split manifest keeps schema 1 when the eligible rows already
express every known link. If an excluded metadata row connects eligible images,
schema 2 adds `identity_components`: an image ID to source-component ID map.
Grouping and allocation use those pre-exclusion links. The map is covered by the
split hash and checked again when the manifest is loaded. The accepted v1
development manifest remains schema 1 with its original allocation and hash.

The classical command uses `classical_prevalence`, `classical_logistic` or
`classical_gmm` as its pipeline and model key. Each produces the same
prediction roles and metrics summary. A failed GMM fit records whether
the setting was invalid or the optimizer did not converge.

Deep runs also write `results/deep_training_metadata.json`. It records the
resolved training mode, pretrained weight identifier, transforms, randomness and
device policy, loss weighting, and checkpoint selection. Its checkpoint reference
includes a SHA-256 digest. The run's artifact map hashes both this metadata and
the checkpoint. See [deep training](deep-training.md) for the mode and selection
rules.

`predictions.csv` is the common row contract. It has exactly these columns:
`schema_version`, `run_id`, `model_key`, `role`, `split_hash`, `image_id`,
`lesion_id`, `group_id`, `target`, `prob_melanoma`, and `prediction`. The group ID
names the manifest's connected component, the unit used for grouped uncertainty
estimates. Each pipeline's role-specific CSV has the same common columns and
keeps its model-specific fields, such as MAP decisions or MC uncertainty.
CSV readers treat identity columns as strings, including numeric-looking IDs and
literal `NA` or `NULL`; probabilities remain numeric and round-trip accurately.

The `artifacts` map gives a SHA-256 digest for every file except `run.json`.
Validation checks the complete file set, hashes, JSON and array syntax, array
shapes, manifest content hash, sample membership, labels, group IDs, and agreement
between the common and role-specific predictions. For deep runs it also checks
the saved MC passes, means, labels, spread, and uncertainty fields against those
prediction rows. A changed input can be checked against `inputs` by calling
`validate_run(run_dir, input_paths=...)` with matching local files.

To replay, check out the recorded commit and apply the saved diff and source
files if present. Use the recorded lockfile version, matching metadata and image
bytes, the saved split manifest, and the model arguments in `config`. Give the
new execution a new run ID. The stored paths are placeholders because data may
live elsewhere on another machine. `summarize_results.py --run-dir runs/<run-id>`
validates a completed run and recomputes its reported operating-point scores
from saved predictions. It does not open the training images or metadata.

Calibration-enabled runs store `models/decision_policy.json` and
`results/calibration_report.json`. Shared `prob_melanoma` values are fitted
probabilities; producer tables also keep raw and corrected scores. Validation
checks policy identity, deep checkpoint and pass count, and replays probability,
decision and referral fields. Prediction-only summaries recompute proper scores,
reliability bins and retained-set metrics from saved scores and the locked policy.
Earlier runs without a policy retain their original score meaning.
