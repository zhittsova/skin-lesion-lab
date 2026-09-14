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

`predictions.csv` is the common row contract. It has exactly these columns:
`schema_version`, `run_id`, `model_key`, `role`, `split_hash`, `image_id`,
`lesion_id`, `group_id`, `target`, `prob_melanoma`, and `prediction`. The group ID
names the manifest's connected component, the unit used for grouped uncertainty
estimates. Each pipeline's role-specific CSV has the same common columns and
keeps its model-specific fields, such as MAP decisions or MC uncertainty.

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
