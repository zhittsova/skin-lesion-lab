# External release contract v2

A file list proves integrity only for the files it lists. External execution
therefore requires a schema-2 release with explicit fitted identities and a
complete inventory of its execution inputs. The caller supplies the reviewed
release SHA-256 separately. Changing both the release and that digest creates
a new release, so the review must establish its timing and intended use.

## Required release fields

| Field | Contract |
| --- | --- |
| `schema_version` | Integer `2`. Version 1 is inspection only. |
| `files` | Canonical project-relative paths mapped to lowercase SHA-256 digests. Include every referenced `run.json` and every artifact in its ledger. |
| `source_files` | Sorted, unique inventory returned by `src.external_release.source_files()`: every Python file under `src/`, plus `scripts/evaluate_external.py`. This deliberately includes more than the current import closure. |
| `environment_path` | A listed JSON file containing `src.external_release.environment()`, captured in the intended runtime. |
| `protocol` | Object with `path` and `sha256`, both bound to `files`. Preserve the reviewed protocol as a separate file. |
| `comparison` | `{"families": ["logistic", "efficientnet-full-unweighted"], "seeds": [17, 42, 73]}`. |
| `reporting` | Frozen `draws`, `seed`, `reference` and `metrics_version`. The primary protocol uses 2,000 draws, seed 2026, logistic reference and metric version 2. |
| `runs` | Mapping from fitted run ID to the entry described below. Paths and family/seed pairs must be unique. |

The inventory also requires `pyproject.toml`, `uv.lock`, `.python-version` and
`docs/evaluation-protocol.md`. A run entry contains `path`, `run_sha256`, all
fields returned by `run_contract.fitted_identity(record, training_metadata)`,
and `estimator`, `policy`, `preprocessing` and `device`. Identity fields include
run ID, pipeline, seed, architecture, pretrained initialization, training mode, loss strategy, positive loss
weight, policy version and configuration digest. The validator derives those
facts from the fitted record and deep training metadata before comparing them
with the release. Every fitted run must also pass
[`run_contract.validate_run`](run-contract.md) before any trusted estimator or
checkpoint is deserialized. That shared validator binds producer and shared
predictions, array schemas and MC contents, calibration provenance and saved
decisions, split/source artifacts, and pipeline metadata. The run ledger must
match the files on disk and the release must include every ledger entry.

For example, removing a calibration producer CSV from disk and both ledgers
while leaving its record reference is invalid even if every remaining hash is
correct. Existing but unregistered files, missing producer keys and disagreement
between producer and shared predictions also reject. Hashes bind the listed
bytes; the fitted-run contract establishes their completeness and consistency.
The release validator then checks the trusted serialized estimator or checkpoint.

For classical fits, `preprocessing` is
`{"kind": "embedded", "artifact": "<run-path>/<estimator-path>"}`. The fitted
logistic scaler and GMM standardization arrays live in that pickle. For deep
fits, it is `{"kind": "transforms", "artifact":
"<run-path>/results/deep_training_metadata.json"}`. The saved evaluation transform
must match the executed transform. Estimator and policy paths must identify the
artifacts required by the pipeline. A listed marker cannot replace them.

The environment file has its own `schema_version: 2`. It records Python,
platform, machine, selected process variables, Torch thread counts and OpenCV
settings. Its `software` map contains the active installed dependency closure
of the ten direct runtime requirements in `pyproject.toml`: NumPy, pandas,
scikit-learn, Torch, torchvision, Pillow, headless OpenCV, Matplotlib, seaborn
and tqdm. The closure follows each installed distribution's requirements,
including nested requirements and applicable platform or extra markers. It
includes numerical dependencies such as SciPy, joblib, threadpoolctl and
SymPy. Names are canonical lowercase distribution names, sorted in the JSON
map. Developer and notebook tools are outside this boundary unless a runtime
requirement brings them in.

Capture and verify the environment in the intended runtime. Linux uses the
project's CPU Torch wheels, whose installed requirements can differ from the
host's. Resolution reads installed metadata only; it does not consult an index.
Missing metadata, an incompatible installed requirement, an omitted package or
any recorded version difference rejects strict execution. The release also
hashes `uv.lock`, but that hash alone does not prove what is installed. Earlier
environment files without version 2, including the incomplete strict snapshots,
reject and must be captured again for a new reviewed release. This does not
upgrade the original schema-1 artifact: its explicit historical inspection path
still checks its original bytes without authorizing inference.

Execution compares the captured environment with the current process and checks
source bytes against both the running checkout and the import-time snapshot.
This binds the declared runtime inputs; it does not promise bitwise equality
across platforms or hardware. Deep inference uses the saved
seed, image size, dropout, batch size and MC pass count. It reconstructs weights
without downloading pretrained weights and forces the deterministic evaluation
behavior defined by the pinned source. Device is declared per run.

Use `run_contract.validate_run` to diagnose fitted runs before assembling a new
release. Strict acceptance repeats that validation automatically. Capture the
source and runtime inventory, copy the reviewed protocol,
construct each run entry from fitted facts, and hash every required file. Inspect
the result with `external.verify_files(release_path, digest, root)` before
external execution. The generated fixture in
[`tests/external_fixtures.py`](../tests/external_fixtures.py) provides a complete
small example, including artifacts that can actually be loaded. Its synthetic
weights and outcomes have no model-quality interpretation.
[`tests/test_release_run_contract.py`](../tests/test_release_run_contract.py)
also fits tiny generated logistic and small-CNN runs through the real training
CLIs and requires both ordinary run validation and strict release acceptance.

## Execution and reporting

`evaluate_external.py run` accepts only v2. Gates run before prediction and again
before completion. Both gates enforce the full fitted-run contract. They check
fitted partitions against external image IDs,
release files, manifest, overlap audit and image bytes. Failed execution retains
a failed record. Existing output directories are never reused.

The report boundary requires exactly the two families and three seeds above,
with no extra run directories. It verifies each saved result's release, audit,
manifest, fitted identity, execution source and environment, then reconstructs
its predictions and policy report from saved scores. It checks inputs and result
files and the full fitted-run contract again before writing the report. A
contract failure prevents an accepted report from being written. CSV reads
preserve lexical image, group and run IDs, including leading zeros and NA-like
strings. Image IDs still follow the source identifier rules.

New run records and report envelopes declare schema 2 and metric version 2.
Prediction tables retain schema 1 for policy v1; policy v2 tables declare schema 2
and include the finite input and ranking score columns. Saved policy application dispatches on policy version. Opt-in GMM v2 uses finite
posterior log odds and saves a separate ranking score, as specified in
[calibration v2](calibration-v2.md). Individual GMM and small-CNN fixtures can be
scored, but they do not replace either family in the primary report matrix.

## Historical inspection

Full fitted-run validation enforces the existing run schema 1 within release
schema 2. It adds no serialized fields or policy semantics, so neither schema
version changes. Incomplete schema-2 releases reject even if an earlier validator
accepted them. They cannot fall back to historical inspection.

The original S11 release is v1. Use `inspect_legacy_release` with an explicit
historical source snapshot and the original protocol to verify its listed bytes
and all six actual mappings. This cannot authorize new inference. Saved-result
reconstruction must explicitly select `legacy=True`, which retains policy-v1
and metric-v1 semantics. The report command exposes that route only with
`--legacy-source-root`, `--legacy-protocol` and a new `--output-report` path.
It labels the output `legacy-v1-inspection` and `strict_release_accepted: false`.

Keep the original release digest and reports. A later inspection cannot establish
that omitted source was frozen before outcomes were seen. The
[dated external status](external-status-2026-09-22.md) records that limitation.
