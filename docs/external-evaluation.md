# Frozen external evaluation

External evaluation uses saved estimators, preprocessing and decision policies.
It does not fit a model, choose a checkpoint, recalibrate probabilities or select
a threshold. The primary comparison is full unweighted EfficientNet-B0 versus
HSV logistic regression, with all three seeds 17, 42 and 73.

The supported external source is the original [Hospital Italiano de Buenos
Aires release](https://doi.org/10.34970/587329), attributed to Hospital Italiano
de Buenos Aires under CC-BY. Its 1,616-image bundle contains both dermoscopy and
clinical photographs. The adapter retains contact-polarized dermoscopy and the
existing melanoma-versus-selected-benign diagnoses. It excludes clinical images,
BCC, SCC, actinic keratosis and unknown diagnoses with recorded reasons. See the
[source catalog](../catalog/sources.json) for identity, permissions and hashes.
Dataset images and row-level metadata stay local.

For new execution, review and record the protocol and a complete
[schema-2 release](external-release-v2.md) before accessing external outcomes.
The gate verifies fitted records, estimators, policies, preprocessing, executed
source and runtime inputs. Preserve the trusted release digest separately.
Changing the release and supplied digest together creates a new release.

The original S11 release uses schema 1 and omits some execution dependencies.
It supports explicitly labeled historical inspection, not strict acceptance or
new inference. Read the [dated status note](external-status-2026-09-22.md) before
using its provenance claim.

`src.external.hiba_manifest(metadata, images)` builds the external membership
and exclusion record without allocating development splits. It connects all
source rows by patient, lesion, decoded pixels and supplied duplicate links
before excluding rows, permits different diagnoses across a patient's lesions,
and rejects conflicting lesion ownership or duplicate-pixel diagnoses. Exact
within-cohort pixel duplicates contribute one image, selected by image ID.

Audit the full original source export against the complete external download
using global archive IDs, file hashes, decoded RGB hashes and perceptual-hash
candidates under rotations/flips. `fingerprint` and `overlap_audit` implement
those comparisons. Bind the reviewed audit to the external manifest digest.
The execution gate requires a clear audit with empty `exact` and `near` lists;
confirmed overlaps or unresolved candidates require a reviewed exclusion/audit
revision before execution. Preserve the original audit. Cross-cohort patient
and lesion IDs have local namespaces. No detected image overlap does not prove
patient disjointness or exhaustive visual uniqueness.

Run each declared seed once in a new output directory:

```sh
uv run --locked python scripts/evaluate_external.py run \
  --release runs/external/release.json --release-sha256 RELEASE_DIGEST \
  --manifest runs/external/manifest.json --manifest-sha256 MANIFEST_DIGEST \
  --audit runs/external/audit.json --audit-sha256 AUDIT_DIGEST \
  --images-dir data/external/images --run-id logistic-17 \
  --runs-dir runs/external/predictions --device cpu
```

Use the frozen device and settings for the deep runs. Apply a declared runtime
cap in the experiment controller and preserve failed attempts. The runner
refuses an existing output directory. It records raw scores, predictions,
calibration/referral summaries, actual software versions and source identity.
External labels are supplied only to reporting; deep loaders use dummy labels.
Deep inference caches the deterministic evaluation tensors once for all MC
passes, with a 512 MiB tensor-size limit. The cache preserves image order and
transforms; a synthetic check requires exact cached/uncached MC arrays.

Use the same release, manifest, audit and output arguments with operation
`report` after all six runs complete. The report checks each run's identity,
checksums, raw scores, saved policy and membership before computing metrics.
Its 2,000 shared unstratified patient-component bootstrap draws (seed 2026)
preserve mixed-outcome patients and repeated images. It averages metrics and
paired differences across seeds within each draw, with 95% percentile intervals.
Undefined draws yield a null interval for the affected endpoint. Seed variation
is reported separately. These intervals condition on the fitted seeds. Referral
coverage changes the evaluated population; neither the illustrative 10:1 costs
nor retained-set results establish clinical utility.
