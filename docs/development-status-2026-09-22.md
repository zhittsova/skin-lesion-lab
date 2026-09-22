# Development evidence status, 22 September 2026

The [evaluation protocol](evaluation-protocol.md) is the frozen plan written
before the S09 comparison. Its prospective wording, including the statement
that no model-quality result had yet been validated, is part of that historical
record. This note describes the subsequent development evidence without changing
the protocol or the accepted report.

S09 completed nine model strata at seeds 17, 42 and 73: 27 development runs in
the accepted registry. The report's SHA-256 is
`0f0e898a1af36e05898692a95c9a3f2dfbb59b5c4bb8ad08c144a6e55fd529c7`.
The [development notebook](../notebooks/02-development-report.ipynb) verifies
that report's bytes, plan digest and embedded run registry, then plots its saved
estimates. It does not validate fitted run files. A separate read-only audit
validated all 27 saved run records and their registered artifacts. The notebook
is a view of accepted evidence, not that audit.

On 1,395 development images, the prespecified unweighted full EfficientNet
versus HSV logistic comparison gave mean ROC-AUC 0.8906 versus 0.7552. The paired
difference was 0.1354, with a 95% component bootstrap interval of 0.0950 to
0.1761. The sensitivity difference interval crossed zero. These intervals
condition on the three seeds and known lesion/duplicate groups; patient identity
and unrecorded near-duplicate links remain limitations. The illustrative 10:1
error cost is not a clinical utility estimate. Referral still left missed
melanoma images. The comparison supports a development result, not clinical
validation or use in diagnosis.

## Retrospective GMM diagnosis

The accepted GMM result uses the original sigmoid calibration and its fitted
probabilities. Its fixed `[1e-10, 1-1e-10]` clipping collapsed distinct raw tail
scores into ties before fitting. Read-only analysis of the three saved score
tables found mean raw-score ROC-AUC 0.7363719 and mean fitted-probability ROC-AUC
0.5704370. Across 1,395 scores per seed, 1,292, 1,291 and 1,220 lay outside the
old clip bounds. All three saved slopes were positive, so the fitted sigmoid
could not recover distinctions lost at its input. The
[diagnostic script](../scripts/diagnose_calibration.py) records score counts,
ties, slopes and source hashes; the notebook shows the per-seed values.

This is a diagnosis of the accepted method, not a replacement endpoint. Raw
ranking says nothing by itself about probability calibration. The
[v2 GMM calibration method](calibration-v2.md) and new metric-null contract are
available for future experiments, but neither produced the accepted S09 result.
Their synthetic tests establish software behavior, not real-data performance.
