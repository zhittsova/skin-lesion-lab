# External evidence status: 22 September 2026

This is a post-outcome inspection note. The original S11 evaluation remains a
historical result: six saved runs cover HSV logistic regression and full,
unweighted EfficientNet-B0 at seeds 17, 42 and 73 on 884 eligible HIBA dermoscopy
images. The paired ROC-AUC difference is 0.0481, with a 95% interval from -0.0016
to 0.1091. That interval includes zero. These results do not establish clinical
usefulness or manuscript readiness.

The original release pinned the six fitted run records, their artifacts and six
source files. It omitted the external evaluator, reporting code and other local
execution dependencies. Its protocol has a recorded digest, but the original
execution gate did not check that protocol. The recorded source commit also
predates the external implementation. The original evidence therefore does not
establish a complete freeze of the executed program before outcomes were seen.

The correction preserves the original release, protocol, fitted artifacts and
external outputs. Historical inspection checks the listed bytes against an
explicit source snapshot and checks each actual family/seed mapping separately.
Saved-score reconstruction applies the original policy and metric contracts.
It does not run models, fit calibration or replace the accepted report.

New execution requires the [v2 release contract](external-release-v2.md). Its
complete source and dependency checks improve future enforcement. They do not
retroactively strengthen the original freeze or demonstrate performance for a
new method. A future evaluation needs a separately reviewed release and a record
of when its protocol and execution inputs were fixed.

The original local evidence is identified by these SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| S11 release | `ccc9dd39a63cc0847949da0cca58e13dacc3767fb683f0beda6536009dd3daf3` |
| S11 protocol | `edbde5faa236da6cddc93420eeb6384c407ae4aa9e0b9967ffd9e343fff20b00` |
| External report | `f592c4c0f8b9491f7db4ad3aedfc9d364a11c8d6fa9f3b57c442ba88e5a645d8` |

Numerical reconstruction supports the saved result, while the incomplete
original freeze limits its provenance claim. The manuscript decision remains
no-go. HIBA attribution remains Hospital Italiano de Buenos Aires, CC-BY,
[DOI 10.34970/587329](https://doi.org/10.34970/587329). The [development status](development-status-2026-09-22.md)
separately records the retrospective GMM diagnosis and future-policy limits.
