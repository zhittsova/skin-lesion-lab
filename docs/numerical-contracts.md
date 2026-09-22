# Numerical and report contracts

New run summaries declare `metrics_version: 2` in both the run configuration and
summary. This makes a missing denominator distinguishable from a measured zero.
A predicted-positive count of zero, for example, gives null precision even when
recall is a defined zero.

## Metrics and exports

`compute_classification_metrics(..., metrics_version=2)` requires aligned,
nonempty binary targets and decisions, and finite probabilities in `[0,1]`.
Precision, sensitivity/recall and specificity return null when their denominator
is zero. F1 uses `2TP/(2TP+FP+FN)` directly: an all-negative prediction on a cohort
containing positives gives F1 zero; a cohort with only true negatives gives null.
ROC-AUC is null when the observed labels contain one class. Accuracy and Brier
score remain defined for every accepted nonempty input. Empty inputs fail.

JSON uses `null` and rejects NaN or infinity. CSV exports leave an undefined cell
empty. Console summaries and threshold-chart labels say `undefined`; plots leave
a gap rather than draw a zero bar. Arrays passed to plotting may use NaN for that
gap, but NaN is not a report value. Referral reports already use null retained
risk when all rows are referred.

## Reliability and ECE

`probability_report`, `compute_calibration_curve` and `compute_calibration_error`
share input validation and, for current metrics, one bin convention. Bin counts
must be positive Python integers, excluding booleans. Edges are generated with
`np.linspace(0, 1, n_bins + 1)`. Each bin includes its lower edge and excludes its
upper edge, except the final bin includes one. Empty JSON bins have null means
and fractions. ECE sums each nonempty bin's absolute probability/frequency gap,
weighted by its sample count.

Historical policy reports replayed with `metrics_version=1` retain their
original multiply-and-floor bin assignment. That explicit replay branch matters at floating-point internal bin
edges. All new reports, including new runs using policy v1, and the ECE helper use the
shared edge convention.

## Costs and historical replay

Costs must be finite and strictly positive. The formula threshold rescales costs
before addition and rejects settings whose threshold rounds to zero or one, or
whose FN/FP ratio cannot be represented as a finite positive float. Thus every
accepted formula rule predicts negative at probability zero and positive at one.
Comparisons use `>=` on the stored float threshold, including a positive decision
at an exact stored-threshold tie. A float cutoff is not an exact rational number.
The default 10:1 setting remains unchanged.

Empirical threshold selection compares error-count costs as exact rational
values of the supplied floats. Ties prefer recall, then specificity, then the
smallest threshold. The candidate just above the largest score permits an
all-negative rule. A 4,644-case finite enumeration independently verifies this
ordering, alongside extreme-cost rejection and rational risk controls.

A saved run with no `metrics_version` is explicitly treated as v1. Recomputing it
uses historical zero-division metrics and labels the output `metrics_version: 1`.
It does not rewrite the saved summary. This compatibility path is bounded to
validated, strict-JSON run artifacts; historical files containing NaN or infinity
are still rejected. New runs cannot request the legacy metric default through
the training CLI. Unknown versions or disagreement between run and summary fail.
Legacy single-class ROC-AUC can be NaN in the explicit v1 in-memory helper, but
cannot be serialized as a new report. Use v2 for new data and reports.

## Benchmark integration

The [benchmark implementation](../src/benchmark.py) uses metric v2 for new
endpoints, including precision, sensitivity, specificity, Brier score and
ROC-AUC. A saved report without a metric version is identified as v1 for
historical replay; accepted report values and hashes remain unchanged. In a
nonempty one-class bootstrap draw, undefined class rates, ROC-AUC, average
precision and PR-AUC are null because those comparisons require both classes.
Defined rates, Brier score, log loss and cost remain numeric. The original
comparison cohort still requires both classes.

The paired report uses the same sampled component indices for every model and
seed. If any constituent endpoint is undefined, its seed mean or contrast is
null. The report counts undefined draws and omits an interval when the point or
any draw is undefined. It does not drop those draws or turn null into zero, so a
missing denominator cannot appear as an improvement. The
[benchmark guide](benchmark.md) describes the comparison and saved report.
If F1 is added to benchmark endpoints later, it must use the direct count
formula above.
