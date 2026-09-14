# Calibration and referral rules

Each run fits its decision policy on the frozen calibration partition after model
selection. `models/decision_policy.json` records the fitted coefficients, input
IDs, split identity, loss weight, decision threshold and referral cutoffs. The
pipelines apply this policy to development data. They do not refit it there.

## Score meanings

Prediction tables keep `raw_score`, `corrected_score`, and `prob_melanoma` separate.
The last field is the fitted probability used for decisions and proper scores.
A fitted calibrator can still generalize poorly; its name does not establish
calibration in a deployment population.

Positive class weighting in BCE changes its ideal optimum. For positive weight
`w` and weighted score `q`, the correction is `q / (q + w * (1 - q))`. This assumes
an ideal weighted-loss fit to the same population. It does not correct arbitrary
model error or prevalence shift. Deep runs use the actual training loss weight,
correct every MC pass, and then average. Unweighted training uses `w = 1`.
Classical scores also use `w = 1`.

The calibrator fits an unweighted sigmoid to the logit of the corrected score
(or corrected MC mean). Its fixed settings are L2 regularization on the slope,
`C = 1`, an unpenalized intercept, `lbfgs`, tolerance `1e-10`, and at most 1,000
iterations. Input probabilities are clipped to `[1e-10, 1 - 1e-10]` before the
logit. Both calibration classes are required, and nonconvergence fails the run.
No calibration hyperparameter search uses development data.

## Decision costs

With zero cost for correct decisions, conditional positive-decision risk is
`C_FP * (1 - p)` and negative-decision risk is `C_FN * p`. Their equality gives
`C_FP / (C_FP + C_FN)` when `p` is a calibrated posterior for the target population.
A tie predicts positive. The default 10:1 false-negative/false-positive costs are
illustrative; they are not clinical estimates. Changes in prevalence can change
calibration, precision and expected costs even if discrimination stays similar.

The primary saved threshold minimizes observed calibration cost on fitted
probabilities. Candidate thresholds are zero, each distinct calibration score,
and the next float above the largest score. Selection compares weighted error
counts using the exact binary values of the supplied costs, then divides for
reporting. Ties prefer higher recall, then
higher specificity, then the smallest candidate threshold. Comparisons use `>=`.
The final candidate permits an all-negative rule, including when a score is one.
Deep summaries also show the formula threshold and the fixed 0.5 operating point.
Calibration-set results include fitting and selection optimism.

## MC uncertainty and referral

MC inference enables dropout while keeping BatchNorm in evaluation mode. Every
pass must return the same unique IDs, paths and binary labels in the same order.
Empty passes, invalid pass counts and nonfinite logits fail. Inference restores
each module's original training flag on success and on errors.

The tables report raw MC sample variance (`ddof = 1`), predictive entropy,
expected entropy and mutual information in nats. MI is predictive entropy minus
mean pass entropy, with small negative roundoff clipped to zero. These quantities
all describe raw MC scores. They do not mix the fitted point probability with
raw pass entropy. Sample variance can reach 0.5 with two passes. Neither variance
nor MI alone establishes clinical usefulness or detection of unfamiliar inputs.

The deep referral rule flags raw sample variance at or above the calibration
90th percentile, or a fitted probability within 0.05 of the saved decision
threshold. Linear interpolation defines quantiles. The saved grid uses quantiles
0, 0.5, 0.8, 0.9 and 1 for development risk-coverage reporting. The MC pass count
must match the fitted policy. Classical referral uses the fixed probability
margin alone. Inference never computes a batch quantile.

Ties at a variance cutoff are referred. All-zero variances therefore refer every
row; the 90th percentile does not guarantee 10% referral. The probability margin
also applies to the all-negative threshold near one. Reports show actual counts,
coverage, retained error, retained class counts and retained average decision
cost. Empty retained sets have null risk. Referral cost and referral outcomes
are unknown, so retained cost is not a total system-cost estimate. The fixed
margin means the grid need not include an all-retained endpoint.

## Evidence artifacts

`results/calibration_report.json` compares raw, corrected and fitted outputs with
Brier score, log loss, and ten reliability bins including counts. Log loss clips
scores to `[1e-15, 1 - 1e-15]` for a finite numerical result. Reliability figures
plot the development fitted probabilities. Calibration rows describe the fitted
data; development rows describe the locked holdout. Risk-coverage rows reuse
saved calibration cutoffs.

`summarize_results.py` validates artifact checksums and prediction consistency,
verifies the calibration-input digest and canonical threshold, then recomputes
calibration and selective reports without the original images.
A run records its model artifact and split, and deep policies also record the
checkpoint checksum and MC pass count. Raw MC arrays remain available for
checking aggregation and uncertainty summaries.

Deterministic tests cover analytical cost, correction and entropy cases, policy
replay, heldout perturbation and inference state. Synthetic checks establish
these mechanics and measure finite-pass variation. Real model comparisons and
independent validation remain separate experiment work.
