# Future GMM calibration protocol v2

Policy v2 keeps a GMM score available before converting it to a probability. This
addresses a specific limitation of v1: its fixed probability clipping can turn
many distinct tail scores into the same calibration input. V2 is a new method for
future experiments. It does not replace the accepted experiment or establish an
improvement in calibration or discrimination.

## Input and fit

Enable it explicitly with `train_pipeline.py --model gmm
--decision-policy-version 2`, alongside the normal input arguments. The default
remains policy v1. Other model families reject this option. The split manifest and
its original protocol version stay unchanged; the policy identifies this method
as `gmm-log-odds-sigmoid-v2`.

The input is the posterior log odds

```text
s = log_likelihood(melanoma) - log_likelihood(benign)
    + log(training_prior_melanoma) - log(training_prior_benign)
```

Both priors must be finite, strictly positive and sum to one within `1e-12`.
Likelihoods and the resulting score must be finite float64 values. There is no
prior smoothing, probability clipping or probability-to-logit conversion here.
A likelihood difference that overflows fails explicitly.

The fit receives only calibration scores, IDs and labels. It checks them against
the exact calibration partition and split hash, and requires both classes and
complete, disjoint partitions. Training and selection still belong to their
original roles. Development labels never enter the fit or threshold selection.

Set `scale = max(1, max(abs(calibration_scores)))`, then fit a sigmoid to
`s / scale` using the same fixed `C=1`, `lbfgs`, L2 slope penalty, unpenalized
intercept, tolerance `1e-10` and 1,000-iteration limit. Scaling keeps extreme finite
calibration inputs manageable, but also changes the effective regularization.
It is part of the declared method, not a neutral implementation detail. A future
comparison needs to freeze this choice before evaluating its holdout.

The fitted probability is `sigmoid(slope * (s / scale) + intercept)`. The scale
is stored and reused unchanged at inference. A nonrepresentable affine result
fails. Cost threshold selection and the fixed 0.05 referral margin follow v1;
v2 currently accepts deterministic, unweighted GMM scores only.

## Saved scores and ranking

Producer tables retain four distinct quantities:

| Column | Meaning |
| --- | --- |
| `calibration_score` | Finite posterior log odds before sigmoid saturation |
| `raw_score`, `corrected_score` | The sigmoid of those log odds; weighting is absent |
| `prob_melanoma` | The fitted sigmoid probability used for decisions and proper scores |
| `ranking_score` | `sign(slope) * calibration_score`, or zero for a zero slope |

A positive slope preserves the input ranking, a negative slope reverses it, and
a zero slope makes the fitted function constant. Float64 sigmoid probabilities
can still tie at zero or one, or after rounding. The ranking column retains the
declared order independently of those probability ties. Existing `roc_auc`
metrics continue to use `prob_melanoma`. An analysis of `ranking_score` must name
that score explicitly instead of quietly substituting it into the same endpoint.

Policy v2 serializes its input kind, scale rule, fit digest, coefficients, ranking
rule and decision settings. Training declares the policy version in the run
configuration and summary. Run validation checks version agreement, calibration
input digest, calibration-derived scale, score replay, ranking and decisions.
`apply_policy` dispatches by version and rejects unsupported versions or score
contracts. [Strict external releases](external-release-v2.md) validate the fitted
run and its policy before accepting a release. External inference applies the
saved v2 policy and records both the finite input and ranking scores in its
schema-2 prediction table. This makes an opt-in GMM v2 run inspectable under the
same release checks; it does not add GMM to the primary two-family report or
measure its real-data performance.

## What remains historical

V1 inference keeps its original fixed `[1e-10, 1-1e-10]` clipping, coefficient
application, weighting and MC aggregation. Historical probability reports also retain
the original bin assignment under explicit metric-v1 replay. A stored synthetic policy tests
this without calling a fitter. The new metric semantics are separately versioned
in [the numerical contract](numerical-contracts.md).

`scripts/diagnose_calibration.py` reads completed GMM v1 runs and writes a new
report outside them. It reports raw and fitted AUC, tail counts, distinct scores,
tied pairs and saved slopes, with source hashes. It does not fit a calibrator.
The accepted three-seed diagnostic gives mean raw AUC 0.7363719 and fitted AUC
0.5704370. All three slopes are positive, while clipping collapses most inputs
into ties. This is an effect of the frozen method, not an arithmetic error in
its reported endpoint. Raw discrimination does not establish calibrated
probabilities, and the v2 tests do not establish future model performance.
