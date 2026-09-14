# Development evaluation protocol v1

This protocol applies to the melanoma-versus-selected-benign task under label
policy `melanoma-selected-benign-v1`. The local ISIC 2018 Task 3 export and all
previously inspected test results are development evidence. Repartitioning those
images does not make them an untouched confirmation cohort. No model-quality
result has been validated under this protocol yet.

## Cohort and allocation

Run `freeze_splits.py` once on the eligible cohort before fitting any comparator.
Every model loads that same manifest. The manifest records the source, label
policy, all normalized rows, image-content digests, a canonical checksum of every
metadata cell, connected groups, sorted image IDs, fractions, seed and hashes.
Changing metadata, content, linkage or eligibility invalidates it. Reordering
metadata rows or columns preserves it. Preparation reads the current image bytes;
a cached DataFrame alone cannot verify that a file has not changed.

The split seed is 42 and does not vary with model seeds. Fractions target linked
groups within each class: training 0.60, selection 0.15, calibration 0.10 and
development holdout 0.15. Start with one group of each class in each role, then
assign each remaining group to the role furthest below its fractional quota,
with role order breaking ties. Seeded permutation of sorted components determines
membership. Image fractions can differ because groups contain different numbers
of images. Fewer than four independent groups of either class fails allocation.
Do not discard patients or alter ratios to get a successful run.

Connect all shared lesion IDs, complete genuine patient IDs, image-content hashes
and supplied `duplicate_cluster_id` links transitively. The supported metadata
adapters currently reject byte-identical images before splitting; separately
normalized cohorts still link identical digests. Near-duplicate detection is not
implemented. Any known near-duplicate links must be supplied before freezing;
unknown links remain a limitation. Cross-source pooling requires a reviewed
adapter and duplicate audit before use.

Patient IDs must come from source metadata, never from lesion IDs or demographics.
An incomplete patient column fails; an entirely absent patient column allows only
a lesion-and-known-duplicate disjoint claim. The current allocator requires one
binary label per connected component. It rejects mixed-outcome patient groups,
which can be legitimate, as unsupported. It does not classify those patients as
mislabeled or silently exclude them. Supporting them needs a new allocation
protocol before fitting. Group-based smoke limits are not implemented: use a
separately prepared synthetic cohort, and never subsample a frozen run by image.

## Data use

| Role | Permitted use |
|---|---|
| Training | Fit normalization, feature transforms, class priors, loss weights and model parameters. GMM BIC selects components using this role only. |
| Selection | Select hyperparameters and the deep checkpoint. No fitting of normalization or calibration. |
| Calibration | After model/checkpoint selection, fit probability calibration and select decision/referral cutoffs. Do not revise the model from these scores. |
| Development holdout | Evaluate the frozen development policy once per declared run. No fitting, cutoff selection or adaptive exclusions. |
| Confirmation | Unavailable to these development entry points. Requires an untouched independent cohort and a separately frozen release. |

The current deep entry point routes checkpoint evaluation to selection and its
empirical cost cutoff to calibration. The classical entry point uses training BIC
and a prespecified illustrative 10:1 cost rule; it reserves calibration rows for
the later calibration work. Existing weighted CNN probabilities, MC dropout and
cost/referral methods still need their planned correctness checks. Passing the
split contract does not validate those methods.

## Planned comparison and budgets

Implement and verify the following matrix in the baseline sessions; do not run it
as a side effect of creating a manifest. Use model seeds 17, 42 and 73 with the
same split. Keep all seeds, including failures; never report only the best seed.

- Majority/prevalence baseline: fit the training prevalence; no search.
- HSV logistic regression: standardized training histograms, L2 regularization,
  `C` in {0.01, 0.1, 1, 10}, otherwise fixed solver settings recorded before runs.
- HSV GMM: components 1 through min(10, available class training rows), full
  covariance, training BIC, fixed regularization and convergence limits recorded
  before runs. BIC is not an estimate of held-out classification quality.
- Small CNN: learning rates {0.001, 0.0003}; dropout 0.3, weight decay 0.0001,
  128-pixel inputs, batch size 64, at most 20 epochs per candidate and seed.
- EfficientNet-B0: explicit pretrained weight version; compare head-only training
  with full fine-tuning, each at learning rates {0.001, 0.0003}; other settings and
  the 20-epoch cap match the CNN. Random frozen features are not a comparator.

Use unweighted loss as the primary deep setting. A positive-class-weighted loss
using training counts is one prespecified ablation with the same search budget.
Treat head-only and fully fine-tuned EfficientNet as separate selection strata,
and keep weighted and unweighted losses separate. The primary EfficientNet
comparator is full fine-tuning with unweighted loss; head-only and weighted runs
are exploratory ablations. Choose the highest selection ROC-AUC within each
family/stratum and seed. For deep models, keep the earliest checkpoint on an exact
tie within a run; ties between learning-rate candidates prefer 0.001, then 0.0003.
For logistic regression, tied candidates prefer smaller C. Evaluate each seed's selected
model; do not select a seed. Stop at the epoch/candidate cap. Do not stop a search
because development scores look favorable. Record elapsed time and hardware;
equal epoch caps do not imply equal compute.

S05/S06 must fix remaining solver, convergence, weight and transform settings in
run configurations before S09 starts. S07 must fix calibration method, referral
coverage and cutoff tie rules before evaluating the development holdout. These
are implementation gates, not permission to optimize settings from holdout
scores. Any material change after holdout inspection creates a new development
version and requires fresh independent confirmation data for final claims.

## Endpoints and uncertainty

The primary development endpoint is image-level ROC-AUC averaged over the three
prespecified model seeds. The primary paired comparison is fine-tuned pretrained
EfficientNet-B0 versus HSV logistic regression. Report every family and ablation;
other comparisons are exploratory. Do not treat multiple images of a lesion as
independent observations for uncertainty estimates.

Secondary endpoints are average precision (named separately from trapezoidal
PR-AUC), sensitivity, specificity, precision, Brier score, log loss and average
10:1 false-negative/false-positive cost. Report both class counts, group counts,
and referral coverage when implemented. The cost ratio is illustrative, not a
clinical utility estimate. Keep bin edges fixed at ten equal-width probability
bins for reliability summaries.

Use the manifest's connected component as the resampling unit. Use patient
components when available; otherwise label intervals as lesion/known-duplicate
cluster intervals. This does not establish independence between unknown patients.
For each of 2,000 bootstrap draws (seed 2026), sample components with replacement
within each binary class and include all images of each drawn component with its
multiplicity. Use the same draw for every model and every seed, calculate each
seed's metric and paired difference, then average across seeds within the draw.
Report the point estimate on original images and percentile 95% intervals from
those paired draws. Report between-seed range and standard deviation separately;
the bootstrap interval conditions on these seeds and is not a training-variance
interval. Class-stratified draws condition on observed class group counts, so
prevalence-dependent intervals do not cover deployment prevalence uncertainty.

Empty, one-class, missing or nonfinite predictions invalidate an endpoint rather
than becoming zero. Preserve failed runs and their cause. A failed seed blocks a
complete three-seed comparison; fixing a correctness defect requires rerunning
all affected configurations, not replacing an unfavorable seed. No performance
minimum is a software acceptance gate.

## Confirmation gate

Before confirmation, freeze the source commit, configurations, manifests, model
selection, calibration and referral rules, primary contrast, endpoints and
exclusions. Obtain a separate cohort whose outcomes have not informed any of
those choices, verify provenance and cross-cohort duplicate/patient links, and
run the frozen policy once. Neither training CLI accepts a confirmation-purpose
manifest. Confirmation requires a separate evaluation-only entry point in the
external-validation session. The current lesion data cannot establish a final
independent clinical claim.
