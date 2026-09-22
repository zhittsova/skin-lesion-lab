# Slide outline and source register

This six-slide owner-review draft asks whether the development advantage survives
a second cohort. Slide 5 answers with an interval that includes zero and an
incomplete original execution freeze. Slide 6 explains why the manuscript decision
remains no-go. The deck uses report-derived text and the original notebook figure.
There are no patient images, stock photographs or simulated model results.

The new export is `outputs/rectification-r07/export/skin-lesion-lab-story-v7-2026-09-22.pptx`.
The v6 deck, earlier exports and original captures remain historical artifacts.
Generated decks and previews remain ignored and unpublished.

| Slide | On-slide evidence | Source |
| --- | --- | --- |
| 1 | Development advantage and later external evidence | S09 development and S11 external reports, identified below. |
| 2 | 1,395 images, 176 melanoma, 1,219 benign, 1,037 known groups, 27 runs | S09 counts and embedded registry. The grouping notebook uses six generated records and constructed predictions only. |
| 3 | Primary AUC 0.891 versus 0.755, difference 0.1354 [0.0950, 0.1761]; separate GMM diagnosis | S09 primary contrast; [dated development diagnosis](development-status-2026-09-22.md). Full primary and GMM run IDs appear below. |
| 4 | Actual reliability capture; primary false negatives 18, 26, 11 of 176 | Accepted notebook cell 6, ID `8edb47106e1a46a883d545849b8ab81b`, using `efficientnet-full-unweighted-17`; S09 per-seed policy reports. |
| 5 | External AUC difference 0.0481 [-0.0016, 0.1091]; original code freeze incomplete | Six-run S11 report, [post-outcome inspection](external-status-2026-09-22.md). |
| 6 | Sensitivity 0.4863; 75, 124, 100 missed melanoma images; coverage 40.16%, 80.77%, 63.01% | S11 report and original no-go research memo. No clinical-use or fairness claim. |

## Exact run references

Copy identifiers literally from the accepted registries. The development report's
stratum names use underscores, but its run IDs use hyphens. The external registry
uses the same six primary IDs in its own cohort namespace.

| Role | Exact IDs |
| --- | --- |
| Primary development EfficientNet | `efficientnet-full-unweighted-17`, `efficientnet-full-unweighted-42`, `efficientnet-full-unweighted-73` |
| Primary development logistic | `logistic-17`, `logistic-42`, `logistic-73` |
| Retrospective development GMM diagnosis | `gmm-17`, `gmm-42`, `gmm-73` |
| External EfficientNet | `efficientnet-full-unweighted-17`, `efficientnet-full-unweighted-42`, `efficientnet-full-unweighted-73` |
| External logistic | `logistic-17`, `logistic-42`, `logistic-73` |

## Capture identity

The preserved source is the accepted S10 execution at
`outputs/notebooks/accepted-jzft_hkd/02-development-report.ipynb`.
Its cell 4 (ID `8dd0d8092fe74a7c96281538738b07e2`) supplies
`cell-4-comparison.png`, which prints all 27 registry IDs. That full nine-model
figure remains a separate capture because its labels are too small for projection.
Cell 6 supplies `cell-6-calibration.png`, embedded uncropped on slide 4. Both
captures are copied byte-for-byte to the new export directory. Fresh notebook
executions validate the current narrative separately; they do not replace these
historical captures. The deck uses a 1280 by 720 canvas. The reliability figure is
readable on screen, but its labels remain marginal for room projection.

The development notebook checks the accepted report's SHA-256, expected plan
digest and embedded registry. It does not open or validate the 27 fitted run
files. A separate artifact audit performed that check.

Report identities: development `benchmark-s09-recovery-20260921`, SHA-256
`0f0e898a1af36e05898692a95c9a3f2dfbb59b5c4bb8ad08c144a6e55fd529c7`;
external `external-s11-hiba-587329`, SHA-256
`f592c4c0f8b9491f7db4ad3aedfc9d364a11c8d6fa9f3b57c442ba88e5a645d8`.

## Speaker notes

### Slide 1

The development comparison asks whether fully fine-tuned, unweighted EfficientNet-B0 ranks melanoma images better than HSV logistic regression. Before the later cohort result, consider whether the development interval alone answers that question for another source. Slide 5 shows why it does not. This portfolio experiment supports no diagnostic or triage use. Sources: S09 development report, SHA-256 0f0e898a1af36e05898692a95c9a3f2dfbb59b5c4bb8ad08c144a6e55fd529c7; S11 external report, SHA-256 f592c4c0f8b9491f7db4ad3aedfc9d364a11c8d6fa9f3b57c442ba88e5a645d8. The external status note explains the incomplete original execution freeze.

### Slide 2

S09 counts: 1,395 images, 176 melanoma and 1,219 selected benign, in 1,037 known lesion/duplicate groups. Patient identifiers were unavailable. The matrix contains 27 completed runs across seeds 17, 42 and 73. The grouping notebook illustrates the principle with six generated records and constructed predictions, not image files. The local export matches ISIC 2018 Task 3 by counts and fields, but its original download receipt is unavailable. Collection: https://api.isic-archive.com/collections/66/. Attribution and CC-BY-NC terms: https://challenge.isic-archive.com/data/. Dataset credit: ViDIR Group, Department of Dermatology, Medical University of Vienna. Paper: Philipp Tschandl, Cliff Rosendahl and Harald Kittler, The HAM10000 dataset (Scientific Data, 2018), https://doi.org/10.1038/sdata.2018.161. Challenge paper: Noel Codella et al., Skin Lesion Analysis Toward Melanoma Detection 2018 (2018), https://arxiv.org/abs/1902.03368. No source images are redistributed.

### Slide 3

The prespecified primary comparison uses efficientnet-full-unweighted-17, efficientnet-full-unweighted-42, efficientnet-full-unweighted-73 against logistic-17, logistic-42, logistic-73. S09 mean AUCs are 0.8906 and 0.7552, rounded on slide to 0.891 and 0.755. The paired difference is 0.1354, interval [0.0950, 0.1761], from 2,000 shared class-stratified known-component draws conditional on three fitted seeds. The sensitivity difference is -0.0076 [-0.0720, 0.0552]. The original comparison capture is cell 4, ID 8dd0d8092fe74a7c96281538738b07e2, of the accepted S10 notebook. Its full chart carries all 27 run IDs. The notebook checks report bytes, plan digest and registry, not fitted run files. Separately, retrospective analysis of gmm-17, gmm-42, gmm-73 gives mean raw AUC 0.7364 and accepted fitted AUC 0.5704. Old calibration clipping tied most tail scores; 1,292, 1,291 and 1,220 of 1,395 scores lay outside its bounds. Positive slopes could not recover removed distinctions. This diagnosis leaves the primary comparison and accepted GMM endpoint unchanged. Raw ranking does not establish calibration. The v2 log-odds method has software tests and still needs prospective real-data evaluation. Sources: development-status-2026-09-22.md and calibration-v2.md.

### Slide 4

The unchanged image is the accepted S10 notebook's actual cell 6 PNG, cell ID 8edb47106e1a46a883d545849b8ab81b, for efficientnet-full-unweighted-17. It is an on-screen reliability illustration, with sparse-bin and projection limits. S09 policy reports for efficientnet-full-unweighted-17, efficientnet-full-unweighted-42 and efficientnet-full-unweighted-73 give false negatives 18, 26 and 11 of 176 melanoma images before referral. Mean Brier score changed from 0.0790 raw to 0.0755 fitted. Referral retained 47.4% to 75.7% of development images and left missed melanoma images. Referring a case does not make its prediction correct. The 10:1 error cost is illustrative, without a clinical utility estimate.

### Slide 5

S11 external report: 884 eligible HIBA dermoscopy images, 194 melanoma and 690 selected benign, in 370 linked components. The six external IDs are efficientnet-full-unweighted-17, efficientnet-full-unweighted-42, efficientnet-full-unweighted-73, logistic-17, logistic-42 and logistic-73. Saved models and policies were applied without external fitting or threshold selection. Mean AUCs are 0.7966 and 0.7485. The paired difference 0.0481 [-0.0016, 0.1091] uses 2,000 shared unstratified component draws conditional on three fitted seeds. The interval includes zero. The original release omitted evaluator/reporting code and other local execution dependencies, its recorded commit predates the external implementation, and its gate did not enforce the protocol digest. Post-outcome inspection checked actual fitted identities and reconstructed saved predictions and statistics. It cannot establish a complete freeze before outcomes were seen. Release schema 2 improves future enforcement only. Source: external-status-2026-09-22.md. HIBA (2019-2022): Hospital Italiano de Buenos Aires, CC-BY, DOI https://doi.org/10.34970/587329; release https://api.isic-archive.com/doi/hospital-italiano-de-buenos-aires-skin-lesions-images-2019-2022/.

### Slide 6

S11 report and research-decision.md: mean EfficientNet sensitivity 0.4863, versus logistic 0.4536. For efficientnet-full-unweighted-17, efficientnet-full-unweighted-42 and efficientnet-full-unweighted-73, false negatives are 75, 124 and 100 of 194 melanoma images before referral. Coverage is 40.16%, 80.77% and 63.01%, with 12, 87 and 46 false negatives still retained. The retrospective cohort is mostly light skin types. The overlap screen found zero exact matches and 160 perceptual candidate pairs cleared in the recorded visual review. The later audit reproduced the screen and review hashes without repeating every adjudication. That screen does not prove cross-source patient disjointness. The manuscript decision remains no-go. Neither calibration, referral nor numerical reconstruction establishes clinical usefulness or skin-tone fairness. Compare the primary interval with these missed cases and the freeze record before interpreting the ranking.
