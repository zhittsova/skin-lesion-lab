# Draft LinkedIn post

Would a development ROC-AUC of 0.891 versus 0.755 make you expect the same model to win on a second cohort?

I built Skin Lesion Lab to compare an HSV color-feature baseline with deep models for melanoma classification in dermoscopy images. The prespecified development comparison favored fully fine-tuned, unweighted EfficientNet-B0 over HSV logistic regression across three seeds.

On 884 eligible HIBA images, using saved models and cutoffs, the paired AUC difference was 0.0481. Its 95% component-bootstrap interval ran from -0.0016 to 0.1091 and included zero. EfficientNet's mean sensitivity was 0.4863. Referral changed coverage without making referred predictions correct. The development ranking did not establish an external advantage or clinical usefulness.

A later review also found that the original release omitted execution code. Reconstructing the saved results supported the numbers, but could not establish a complete freeze before outcomes were seen. A stricter release gate now applies to future evaluations.

The GMM comparison had a different problem: old calibration clipping tied most tail scores. Its saved raw AUC averaged 0.7364 versus 0.5704 after calibration. That diagnosis leaves the original endpoint intact. A tested log-odds interface is available for a future experiment, without a real-data performance claim.

This portfolio establishes no new medical method, and the manuscript decision remains no-go. My takeaway is to inspect the interval, missed cases and freeze record together before interpreting a model ranking. The accompanying draft explains each with source evidence.

The development export matches ISIC 2018 Task 3 by counts and fields; the original download receipt is unavailable. Credit: ViDIR Group, Department of Dermatology, Medical University of Vienna; Tschandl, Rosendahl and Kittler, HAM10000 (2018), https://doi.org/10.1038/sdata.2018.161. ISIC challenge reference: Codella et al. (2018), https://arxiv.org/abs/1902.03368. Attribution and noncommercial terms: https://challenge.isic-archive.com/data/. HIBA: Hospital Italiano de Buenos Aires, CC-BY, https://doi.org/10.34970/587329.

Draft only. Add the approved public story link before posting.
