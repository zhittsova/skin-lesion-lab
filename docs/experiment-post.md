# Draft LinkedIn post

I built Skin Lesion Lab to compare a simple color-feature baseline with deep models for melanoma classification in dermoscopy images. The prespecified primary development comparison favored a fully fine-tuned EfficientNet-B0 over HSV logistic regression: mean ROC-AUC 0.891 versus 0.755 across three seeds.

I kept the models and cutoffs fixed for a separate HIBA cohort. There, the paired ROC-AUC difference was 0.0481, but its 95% component-bootstrap interval ran from -0.0016 to 0.1091. It includes zero. Under the saved cutoffs, mean EfficientNet sensitivity was only 0.4863. Referral also changed how many images the model retained without making the referred predictions correct.

The local development export matches the ISIC 2018 Task 3 training collection; its [data terms](https://challenge.isic-archive.com/data/) require attribution and noncommercial use. HIBA data are CC-BY, with attribution to Hospital Italiano de Buenos Aires ([DOI 10.34970/587329](https://doi.org/10.34970/587329)). This is a reproducible model-comparison portfolio, not a clinical triage system or a new medical method. The accompanying project write-up has the cohorts, error counts and limitations. Add its public link before posting. This post is a draft; publication needs owner approval.
