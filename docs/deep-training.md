# Deep training

The deep CLI supports a small CNN and EfficientNet-B0. All runs use the frozen
training, selection, calibration and development groups described in the
[evaluation protocol](evaluation-protocol.md).

## Training modes

The small CNN trains from random initialization. EfficientNet-B0 has three
supported configurations:

| Options | Training mode |
|---|---|
| `--pretrained` | Train the classifier head with a frozen pretrained backbone. |
| `--pretrained --fine-tune-backbone` | Fine-tune the entire pretrained model. |
| `--fine-tune-backbone` | Train the entire model from random initialization. |

EfficientNet rejects a randomly initialized frozen backbone. Pretrained runs
use the explicit torchvision `EfficientNet_B0_Weights.IMAGENET1K_V1` enum.
An uncached pretrained run needs download access. Offline tests exercise the
architecture and state behavior with synthetic weights; they do not establish
the quality of pretrained representations.

Head training keeps the backbone in evaluation mode, including its BatchNorm
running statistics and stochastic depth. Only the classifier parameters update.
Full training updates the backbone parameters and its BatchNorm statistics.

## Loss and selection

The default loss is unweighted binary cross-entropy with logits. Use
`--loss-strategy pos_weight` for the prespecified weighted ablation, where the
positive-class weight is the training negative count divided by the training
positive count. Class weights never use held-out rows.

Each run trains for the requested epoch budget. The checkpoint with the highest
selection ROC-AUC wins; an exact tie keeps the earliest epoch. Calibration and
development rows do not select the checkpoint. The saved checkpoint and training
metadata record the selection rule and selected epoch.

Weighting changes the interpretation of raw sigmoid scores.
[Calibration and decision rules](calibration.md) fit on the calibration split
and persist before development inference. Fitting does not establish deployment
calibration, and the illustrative 10:1 costs do not establish a clinical operating
point.

## Reproducibility

The CLI defaults to CPU. `--device auto` selects an available accelerator;
`--device cuda` or `--device mps` requires that device to be available. The locked
Linux environment installs CPU PyTorch. Accelerator execution needs its own
compatible environment and verification.

Each run records its seed, device, transforms, loss and model configuration.
The seed initializes Python, NumPy and PyTorch. Each data loader has a seeded
PyTorch generator; worker initialization seeds NumPy and Python from the worker's
PyTorch seed. Training, selection, calibration and development loaders use the
run seed plus 0, 1, 2 and 3, respectively.

Training transforms include augmentation; selection and inference use
deterministic transforms. Both architectures use ImageNet normalization. The
recorded transforms describe the actual pipeline, including its resize policy,
which differs from the pretrained weight enum's default transform.

The [run contract](run-contract.md) binds the checkpoint and training metadata
to their file hashes. Reload checks concern deterministic evaluation with dropout
disabled. MC dropout intentionally produces stochastic predictions and has a
separate validation scope.

CPU checkpoint tests use absolute tolerance `1e-6` and relative tolerance zero.
This tolerance does not establish accelerator or cross-device equivalence.

Synthetic gradient, state and tiny-data learning checks test implementation
correctness. They do not establish performance on lesion images or clinical
validity. The full comparison uses the protocol's seeds and budgets in a later
experiment session.
