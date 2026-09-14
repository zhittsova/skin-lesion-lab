"""CPU checks for the S06 deep training contract."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from src import deep
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class DeepTrainingTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        cudnn_deterministic = torch.backends.cudnn.deterministic
        cudnn_benchmark = torch.backends.cudnn.benchmark
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.addCleanup(torch.use_deterministic_algorithms, deterministic)
        self.addCleanup(
            setattr, torch.backends.cudnn, "deterministic", cudnn_deterministic
        )
        self.addCleanup(setattr, torch.backends.cudnn, "benchmark", cudnn_benchmark)
        torch.set_num_threads(1)
        deep.set_seed(17)
        self.device = torch.device("cpu")

    def test_random_frozen_efficientnet_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pretrained|frozen"):
            deep.build_model("efficientnet_b0", pretrained=False, freeze_backbone=True)

    def test_frozen_backbone_keeps_parameters_and_batchnorm_buffers(self):
        real_constructor = models.efficientnet_b0
        with patch.object(deep.models, "efficientnet_b0") as constructor:
            constructor.side_effect = lambda **kwargs: real_constructor(weights=None)
            model = deep.build_model(
                "efficientnet_b0", pretrained=True, freeze_backbone=True, dropout=0
            )
            constructor.assert_called_once_with(
                weights=EfficientNet_B0_Weights.IMAGENET1K_V1
            )

        before = {
            key: value.detach().clone()
            for key, value in model.features.state_dict().items()
        }
        batch = (
            torch.randn(2, 3, 32, 32),
            torch.tensor([0.0, 1.0]),
            ["a", "b"],
            ["a.jpg", "b.jpg"],
        )
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.01)
        classifier_before = copy.deepcopy(model.classifier.state_dict())
        loss = deep.train_one_epoch(
            model, [batch], nn.BCEWithLogitsLoss(), optimizer, self.device
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        for key, value in model.features.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertTrue(
            any(
                not torch.equal(value, classifier_before[key])
                for key, value in model.classifier.state_dict().items()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.classifier.parameters()
            )
        )

    def test_both_architectures_forward_backward_and_reload(self):
        batch = (
            torch.randn(2, 3, 32, 32),
            torch.tensor([0.0, 1.0]),
            ["a", "b"],
            ["a.jpg", "b.jpg"],
        )
        for architecture in ("small_cnn", "efficientnet_b0"):
            with self.subTest(architecture=architecture):
                model = deep.build_model(
                    architecture,
                    pretrained=False,
                    freeze_backbone=architecture == "small_cnn",
                    dropout=0,
                )
                before = {
                    key: value.detach().clone()
                    for key, value in model.named_parameters()
                }
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
                loss = deep.train_one_epoch(
                    model, [batch], nn.BCEWithLogitsLoss(), optimizer, self.device
                )
                self.assertTrue(torch.isfinite(torch.tensor(loss)))
                changed = [
                    key
                    for key, value in model.named_parameters()
                    if not torch.equal(value, before[key])
                ]
                self.assertTrue(changed)
                if architecture == "efficientnet_b0":
                    self.assertTrue(
                        any(key.startswith("features.") for key in changed), changed
                    )
                self.assertTrue(
                    all(
                        parameter.grad is not None
                        and torch.isfinite(parameter.grad).all()
                        for parameter in model.parameters()
                    )
                )
                model.eval()
                with torch.no_grad():
                    expected = model(batch[0]).clone()
                with tempfile.TemporaryDirectory() as root:
                    path = Path(root) / "checkpoint.pt"
                    torch.save({"model_state_dict": model.state_dict()}, path)
                    reloaded = deep.build_model(
                        architecture,
                        pretrained=False,
                        freeze_backbone=architecture == "small_cnn",
                        dropout=0,
                    )
                    reloaded.load_state_dict(
                        torch.load(path, map_location="cpu", weights_only=True)[
                            "model_state_dict"
                        ]
                    )
                    reloaded.eval()
                    with torch.no_grad():
                        actual = reloaded(batch[0])
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
                print(
                    f"reload {architecture} max_abs_diff={float((actual - expected).abs().max()):.9g}"
                )

    def test_tiny_separable_data_reduces_loss(self):
        model = deep.build_model("small_cnn", dropout=0)
        labels = torch.tensor([0.0, 1.0] * 8)
        images = (2 * labels[:, None, None, None] - 1).expand(16, 3, 32, 32).clone()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0)
        batches = [
            (
                images[start : start + 8],
                labels[start : start + 8],
                list(range(start, start + 8)),
                [""] * 8,
            )
            for start in (0, 8)
        ]
        initial = deep.evaluate_loss(model, batches, criterion, self.device)
        for _ in range(20):
            deep.train_one_epoch(model, batches, criterion, optimizer, self.device)
        final = deep.evaluate_loss(model, batches, criterion, self.device)
        print(
            f"tiny initial_loss={initial:.9g} final_loss={final:.9g} ratio={final / initial:.9g}"
        )
        self.assertLessEqual(final, 0.6 * initial, (initial, final))

    def test_evaluation_loss_counts_samples_in_uneven_batches(self):
        class FirstPixel(nn.Module):
            def forward(self, images):
                return images[:, 0, 0, 0]

        inputs = torch.zeros(3, 3, 32, 32)
        inputs[2, 0, 0, 0] = 4
        batches = [
            (inputs[:2], torch.zeros(2), ["a", "b"], ["", ""]),
            (inputs[2:], torch.zeros(1), ["c"], [""]),
        ]
        expected = nn.BCEWithLogitsLoss()(torch.tensor([0.0, 0.0, 4.0]), torch.zeros(3))
        actual = deep.evaluate_loss(
            FirstPixel(), batches, nn.BCEWithLogitsLoss(), self.device
        )
        self.assertAlmostEqual(actual, float(expected), places=6)

    def test_invalid_batch_is_rejected_before_update(self):
        model = deep.build_model("small_cnn", dropout=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        before = {
            key: value.detach().clone() for key, value in model.named_parameters()
        }
        for images, labels in (
            (torch.zeros(2, 1, 32, 32), torch.zeros(2)),
            (torch.zeros(2, 3, 32, 32), torch.tensor([0.0, 2.0])),
            (torch.zeros(2, 3, 32, 32), torch.tensor([0.0, float("nan")])),
        ):
            with self.subTest(shape=tuple(images.shape), labels=labels.tolist()):
                with self.assertRaisesRegex(ValueError, "image|label"):
                    deep.train_one_epoch(
                        model,
                        [(images, labels, [], [])],
                        nn.BCEWithLogitsLoss(),
                        optimizer,
                        self.device,
                    )
                for key, value in model.named_parameters():
                    self.assertTrue(torch.equal(value, before[key]), key)


if __name__ == "__main__":
    unittest.main()
