"""Exercise actual entry-point routing with synthetic fit and selection spies."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from src import splitting
from tests.test_splitting import cohort


class RoutingComplete(Exception):
    pass


class PipelineIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frame = cohort(40)
        self.manifest = splitting.create_manifest(self.frame)
        self.path = self.root / "splits.json"
        splitting.save_manifest(self.manifest, self.path)
        (self.root / "metadata.csv").write_text("synthetic metadata\n")
        self.parts = splitting.manifest_indices(self.frame, self.manifest)
        self.ids = self.frame.isic_id.to_numpy()
        self.labels = self.frame.target.to_numpy().copy()
        self.args = argparse.Namespace(
            metadata_path=self.root / "metadata.csv",
            images_dir=self.root,
            source="ham10000",
            split_manifest=self.path,
            results_dir=self.root / "results",
            models_dir=self.root / "models",
            runs_dir=self.root / "runs",
            seed=17,
            architecture="small_cnn",
            image_size=32,
            batch_size=8,
            num_workers=0,
            dropout=0.3,
            pretrained=False,
            fine_tune_backbone=False,
            learning_rate=0.001,
            weight_decay=0.0001,
            cost_fn=10.0,
            cost_fp=1.0,
            epochs=1,
            mc_samples=2,
            max_train_images=None,
            max_val_images=None,
            max_test_images=None,
        )

    def test_classical_fit_uses_only_training_features_and_labels(self):
        import train_pipeline as pipeline

        for heldout_value in (777, 999):
            labels = self.labels.copy()
            for role in ("selection", "calibration", "development"):
                labels[self.parts[role]] = heldout_value
            seen = []

            def extract(images):
                return np.array(images, dtype=float).reshape(-1, 1)

            def load(path, target_size):
                index = list(self.ids).index(Path(path).stem)
                return index if index in self.parts["train"] else heldout_value + index

            def standardize(x):
                seen.append(x.copy())
                return x, np.array([0]), np.array([1])

            def fit(x, y, **kwargs):
                np.testing.assert_array_equal(x[:, 0], self.parts["train"])
                np.testing.assert_array_equal(y, self.labels[self.parts["train"]])
                self.assertEqual(len(seen), 1)
                np.testing.assert_array_equal(seen[0][:, 0], self.parts["train"])
                raise RoutingComplete

            with (
                patch.object(pipeline, "parse_args", return_value=self.args),
                patch.object(
                    pipeline.data,
                    "prepare_dataset",
                    return_value=(self.frame, self.ids, labels),
                ),
                patch.object(pipeline, "plots", Mock()),
                patch.object(pipeline.data, "load_image_hsv", side_effect=load),
                patch.object(
                    pipeline.features,
                    "compute_hsv_histograms_batch",
                    side_effect=extract,
                ),
                patch.object(
                    pipeline.features, "standardize_features", side_effect=standardize
                ),
                patch.object(pipeline.gmm, "train_class_gmms_auto", side_effect=fit),
                self.assertRaises(RoutingComplete),
            ):
                pipeline.main()

    def test_classical_stops_on_image_failure_without_dropping_members(self):
        import train_pipeline as pipeline

        with (
            patch.object(pipeline, "parse_args", return_value=self.args),
            patch.object(
                pipeline.data,
                "prepare_dataset",
                return_value=(self.frame, self.ids, self.labels),
            ),
            patch.object(pipeline, "plots", Mock()),
            patch.object(
                pipeline.data,
                "load_image_hsv",
                side_effect=FileNotFoundError("changed image"),
            ),
            self.assertRaisesRegex(FileNotFoundError, "changed image"),
        ):
            pipeline.main()

    def test_deep_checkpoint_and_threshold_use_distinct_roles(self):
        import train_deep_pipeline as pipeline

        for sentinel in (777, 999):
            labels = self.labels.copy()
            labels[self.parts["development"]] = sentinel
            calls = []

            def loader(image_ids, labels, indices, *args, **kwargs):
                return {
                    "ids": image_ids[indices],
                    "labels": labels[indices],
                    "indices": indices,
                }

            def check_loader(batch, role):
                np.testing.assert_array_equal(batch["indices"], self.parts[role])
                np.testing.assert_array_equal(
                    batch["labels"], self.labels[self.parts[role]]
                )
                calls.append(role)

            def train(**kwargs):
                check_loader(kwargs["dataloader"], "train")
                return 0.5

            def loss(model, batch, *args):
                check_loader(batch, "selection")
                return 0.5

            def predict(model, batch, *args):
                check_loader(batch, "selection")
                return {
                    "label": batch["labels"],
                    "probability": np.full(len(batch["labels"]), 0.5),
                }

            def mc(**kwargs):
                batch = kwargs["dataloader"]
                check_loader(batch, "calibration")
                return {
                    "label": batch["labels"],
                    "mean_probability": np.full(len(batch["labels"]), 0.5),
                    "all_probabilities": np.full((2, len(batch["labels"])), 0.5),
                }

            def threshold(y, probabilities, **kwargs):
                np.testing.assert_array_equal(y, self.labels[self.parts["calibration"]])
                self.assertEqual(
                    calls, ["train", "selection", "selection", "calibration"]
                )
                raise RoutingComplete

            with (
                patch.object(pipeline, "parse_args", return_value=self.args),
                patch.object(
                    pipeline.data,
                    "prepare_dataset",
                    return_value=(self.frame, self.ids, labels),
                ),
                patch.object(pipeline, "make_loader", side_effect=loader),
                patch.object(
                    pipeline.deep,
                    "get_default_device",
                    return_value=torch.device("cpu"),
                ),
                patch.object(
                    pipeline.deep,
                    "build_model",
                    side_effect=lambda **kw: torch.nn.Linear(1, 1),
                ),
                patch.object(pipeline.deep, "train_one_epoch", side_effect=train),
                patch.object(pipeline.deep, "evaluate_loss", side_effect=loss),
                patch.object(
                    pipeline.deep, "predict_probabilities", side_effect=predict
                ),
                patch.object(pipeline.deep, "predict_with_mc_dropout", side_effect=mc),
                patch.object(
                    pipeline,
                    "find_best_threshold_by_validation_cost",
                    side_effect=threshold,
                ),
                self.assertRaises(RoutingComplete),
            ):
                pipeline.main()

    def test_both_entrypoints_reject_confirmation_before_fit(self):
        import train_deep_pipeline
        import train_pipeline

        self.manifest["purpose"] = "confirmation"
        import json

        self.path.write_text(json.dumps(self.manifest))
        for pipeline in (train_pipeline, train_deep_pipeline):
            with (
                self.subTest(pipeline=pipeline.__name__),
                patch.object(pipeline, "parse_args", return_value=self.args),
                patch.object(
                    pipeline.data,
                    "prepare_dataset",
                    return_value=(self.frame, self.ids, self.labels),
                ),
                patch.object(pipeline, "plots", Mock()),
                self.assertRaisesRegex(ValueError, "manifest"),
            ):
                pipeline.main()

    def test_deep_restores_earliest_best_checkpoint_before_calibration(self):
        import train_deep_pipeline as pipeline

        self.args.epochs = 3
        model = torch.nn.Linear(1, 1)
        epochs = []

        def train(**kwargs):
            epochs.append(len(epochs) + 1)
            with torch.no_grad():
                model.weight.fill_(epochs[-1])
                model.bias.zero_()
            return 0.5

        def infer(**kwargs):
            self.assertEqual(epochs, [1, 2, 3])
            torch.testing.assert_close(model.weight, torch.ones_like(model.weight))
            checkpoints = list(self.args.runs_dir.glob("*/models/*.pt"))
            self.assertEqual(len(checkpoints), 1)
            checkpoint = torch.load(checkpoints[0], weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["selection_auc"], 0.75)
            raise RoutingComplete

        with (
            patch.object(pipeline, "parse_args", return_value=self.args),
            patch.object(
                pipeline.data,
                "prepare_dataset",
                return_value=(self.frame, self.ids, self.labels),
            ),
            patch.object(pipeline, "make_loader", return_value=Mock()),
            patch.object(pipeline.deep, "build_model", return_value=model),
            patch.object(pipeline.deep, "train_one_epoch", side_effect=train),
            patch.object(pipeline.deep, "evaluate_loss", return_value=0.5),
            patch.object(
                pipeline.deep,
                "predict_probabilities",
                return_value={"label": [0, 1], "probability": [0.25, 0.75]},
            ),
            patch.object(
                pipeline,
                "metrics_for_threshold",
                side_effect=[
                    {"roc_auc": score, "recall": 0.5, "specificity": 0.5}
                    for score in (0.75, 0.75, 0.5)
                ],
            ),
            patch.object(pipeline.deep, "predict_with_mc_dropout", side_effect=infer),
            self.assertRaises(RoutingComplete),
        ):
            pipeline.main()

    def test_random_frozen_configuration_records_failure_before_reading_data(self):
        import train_deep_pipeline as pipeline

        self.args.architecture = "efficientnet_b0"
        self.args.pretrained = False
        self.args.fine_tune_backbone = False
        with (
            patch.object(pipeline, "parse_args", return_value=self.args),
            patch.object(pipeline.data, "prepare_dataset") as prepare,
            self.assertRaisesRegex(ValueError, "pretrained|frozen"),
        ):
            pipeline.main()
        prepare.assert_not_called()
        records = list(self.args.runs_dir.glob("*/run.json"))
        self.assertEqual(len(records), 1)
        record = json.loads(records[0].read_text())
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["config"]["pretrained"])
        self.assertFalse(record["config"]["fine_tune_backbone"])

    def test_deep_cannot_reuse_checkpoint_without_selection_this_run(self):
        import train_deep_pipeline as pipeline

        for epochs, score in ((0, float("nan")), (1, float("nan")), (1, 0.5)):
            with self.subTest(epochs=epochs, score=score):
                self.args.epochs = epochs
                model = torch.nn.Linear(1, 1)
                self.args.models_dir.mkdir(exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "split_hash": "stale",
                        "cohort_hash": "stale",
                    },
                    self.args.models_dir / "small_cnn_mc_dropout.pt",
                )
                with (
                    patch.object(pipeline, "parse_args", return_value=self.args),
                    patch.object(
                        pipeline.data,
                        "prepare_dataset",
                        return_value=(self.frame, self.ids, self.labels),
                    ),
                    patch.object(pipeline, "make_loader", return_value=Mock()),
                    patch.object(
                        pipeline.deep,
                        "get_default_device",
                        return_value=torch.device("cpu"),
                    ),
                    patch.object(pipeline.deep, "build_model", return_value=model),
                    patch.object(pipeline.deep, "train_one_epoch", return_value=0.5),
                    patch.object(
                        pipeline.deep, "evaluate_loss", return_value=float("nan")
                    ),
                    patch.object(
                        pipeline.deep,
                        "predict_probabilities",
                        return_value={
                            "label": np.array([0, 1]),
                            "probability": np.array([0.5, 0.5]),
                        },
                    ),
                    patch.object(
                        pipeline,
                        "metrics_for_threshold",
                        return_value={
                            "roc_auc": score,
                            "recall": 0.5,
                            "specificity": 0.5,
                        },
                    ),
                    patch.object(
                        torch,
                        "load",
                        return_value={
                            "model_state_dict": model.state_dict(),
                            "split_hash": "stale",
                            "cohort_hash": "stale",
                        },
                    ),
                    patch.object(
                        pipeline.deep,
                        "predict_with_mc_dropout",
                        side_effect=RoutingComplete,
                    ) as inference,
                    self.assertRaisesRegex(ValueError, "epochs|checkpoint"),
                ):
                    pipeline.main()
                inference.assert_not_called()
