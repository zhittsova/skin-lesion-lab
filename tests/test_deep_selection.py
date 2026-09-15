"""Selection-contract tests for multi-candidate deep runs."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from src import splitting
from tests.test_splitting import cohort


class StopAfterDevelopmentRouting(Exception):
    pass


class DeepSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frame = cohort(40)
        self.manifest = splitting.create_manifest(self.frame)
        self.manifest_path = self.root / "splits.json"
        splitting.save_manifest(self.manifest, self.manifest_path)
        self.metadata_path = self.root / "metadata.csv"
        self.metadata_path.write_text("synthetic metadata\n")
        self.parts = splitting.manifest_indices(self.frame, self.manifest)
        self.ids = self.frame.isic_id.to_numpy()
        self.labels = self.frame.target.to_numpy()
        self.args = argparse.Namespace(
            metadata_path=self.metadata_path,
            images_dir=self.root,
            source="ham10000",
            split_manifest=self.manifest_path,
            results_dir=self.root / "results",
            models_dir=self.root / "models",
            runs_dir=self.root / "arrays",
            seed=17,
            architecture="small_cnn",
            image_size=32,
            batch_size=8,
            num_workers=0,
            dropout=0.3,
            pretrained=False,
            fine_tune_backbone=False,
            learning_rate=0.001,
            learning_rates=[0.001, 0.0003],
            weight_decay=0.0001,
            cost_fn=10.0,
            cost_fp=1.0,
            epochs=2,
            mc_samples=2,
            loss_strategy="unweighted",
            device="cpu",
        )

    def _loader(self, image_ids, labels, indices, *args, seed, **kwargs):
        role = next(
            role
            for role, expected in self.parts.items()
            if np.array_equal(indices, expected)
        )
        return {
            "role": role,
            "ids": image_ids[indices],
            "labels": labels[indices],
            "seed": seed,
        }

    def _run_until_development(self, scores_by_lr, expected_winner):
        import train_deep_pipeline as pipeline

        epochs_by_lr = {rate: 0 for rate in self.args.learning_rates}
        selection_calls = []
        inference_calls = []

        def build_model(**kwargs):
            return torch.nn.Linear(1, 1)

        def train(model, dataloader, optimizer, **kwargs):
            self.assertEqual(dataloader["role"], "train")
            self.assertEqual(dataloader["seed"], self.args.seed)
            rate = optimizer.param_groups[0]["lr"]
            epochs_by_lr[rate] += 1
            marker = (10 if rate == 0.001 else 3) + epochs_by_lr[rate]
            with torch.no_grad():
                model.weight.fill_(marker)
                model.bias.zero_()
            return 0.5

        def selection_predict(model, dataloader, device):
            self.assertEqual(dataloader["role"], "selection")
            self.assertEqual(dataloader["seed"], self.args.seed + 1)
            marker = int(model.weight.item())
            rate = 0.001 if marker >= 10 else 0.0003
            epoch = epochs_by_lr[rate]
            score = scores_by_lr[rate][epoch - 1]
            selection_calls.append((rate, epoch))
            return {
                "label": np.array([0, 1]),
                "probability": np.array([score, score]),
            }

        def metrics(y_true, y_prob, *args, **kwargs):
            return {
                "roc_auc": float(y_prob[0]),
                "recall": 0.5,
                "specificity": 0.5,
            }

        def mc(model, dataloader, **kwargs):
            marker = int(model.weight.item())
            inference_calls.append((dataloader["role"], marker))
            if dataloader["role"] == "development":
                raise StopAfterDevelopmentRouting
            probabilities = np.full((2, len(dataloader["labels"])), 0.5)
            return {
                "label": dataloader["labels"],
                "mean_probability": probabilities.mean(axis=0),
                "all_probabilities": probabilities,
                "image_id": dataloader["ids"].tolist(),
            }

        policy = {
            "decision": {"formula_threshold": 0.5},
            "referral": {"margin": 0.1},
        }
        with (
            patch.object(
                pipeline.data,
                "prepare_dataset",
                return_value=(self.frame, self.ids, self.labels),
            ),
            patch.object(pipeline, "make_loader", side_effect=self._loader),
            patch.object(
                pipeline.deep,
                "get_default_device",
                return_value=torch.device("cpu"),
            ),
            patch.object(pipeline.deep, "build_model", side_effect=build_model),
            patch.object(pipeline.deep, "train_one_epoch", side_effect=train),
            patch.object(pipeline.deep, "evaluate_loss", return_value=0.5),
            patch.object(
                pipeline.deep,
                "predict_probabilities",
                side_effect=selection_predict,
            ),
            patch.object(pipeline, "metrics_for_threshold", side_effect=metrics),
            patch.object(
                pipeline.deep,
                "predict_with_mc_dropout",
                side_effect=mc,
            ),
            patch.object(pipeline.calibration, "fit_policy", return_value=policy),
            patch.object(
                pipeline.calibration,
                "apply_policy",
                return_value={"calibrated_probability": np.full(8, 0.5)},
            ),
            patch.object(
                pipeline,
                "find_best_threshold_by_validation_cost",
                return_value=(0.5, __import__("pandas").DataFrame()),
            ),
            self.assertRaises(StopAfterDevelopmentRouting),
        ):
            pipeline._run(self.args)

        self.assertEqual(
            selection_calls,
            [
                (rate, epoch)
                for rate in self.args.learning_rates
                for epoch in range(1, self.args.epochs + 1)
            ],
        )
        expected_marker = (10 if expected_winner == 0.001 else 3) + 1
        self.assertEqual(
            inference_calls,
            [("calibration", expected_marker), ("development", expected_marker)],
        )
        return json.loads(
            (self.args.results_dir / "deep_candidate_search.json").read_text()
        )

    def test_candidates_use_selection_then_only_earliest_winner_is_evaluated(self):
        search = self._run_until_development(
            {0.001: [0.6, 0.7], 0.0003: [0.9, 0.9]},
            expected_winner=0.0003,
        )

        self.assertEqual(search["winner"]["learning_rate"], 0.0003)
        self.assertEqual(search["winner"]["epoch"], 1)
        self.assertEqual(
            [candidate["status"] for candidate in search["candidates"]],
            ["completed", "completed"],
        )
        checkpoints = [
            self.root / candidate["checkpoint"] for candidate in search["candidates"]
        ]
        self.assertTrue(all(path.is_file() for path in checkpoints))

    def test_learning_rate_tie_uses_protocol_order_not_cli_order(self):
        self.args.learning_rates = [0.0003, 0.001]
        search = self._run_until_development(
            {0.001: [0.75, 0.75], 0.0003: [0.75, 0.75]},
            expected_winner=0.001,
        )
        self.assertEqual(search["winner"]["learning_rate"], 0.001)
        self.assertEqual(search["winner"]["epoch"], 1)

    def test_invalid_learning_rate_candidates_are_rejected(self):
        import train_deep_pipeline as pipeline

        for candidates, message in (
            ([0.001, 0.0], "finite and positive"),
            ([0.001, float("nan")], "finite and positive"),
            ([0.001, 0.001], "unique"),
        ):
            with self.subTest(candidates=candidates):
                self.args.learning_rates = candidates
                with self.assertRaisesRegex(ValueError, message):
                    pipeline.validate_training_config(self.args)

    def test_candidate_failure_is_persisted_and_blocks_evaluation(self):
        import train_deep_pipeline as pipeline

        def train(model, dataloader, optimizer, **kwargs):
            if optimizer.param_groups[0]["lr"] == 0.0003:
                raise KeyboardInterrupt("candidate interrupted")
            return 0.5

        with (
            patch.object(
                pipeline.data,
                "prepare_dataset",
                return_value=(self.frame, self.ids, self.labels),
            ),
            patch.object(pipeline, "make_loader", side_effect=self._loader),
            patch.object(
                pipeline.deep,
                "get_default_device",
                return_value=torch.device("cpu"),
            ),
            patch.object(
                pipeline.deep,
                "build_model",
                side_effect=lambda **kwargs: torch.nn.Linear(1, 1),
            ),
            patch.object(pipeline.deep, "train_one_epoch", side_effect=train),
            patch.object(pipeline.deep, "evaluate_loss", return_value=0.5),
            patch.object(
                pipeline.deep,
                "predict_probabilities",
                return_value={
                    "label": np.array([0, 1]),
                    "probability": np.array([0.25, 0.75]),
                },
            ),
            patch.object(
                pipeline,
                "metrics_for_threshold",
                return_value={
                    "roc_auc": 0.75,
                    "recall": 0.5,
                    "specificity": 0.5,
                },
            ),
            patch.object(pipeline.deep, "predict_with_mc_dropout") as infer,
            self.assertRaisesRegex(KeyboardInterrupt, "candidate interrupted"),
        ):
            pipeline._run(self.args)

        infer.assert_not_called()
        search = json.loads(
            (self.args.results_dir / "deep_candidate_search.json").read_text()
        )
        self.assertEqual(search["candidates"][0]["status"], "completed")
        failed = search["candidates"][1]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failure"]["type"], "KeyboardInterrupt")
        self.assertEqual(failed["failure"]["message"], "candidate interrupted")
        self.assertIsNone(search["winner"])


if __name__ == "__main__":
    unittest.main()
