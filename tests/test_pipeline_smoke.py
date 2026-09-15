"""Synthetic CPU integration of the shared allocation and output membership."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from src import calibration, data, gmm, run_contract, splitting


class SharedManifestSmokeTests(unittest.TestCase):
    def test_both_pipelines_and_summary_preserve_the_same_development_ids(self):
        import summarize_results
        import train_deep_pipeline
        import train_pipeline

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with (
            tempfile.TemporaryDirectory() as temporary,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            rows = []
            rng = np.random.default_rng(123)
            for label in (0, 1):
                for i in range(20):
                    image_id = f"I{label}_{i:03d}"
                    pixels = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(images / f"{image_id}.jpg")
                    rows.append(
                        {
                            "image_id": image_id,
                            "lesion_id": f"L{label}_{i // 2:03d}",
                            "dx": "mel" if label else "nv",
                        }
                    )
            metadata = root / "metadata.csv"
            pd.DataFrame(rows).to_csv(metadata, index=False)
            frame, _, _ = data.prepare_dataset(
                str(metadata),
                str(images),
                source="ham10000",
                attrition_path=root / "audit.json",
            )
            manifest = splitting.create_manifest(frame)
            path = root / "manifest.json"
            splitting.save_manifest(manifest, path)
            shared = [
                "--source",
                "ham10000",
                "--metadata-path",
                str(metadata),
                "--images-dir",
                str(images),
                "--split-manifest",
                str(path),
            ]
            original_fit = gmm.train_class_gmms_auto

            def tiny_fit(x, y, **kwargs):
                # Limit only this fixture's search cost; use the real fitter and scorer.
                return original_fit(
                    x, y, max_components=1, cv_type="diag", random_state=42
                )

            classical_root = root / "classical_runs"
            with (
                patch(
                    "sys.argv",
                    [
                        "train_pipeline.py",
                        *shared,
                        "--runs-dir",
                        str(classical_root),
                        "--run-id",
                        "classical-fixture",
                    ],
                ),
                patch.object(train_pipeline, "plots", Mock()),
                patch.object(gmm, "train_class_gmms_auto", side_effect=tiny_fit),
            ):
                train_pipeline.main()
            classical = classical_root / "classical-fixture"
            deep_root = root / "deep_runs"
            with (
                patch(
                    "sys.argv",
                    [
                        "train_deep_pipeline.py",
                        *shared,
                        "--runs-dir",
                        str(deep_root),
                        "--run-id",
                        "deep-fixture",
                        "--epochs",
                        "1",
                        "--mc-samples",
                        "2",
                        "--image-size",
                        "32",
                        "--batch-size",
                        "8",
                        "--seed",
                        "73",
                        "--learning-rates",
                        "0.001",
                        "0.0003",
                    ],
                ),
                patch.object(
                    train_deep_pipeline.deep,
                    "get_default_device",
                    return_value=torch.device("cpu"),
                ),
                patch.object(train_deep_pipeline, "plots", Mock()),
                patch.object(train_deep_pipeline, "plot_uncertainty_distribution"),
                patch.object(train_deep_pipeline, "plot_mean_vs_epistemic_uncertainty"),
            ):
                train_deep_pipeline.main()
            deep_results = deep_root / "deep-fixture"
            training = json.loads(
                (deep_results / "results" / "deep_training_metadata.json").read_text()
            )
            search = json.loads(
                (deep_results / "results/deep_candidate_search.json").read_text()
            )
            self.assertEqual(
                [c["learning_rate"] for c in search["candidates"]], [0.001, 0.0003]
            )
            self.assertTrue(
                all(c["status"] == "completed" for c in search["candidates"])
            )
            winner = max(search["candidates"], key=lambda c: c["selection_auc"])
            self.assertEqual(search["winner"]["learning_rate"], winner["learning_rate"])
            self.assertEqual(
                training["selection"]["learning_rate"], winner["learning_rate"]
            )
            record = json.loads((deep_results / "run.json").read_text())
            self.assertEqual(record["config"]["learning_rates"], [0.001, 0.0003])
            checkpoint_path = deep_results / training["checkpoint"]["path"]
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            self.assertEqual(
                training["checkpoint"]["sha256"], run_contract.sha256(checkpoint_path)
            )
            self.assertEqual(training["randomness"]["seed"], 73)
            self.assertEqual(training["device"], "cpu")
            self.assertEqual(training["loss"]["strategy"], "unweighted")
            self.assertIsNone(training["loss"]["pos_weight"])
            self.assertIsNone(training["weights"]["enum"])
            self.assertEqual(checkpoint["seed"], training["randomness"]["seed"])
            self.assertEqual(checkpoint["epoch"], training["selection"]["epoch"])
            self.assertEqual(
                checkpoint["selection_auc"], training["selection"]["score"]
            )
            self.assertEqual(checkpoint["split_hash"], manifest["split_hash"])
            for location, filename in (
                (classical, "predictions_development.csv"),
                (deep_results, "small_cnn_predictions_development.csv"),
            ):
                predictions = pd.read_csv(location / "results" / "tables" / filename)
                policy_path = location / "models" / "decision_policy.json"
                policy = json.loads(policy_path.read_text())
                self.assertEqual(
                    policy["fit"]["image_ids"], manifest["partitions"]["calibration"]
                )
                self.assertEqual(policy["fit"]["split_hash"], manifest["split_hash"])
                if location == classical:
                    scores = predictions.raw_score.to_numpy()
                    variances = None
                else:
                    scores = np.load(
                        location
                        / "arrays"
                        / "small_cnn_development_mc_probabilities.npy"
                    )
                    variances = scores.var(axis=0, ddof=1)
                applied = calibration.apply_policy(policy, scores, variances=variances)
                np.testing.assert_allclose(
                    predictions.prob_melanoma,
                    applied["calibrated_probability"],
                    rtol=1e-6,
                )
                np.testing.assert_array_equal(
                    predictions.prediction, applied["prediction"]
                )
                np.testing.assert_array_equal(
                    predictions.review_recommended, applied["review_recommended"]
                )

                self.assertEqual(
                    predictions.image_id.tolist(), manifest["partitions"]["development"]
                )
                self.assertTrue(predictions.group_id.notna().all())
                self.assertIn(
                    "prediction_map" if location == classical else "predictive_std",
                    predictions.columns,
                )
            for location, filename in (
                (classical, "metrics_summary.json"),
                (deep_results, "small_cnn_metrics_summary.json"),
            ):
                summary = json.loads((location / "results" / filename).read_text())
                self.assertEqual(summary["split"]["split_hash"], manifest["split_hash"])
                self.assertEqual(set(summary["split"]["splits"]), set(splitting.ROLES))
                self.assertNotIn(str(root), json.dumps(summary))
            metadata.unlink()
            shutil.rmtree(images)
            with patch(
                "sys.argv", ["summarize_results.py", "--run-dir", str(classical)]
            ):
                summarize_results.main()
            self.assertEqual(
                json.loads(
                    (classical / "results" / "metrics_summary.json").read_text()
                )["split"]["split_hash"],
                manifest["split_hash"],
            )
            for run in (classical, deep_results):
                record, predictions = run_contract.validate_run(run)
                self.assertEqual(record["status"], "completed")
                self.assertEqual(record["split_hash"], manifest["split_hash"])
                self.assertTrue(predictions.group_id.notna().all())
                summary_file = (
                    "metrics_summary.json"
                    if run == classical
                    else "small_cnn_metrics_summary.json"
                )
                reported = json.loads((run / "results" / summary_file).read_text())
                recomputed = run_contract.recompute_report(run)
                self.assertEqual(
                    recomputed["calibration_report"], reported["calibration_report"]
                )
                for role, points in recomputed["metrics"].items():
                    for point, scores in points.items():
                        for metric, value in scores.items():
                            self.assertAlmostEqual(
                                value, reported["metrics"][role][point][metric]
                            )
                    self.assertAlmostEqual(
                        recomputed["expected_calibration_error"][role],
                        reported["expected_calibration_error"][role],
                        delta=1e-6,
                    )
                if run == deep_results:
                    self.assertIn(
                        "results/deep_training_metadata.json", record["artifacts"]
                    )
                    self.assertEqual(
                        record["artifacts"][training["checkpoint"]["path"]],
                        training["checkpoint"]["sha256"],
                    )
                    selected = next(
                        row
                        for row in reported["training"]["history"]
                        if row["epoch"] == training["selection"]["epoch"]
                    )
                    self.assertEqual(
                        selected["selection_auc"], training["selection"]["score"]
                    )
                    for metric, value in recomputed["mc_dropout_uncertainty"].items():
                        self.assertAlmostEqual(
                            value, reported["mc_dropout_uncertainty"][metric]
                        )
            for location in (classical, deep_results):
                policy_file = location / "models" / "decision_policy.json"
                record_file = location / "run.json"
                original_policy, original_record = (
                    policy_file.read_bytes(),
                    record_file.read_bytes(),
                )
                changed_policy = json.loads(original_policy)
                changed_policy["decision"]["threshold"] = 1.0
                policy_file.write_text(json.dumps(changed_policy))
                changed_record = json.loads(original_record)
                changed_record["artifacts"]["models/decision_policy.json"] = (
                    run_contract.sha256(policy_file)
                )
                record_file.write_text(json.dumps(changed_record))
                with self.assertRaisesRegex(ValueError, "decision policy"):
                    run_contract.validate_run(location)
                policy_file.write_bytes(original_policy)
                record_file.write_bytes(original_record)

            producer = (
                deep_results
                / "results"
                / "tables"
                / "small_cnn_predictions_development.csv"
            )
            original = producer.read_bytes()
            record_path = deep_results / "run.json"
            record = json.loads(record_path.read_text())
            producer_key = "results/tables/small_cnn_predictions_development.csv"
            for field in ("predictive_std", "expected_entropy"):
                altered = pd.read_csv(producer, float_precision="round_trip")
                altered.loc[0, field] += 0.2
                altered.to_csv(producer, index=False)
                record["artifacts"][producer_key] = run_contract.sha256(producer)
                record_path.write_text(json.dumps(record))
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(ValueError, "deep uncertainty"),
                ):
                    run_contract.validate_run(deep_results)
                producer.write_bytes(original)
                record["artifacts"][producer_key] = run_contract.sha256(producer)
                record_path.write_text(json.dumps(record))

            array = deep_results / "arrays" / "small_cnn_development_y_true.npy"
            changed = np.load(array, allow_pickle=False)
            np.save(array, 1 - changed)
            record["artifacts"]["arrays/small_cnn_development_y_true.npy"] = (
                run_contract.sha256(array)
            )
            record_path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "deep array content"):
                run_contract.validate_run(deep_results)
