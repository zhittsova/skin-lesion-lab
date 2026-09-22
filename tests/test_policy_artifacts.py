"""Saved policy and summary must describe the same calibration-only decisions."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from src import calibration, run_contract


class PolicyArtifactTests(unittest.TestCase):
    def test_rehashed_summary_cannot_replace_saved_cutoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "models").mkdir()
            (root / "results").mkdir()
            ids = ["a", "b", "c", "d"]
            labels = np.array([0, 0, 1, 1])
            scores = np.array([0.1, 0.2, 0.7, 0.9])
            policy = calibration.fit_policy(
                labels, scores, image_ids=ids, split_hash="x"
            )
            (root / "models/decision_policy.json").write_text(json.dumps(policy))
            applied = calibration.apply_policy(policy, scores)
            frame = pd.DataFrame(
                {
                    "image_id": ids,
                    "target": labels,
                    "raw_score": scores,
                    "corrected_score": applied["corrected_score"],
                    "prob_melanoma": applied["calibrated_probability"],
                    "prediction": applied["prediction"],
                    "review_recommended": applied["review_recommended"],
                }
            )
            frame.to_csv(root / "results/calibration.csv", index=False)
            record = {
                "pipeline": "classical_logistic",
                "config": {"decision_policy_version": 1},
                "prediction_files": {"calibration": "results/calibration.csv"},
            }
            manifest = {"split_hash": "x", "partitions": {"calibration": ids}}
            summary = {
                "cost_matrix": {
                    "false_negative": 10.0,
                    "false_positive": 1.0,
                    "cost_threshold": policy["decision"]["threshold"],
                    "map_threshold": 0.5,
                }
            }
            summary_path = root / "results/metrics_summary.json"
            summary_path.write_text(json.dumps(summary))
            run_contract._check_decision_policy(root, record, manifest, frame)
            summary["cost_matrix"]["cost_threshold"] = 0.99
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "decision policy"):
                run_contract._check_decision_policy(root, record, manifest, frame)

    def test_fit_digest_and_canonical_threshold_are_verified(self):
        for defect in ("flipped_fit_labels", "noncanonical_threshold"):
            with (
                self.subTest(defect=defect),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                (root / "models").mkdir()
                (root / "results").mkdir()
                ids = ["a", "b", "c", "d"]
                labels = np.array([0, 0, 1, 1])
                scores = np.array([0.1, 0.2, 0.7, 0.9])
                fitted_labels = 1 - labels if defect == "flipped_fit_labels" else labels
                policy = calibration.fit_policy(
                    fitted_labels, scores, image_ids=ids, split_hash="x"
                )
                if defect == "noncanonical_threshold":
                    policy["decision"]["threshold"] = 0.5
                applied = calibration.apply_policy(policy, scores)
                (root / "models/decision_policy.json").write_text(json.dumps(policy))
                pd.DataFrame(
                    {
                        "image_id": ids,
                        "target": labels,
                        "raw_score": scores,
                        "prob_melanoma": applied["calibrated_probability"],
                        "corrected_score": applied["corrected_score"],
                        "prediction": applied["prediction"],
                        "review_recommended": applied["review_recommended"],
                    }
                ).to_csv(root / "results/calibration.csv", index=False)
                (root / "results/metrics_summary.json").write_text(
                    json.dumps(
                        {
                            "cost_matrix": {
                                "false_negative": 10.0,
                                "false_positive": 1.0,
                                "cost_threshold": policy["decision"]["threshold"],
                                "map_threshold": 0.5,
                            }
                        }
                    )
                )
                record = {
                    "pipeline": "classical_logistic",
                    "config": {"decision_policy_version": 1},
                    "prediction_files": {"calibration": "results/calibration.csv"},
                }
                manifest = {"split_hash": "x", "partitions": {"calibration": ids}}
                with self.assertRaisesRegex(ValueError, "decision policy"):
                    run_contract._check_decision_policy(root, record, manifest, None)
