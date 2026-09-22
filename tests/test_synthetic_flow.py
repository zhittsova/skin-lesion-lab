"""Independent arithmetic and device provenance used by the synthetic CLI gate."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.smoke_experiment import check_run, execute, independent_metrics


class IndependentMetricsTests(unittest.TestCase):
    def test_hand_computed_confusion_cost_and_brier(self):
        observed = independent_metrics(
            targets=[0, 0, 1, 1],
            probabilities=[0.1, 0.8, 0.4, 0.9],
            decisions=[0, 1, 0, 1],
            cost_fn=10,
            cost_fp=1,
        )
        self.assertEqual(
            {key: observed[key] for key in ("tn", "fp", "fn", "tp")},
            {"tn": 1, "fp": 1, "fn": 1, "tp": 1},
        )
        self.assertAlmostEqual(observed["brier_score"], 0.255)
        self.assertAlmostEqual(observed["average_cost"], 2.75)
        self.assertAlmostEqual(observed["roc_auc"], 0.75)


class DeepBackendContractTests(unittest.TestCase):
    def test_deep_run_rejects_an_observed_backend_mismatch_before_artifacts(self):
        manifest = {"split_hash": "frozen"}
        cases = [
            ("cuda", "cuda", "cpu"),
            ("cuda", "cpu", "cuda"),
            ("cpu", "cpu", "cuda"),
        ]
        for requested, configured, observed in cases:
            with self.subTest(
                requested=requested, configured=configured, observed=observed
            ):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    run = root / "runs" / "deep-smoke"
                    run.mkdir(parents=True)
                    (run / "run.json").write_text(
                        json.dumps(
                            {
                                "status": "completed",
                                "run_id": "deep-smoke",
                                "split_hash": "frozen",
                                "config": {"device": configured},
                                "environment": {"device": observed},
                            }
                        )
                    )
                    with self.assertRaisesRegex(
                        AssertionError, "requested or observed backend"
                    ):
                        check_run(
                            root,
                            "deep-smoke",
                            manifest,
                            deep=True,
                            requested_device=requested,
                        )

    def test_execute_rejects_unsupported_device_before_creating_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "device"):
                execute(root, device="mps")
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
