"""Independent arithmetic used by the synthetic CLI gate."""

import unittest

from scripts.smoke_experiment import independent_metrics


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


if __name__ == "__main__":
    unittest.main()
