"""Analytical and held-out isolation oracles for decision policies."""

import json
import unittest

import numpy as np
from src import bayes, calibration, uncertainty


class CalibrationTests(unittest.TestCase):
    def test_correct_each_pass_before_average_and_platt_constant_oracle(self):
        scores = np.array([[1 / 5], [4 / 5]])
        np.testing.assert_allclose(calibration._corrected_mean(scores, 4), [19 / 68])
        np.testing.assert_array_equal(
            calibration.correct_weighted_scores([0, 1], 1e308), [0, 1]
        )
        policy = calibration.fit_policy(
            [0, 0, 0, 1], [0.5] * 4, image_ids=["a", "b", "c", "d"], split_hash="x"
        )
        self.assertAlmostEqual(policy["calibrator"]["slope"], 0, places=7)
        self.assertAlmostEqual(policy["calibrator"]["intercept"], -np.log(3), places=6)
        np.testing.assert_allclose(
            calibration.apply_policy(policy, [0.5])["calibrated_probability"],
            [0.25],
            atol=1e-7,
        )

    def test_referral_requires_same_mc_pass_count(self):
        policy = calibration.fit_policy(
            [0, 1],
            [[0.1, 0.8], [0.2, 0.9]],
            image_ids=["a", "b"],
            split_hash="x",
            variances=[0.005, 0.005],
        )
        with self.assertRaisesRegex(ValueError, "pass count"):
            calibration.apply_policy(policy, [[0.1, 0.8]] * 3, variances=[0, 0])

    def test_weight_correction_and_cost_risks(self):
        np.testing.assert_allclose(
            calibration.correct_weighted_scores(np.array([0, 0.5, 0.75, 1]), 3),
            [0, 0.25, 0.5, 1],
        )
        # At p=.25, the positive and negative risks are both .75.
        np.testing.assert_array_equal(
            bayes.threshold_with_costs(np.array([0, 0.2, 0.25, 0.8, 1]), 3, 1),
            [0, 0, 1, 1, 1],
        )
        for bad in (0, -1, np.nan, np.inf):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                calibration.correct_weighted_scores(np.array([0.5]), bad)
            with self.subTest(cost=bad), self.assertRaises(ValueError):
                bayes.threshold_with_costs(np.array([0.5]), bad, 1)
        self.assertEqual(calibration.cost_threshold(1e308, 1e308), 0.5)

    def test_entropy_hand_cases_and_invalid_samples(self):
        result = uncertainty.summarize_mc_dropout_probabilities(
            np.array([[0, 0.5, 1], [1, 0.5, 1]])
        )
        np.testing.assert_allclose(
            result["predictive_entropy"], [np.log(2), np.log(2), 0], atol=1e-15
        )
        np.testing.assert_allclose(
            result["expected_entropy"], [0, np.log(2), 0], atol=1e-15
        )
        np.testing.assert_allclose(
            result["mutual_information"], [np.log(2), 0, 0], atol=1e-15
        )
        np.testing.assert_allclose(result["variance"], [0.5, 0, 0])
        for bad in (
            np.array([[np.nan], [0.5]]),
            np.array([[np.inf], [0.5]]),
            np.empty((2, 0)),
            np.array([[1.1], [0.5]]),
            np.array([[0.5]]),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                uncertainty.summarize_mc_dropout_probabilities(bad)
        for bad in ([np.nan], [np.inf], [-0.1], [1.1], []):
            with self.subTest(entropy=bad), self.assertRaises(ValueError):
                uncertainty.binary_entropy(np.array(bad))

    def test_exact_cost_tie_prefers_recall_before_rounding(self):
        scores = np.arange(1, 13) / 13
        labels = [1] + [0] * 10 + [1]
        threshold, rows = calibration.select_cost_threshold(labels, scores, 10, 1)
        self.assertEqual(threshold, 0)
        # A real one-ULP cost difference must not be treated as a tie.
        threshold, _ = calibration.select_cost_threshold(
            labels, scores, np.nextafter(10.0, 0.0), 1
        )
        self.assertEqual(threshold, 12 / 13)

    def test_exact_thresholds_include_all_negative(self):
        # A coarse .002 grid cannot isolate these two scores.
        cutoff, rows = calibration.select_cost_threshold([0, 1], [0.5001, 0.5002], 1, 1)
        self.assertEqual(cutoff, 0.5002)
        self.assertEqual(min(row["average_cost"] for row in rows), 0)
        cutoff, _ = calibration.select_cost_threshold([0, 0, 1], [1, 1, 1], 1, 10)
        self.assertGreater(cutoff, 1)
        # Equal costs favor the rule with greater sensitivity.
        cutoff, _ = calibration.select_cost_threshold([0, 1], [0.5, 0.5], 1, 1)
        self.assertEqual(cutoff, 0)

    def test_saved_policy_isolated_from_heldout_batch(self):
        policy = calibration.fit_policy(
            [0, 0, 1, 1],
            [0.1, 0.3, 0.6, 0.8],
            image_ids=["a", "b", "c", "d"],
            split_hash="frozen",
            weight=3,
            variances=[0, 0.01, 0.02, 0.03],
            cost_fn=3,
            cost_fp=1,
        )
        snapshot = json.dumps(policy, sort_keys=True, allow_nan=False)
        loaded = json.loads(snapshot)
        first = calibration.apply_policy(loaded, [0.3], variances=[0.01])
        larger = calibration.apply_policy(
            loaded, [0.3, 0.99, 0.01], variances=[0.01, 0.5, 0]
        )
        for column in ("calibrated_probability", "prediction", "review_recommended"):
            self.assertEqual(first[column][0], larger[column][0])
        calibration.policy_report(
            loaded, [0, 1, 0], [0.3, 0.99, 0.01], variances=[0.01, 0.5, 0]
        )
        calibration.policy_report(
            loaded, [1, 0, 1], [0.3, 0.99, 0.01], variances=[0.01, 0.5, 0]
        )
        self.assertEqual(json.dumps(loaded, sort_keys=True, allow_nan=False), snapshot)
        self.assertEqual(policy["fit"]["role"], "calibration")
        self.assertEqual(policy["referral"]["variance_cutoff"], 0.027)

    def test_proper_scores_reliability_and_empty_retained(self):
        scores = calibration.probability_report([0, 1], [0.25, 0.75], n_bins=2)
        self.assertAlmostEqual(scores["brier_score"], 1 / 16)
        self.assertAlmostEqual(scores["log_loss"], -np.log(0.75))
        self.assertEqual([b["count"] for b in scores["reliability_bins"]], [1, 1])
        policy = calibration.fit_policy(
            [0, 1], [0.5, 0.5], image_ids=["a", "b"], split_hash="x", variances=[0, 0]
        )
        result = calibration.policy_report(policy, [0, 1], [0.5, 0.5], variances=[0, 0])
        self.assertEqual(result["selective"]["referred_count"], 2)
        self.assertEqual(result["selective"]["retained_count"], 0)
        self.assertIsNone(result["selective"]["retained_error"])

    def test_invalid_fit_and_referral_inputs_fail(self):
        for labels, scores, ids in (
            ([0, 0], [0.2, 0.5], ["a", "b"]),
            ([0, 1], [0.2, np.nan], ["a", "b"]),
            ([0, 1], [0.2, 0.5], ["a", "a"]),
            ([0, 1], [0.2], ["a", "b"]),
            ([0, 2], [0.2, 0.5], ["a", "b"]),
        ):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                calibration.fit_policy(labels, scores, image_ids=ids, split_hash="x")
        for bad in ([np.nan], [-1], [0.6]):
            with self.subTest(variance=bad), self.assertRaises(ValueError):
                uncertainty.uncertainty_flags(
                    np.array([0.5]), np.array(bad), threshold=0.5, variance_cutoff=0.01
                )


if __name__ == "__main__":
    unittest.main()
