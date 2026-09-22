import unittest

import numpy as np
import torch
from src import bayes, deep, features, splitting, uncertainty


class BayesianPipelineTests(unittest.TestCase):
    def test_posterior_rows_sum_to_one(self):
        log_likelihoods = np.array([[-2.0, -1.0], [-1.5, -3.0], [-0.3, -0.4]])
        priors = np.array([0.8, 0.2])

        posteriors = bayes.compute_posterior_probabilities(log_likelihoods, priors)

        np.testing.assert_allclose(posteriors.sum(axis=1), np.ones(3))
        self.assertTrue(np.all(posteriors >= 0))
        self.assertTrue(np.all(posteriors <= 1))

    def test_cost_threshold_formula(self):
        probs = np.array([0.05, 0.1, 0.5])

        preds = bayes.threshold_with_costs(probs, cost_fn=10.0, cost_fp=1.0)

        np.testing.assert_array_equal(preds, np.array([0, 1, 1]))

    def test_lesion_split_has_no_overlap(self):
        lesion_ids = np.array([f"lesion_{i:03d}" for i in range(80) for _ in range(2)])
        labels = np.array([0] * 120 + [1] * 40)

        split_indices = splitting.split_dataset(
            lesion_ids,
            labels,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            random_state=1,
        )
        report = splitting.get_split_report(labels, split_indices, lesion_ids)

        self.assertTrue(report["leakage_free"])
        self.assertEqual(report["lesion_overlap_counts"]["train_val"], 0)
        self.assertEqual(report["lesion_overlap_counts"]["train_test"], 0)
        self.assertEqual(report["lesion_overlap_counts"]["val_test"], 0)

    def test_hsv_feature_dimension(self):
        self.assertEqual(features.get_feature_dimension(), 128)

    def test_mc_dropout_summary(self):
        stochastic_probs = np.array(
            [
                [0.1, 0.45, 0.9],
                [0.2, 0.55, 0.8],
                [0.15, 0.50, 0.85],
            ]
        )

        summary = uncertainty.summarize_mc_dropout_probabilities(stochastic_probs)
        flags = uncertainty.uncertainty_flags(
            summary["mean_prob_melanoma"],
            summary["variance"],
            threshold=0.5,
            variance_cutoff=0.01,
        )

        self.assertEqual(summary["mean_prob_melanoma"].shape, (3,))
        self.assertEqual(summary["variance"].shape, (3,))
        self.assertTrue(flags[1])

    def test_small_dropout_cnn_forward_shape(self):
        model = deep.SmallDropoutCnn(dropout=0.2)
        batch = torch.zeros((4, 3, 64, 64), dtype=torch.float32)

        logits = model(batch)

        self.assertEqual(tuple(logits.shape), (4,))


if __name__ == "__main__":
    unittest.main()
