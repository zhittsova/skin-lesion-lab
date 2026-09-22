"""Version dispatch and finite score calibration without holdout fitting."""

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score
from src import bayes, calibration, splitting
from tests.test_splitting import cohort


class ScoreContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest = splitting.create_manifest(cohort(40))
        self.ids = self.manifest["partitions"]["calibration"]
        rows = {r["image_id"]: r for r in self.manifest["rows"]}
        self.labels = np.array([rows[i]["target"] for i in self.ids])
        self.scores = np.where(self.labels, 1000.0, -1000.0)

    def fit(self, **kwargs):
        options = dict(
            image_ids=self.ids,
            split_hash=self.manifest["split_hash"],
            schema_version=2,
            split_manifest=self.manifest,
        )
        options.update(kwargs)
        return calibration.fit_policy(self.labels, self.scores, **options)

    def test_legacy_synthetic_policy_replays_exactly_without_fit(self):
        saved = json.loads(
            (Path(__file__).parent / "fixtures/legacy-policy-v1.json").read_text()
        )
        with patch.object(
            calibration.LogisticRegression,
            "fit",
            side_effect=AssertionError("no fitting"),
        ):
            result = calibration.apply_policy(saved["policy"], saved["scores"])
        self.assertEqual(
            result["calibrated_probability"][0], result["calibrated_probability"][3]
        )
        self.assertEqual(
            result["calibrated_probability"][-1], result["calibrated_probability"][-2]
        )
        for key, expected in saved["expected"].items():
            np.testing.assert_array_equal(result[key], expected)

    def test_finite_gmm_log_odds_survive_probability_saturation(self):
        likelihoods = np.array(
            [[0.0, -1000.0], [0.0, -800.0], [-800.0, 0.0], [-1000.0, 0.0]]
        )
        score = bayes.compute_posterior_log_odds(likelihoods, [0.25, 0.75])
        np.testing.assert_allclose(
            score,
            [-1000 + np.log(3), -800 + np.log(3), 800 + np.log(3), 1000 + np.log(3)],
        )
        self.assertTrue(np.all(np.diff(score) > 0))
        np.testing.assert_array_equal(calibration.expit(score), [0.0, 0.0, 1.0, 1.0])
        for bad in (np.array([[np.inf, 1]]), np.array([1.0, 2.0]), np.empty((0, 2))):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                bayes.compute_posterior_log_odds(bad, [0.5, 0.5])
        for priors in ([0, 1], [0.5, np.nan], [0.3, 0.3]):
            with self.subTest(priors=priors), self.assertRaises(ValueError):
                bayes.compute_posterior_log_odds(likelihoods, priors)

    def test_v2_serialization_ranking_and_signed_slopes(self):
        policy = json.loads(json.dumps(self.fit(), allow_nan=False))
        self.assertEqual(policy["schema_version"], 2)
        self.assertEqual(policy["fit"]["image_ids"], self.ids)
        self.assertEqual(policy["score_transform"]["input_kind"], "finite_log_odds")
        probes = np.array(
            [-1e308, -1000.0, -800.0, -30.0, 0.0, 30.0, 800.0, 1000.0, 1e308]
        )
        for slope in (1.0, -1.0, 0.0):
            policy["calibrator"].update(slope=slope, intercept=0.0)
            result = calibration.apply_policy(policy, probes)
            np.testing.assert_array_equal(
                result["ranking_score"], np.sign(slope) * probes
            )
            self.assertTrue(np.isfinite(result["calibrated_probability"]).all())
            if slope:
                self.assertEqual(
                    roc_auc_score([0, 0, 1, 1], result["ranking_score"][[1, 2, 6, 7]]),
                    float(slope > 0),
                )
            else:
                np.testing.assert_array_equal(result["calibrated_probability"], 0.5)
        # Ordered logits below and above the old clip have distinct ranking values.
        result = calibration.apply_policy(self.fit(), [-40.0, -30.0, 30.0, 40.0])
        self.assertTrue(np.all(np.diff(result["ranking_score"]) > 0))
        self.scores[:] = 0
        constant = self.fit()
        self.assertEqual(constant["calibrator"]["slope"], 0.0)
        np.testing.assert_allclose(
            calibration.apply_policy(constant, [0.0])["calibrated_probability"],
            self.labels.mean(),
            atol=1e-6,
        )

    def test_v2_requires_calibration_identity_and_isolated_fit(self):
        for options in (
            dict(image_ids=list(reversed(self.ids))),
            dict(image_ids=[self.ids[0]] * len(self.ids)),
            dict(split_hash="wrong"),
            dict(split_manifest=None),
            dict(weight=2),
            dict(schema_version=999),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.fit(**options)
        original = self.labels.copy()
        self.labels = 1 - self.labels
        with self.assertRaisesRegex(ValueError, "calibration"):
            self.fit()
        self.labels = original
        policy = self.fit()
        serialized = json.dumps(policy, sort_keys=True)
        changed = copy.deepcopy(self.manifest)
        for row in changed["rows"]:
            if row["image_id"] in changed["partitions"]["development"]:
                row["target"] = 1 - row["target"]
        changed["split_hash"] = splitting.canonical_hash(
            {k: v for k, v in changed.items() if k != "split_hash"}
        )
        altered = self.fit(split_manifest=changed, split_hash=changed["split_hash"])
        self.assertEqual(policy["calibrator"], altered["calibrator"])
        self.assertEqual(policy["decision"], altered["decision"])
        overlap = copy.deepcopy(self.manifest)
        overlap["partitions"]["development"].append(self.ids[0])
        overlap["split_hash"] = splitting.canonical_hash(
            {k: v for k, v in overlap.items() if k != "split_hash"}
        )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.fit(split_manifest=overlap, split_hash=overlap["split_hash"])
        # Supplying heldout data cannot change policy application or trigger fit.
        with patch.object(
            calibration.LogisticRegression,
            "fit",
            side_effect=AssertionError("no refit"),
        ):
            one = calibration.apply_policy(policy, [-30.0])
            many = calibration.apply_policy(policy, [-30.0, 1e308])
            calibration.policy_report(policy, [1, 0], [-30.0, 1e308])
        self.assertEqual(
            one["calibrated_probability"][0], many["calibrated_probability"][0]
        )
        self.assertEqual(serialized, json.dumps(policy, sort_keys=True))
        for bad in ([np.nan], [np.inf], [], [[1, 2]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                calibration.apply_policy(policy, bad)
        for version in (0, 3, True, "2"):
            corrupted = copy.deepcopy(policy)
            corrupted["schema_version"] = version
            with self.subTest(version=version), self.assertRaises(ValueError):
                calibration.apply_policy(corrupted, [0.0])
        corrupted = copy.deepcopy(policy)
        corrupted["score_transform"]["input_kind"] = "probability"
        with self.assertRaises(ValueError):
            calibration.apply_policy(corrupted, [0.0])
