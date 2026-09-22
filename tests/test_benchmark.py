"""Independent arithmetic and cluster-resampling oracles for the benchmark."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from src import benchmark


def fixture():
    return pd.DataFrame(
        {
            "image_id": ["a", "b", "c", "d"],
            "group_id": ["negative", "negative", "positive", "positive"],
            "target": [0, 0, 1, 1],
            "prob_melanoma": [0.1, 0.4, 0.35, 0.8],
            "prediction": [0, 0, 0, 1],
        }
    )


class BenchmarkTests(unittest.TestCase):
    def test_metrics_match_hand_calculation_and_distinguish_pr_areas(self):
        f = fixture()
        m = benchmark.endpoints(f)
        self.assertEqual(m["roc_auc"], 0.75)
        self.assertAlmostEqual(m["average_precision"], 5 / 6)
        self.assertAlmostEqual(m["pr_auc"], 19 / 24)
        self.assertAlmostEqual(m["brier_score"], 0.158125)
        self.assertAlmostEqual(m["log_loss"], -np.log([0.9, 0.6, 0.35, 0.8]).mean())
        self.assertEqual(m["sensitivity"], 0.5)
        self.assertEqual(m["specificity"], 1)
        self.assertEqual(m["precision"], 1)
        self.assertEqual(m["average_cost"], 2.5)

    def test_invalid_inputs_and_undefined_precision(self):
        for f in [
            fixture().iloc[:0],
            fixture().assign(prob_melanoma=np.nan),
            fixture().assign(prediction=2),
        ]:
            with self.assertRaises(ValueError):
                benchmark.endpoints(f)
        self.assertIsNone(
            benchmark.endpoints(fixture().assign(prediction=0))["precision"]
        )

    def test_current_endpoints_preserve_defined_zero_and_one_class_nulls(self):
        f = fixture()
        for frame, expected in (
            (
                f.assign(prediction=0),
                {
                    "precision": None,
                    "sensitivity": 0.0,
                    "specificity": 1.0,
                    "average_cost": 5.0,
                },
            ),
            (
                f.iloc[:2].assign(prediction=0),
                {
                    "precision": None,
                    "sensitivity": None,
                    "specificity": 1.0,
                    "average_cost": 0.0,
                },
            ),
            (
                f.iloc[2:].assign(prediction=0),
                {
                    "precision": None,
                    "sensitivity": 0.0,
                    "specificity": None,
                    "average_cost": 10.0,
                },
            ),
            (
                f.iloc[:2].assign(prediction=1),
                {
                    "precision": 0.0,
                    "sensitivity": None,
                    "specificity": 0.0,
                    "average_cost": 1.0,
                },
            ),
        ):
            with self.subTest(
                targets=frame.target.tolist(), decisions=frame.prediction.tolist()
            ):
                got = benchmark.endpoints(frame)
                for name, value in expected.items():
                    self.assertEqual(got[name], value)
                if frame.target.nunique() == 1:
                    for name in ("roc_auc", "average_precision", "pr_auc"):
                        self.assertIsNone(got[name])
                self.assertAlmostEqual(
                    got["brier_score"],
                    sum((p - y) ** 2 for p, y in zip(frame.prob_melanoma, frame.target))
                    / len(frame),
                )
                json.dumps(got, allow_nan=False)

    def test_one_class_draws_count_nulls_without_changing_seed_mapping(self):
        f = fixture()
        models = {"a": {s: f for s in (17, 42, 73)}, "b": {s: f for s in (17, 42, 73)}}
        with patch("src.benchmark.group_draws", return_value=iter([np.array([0, 1])])):
            report = benchmark.paired_report(models, reference="b", draws=1)
        self.assertEqual(report["metrics_version"], 2)
        self.assertEqual(report["models"]["a"]["roc_auc"]["undefined_draws"], 1)
        self.assertIsNone(report["models"]["a"]["roc_auc"]["interval"])
        self.assertEqual(report["models"]["a"]["specificity"]["undefined_draws"], 0)
        self.assertEqual(report["models"]["a"]["specificity"]["interval"], [1, 1])
        self.assertEqual(report["differences"]["a"]["roc_auc"]["undefined_draws"], 1)
        self.assertEqual(report["differences"]["a"]["specificity"]["interval"], [0, 0])
        json.dumps(report, allow_nan=False)

    def test_legacy_endpoint_replay_is_explicit(self):
        current = benchmark.endpoints(fixture().assign(prediction=0))
        legacy = benchmark.endpoints(fixture().assign(prediction=0), metrics_version=1)
        self.assertIsNone(current["precision"])
        self.assertIsNone(legacy["precision"])
        self.assertEqual(legacy["roc_auc"], 0.75)
        with self.assertRaises(ValueError):
            benchmark.endpoints(fixture().iloc[:2], metrics_version=1)
        with self.assertRaises(ValueError):
            benchmark.endpoints(fixture(), metrics_version=3)
        self.assertEqual(benchmark.report_metrics_version({}), 1)
        self.assertEqual(benchmark.report_metrics_version({"metrics_version": 2}), 2)
        with self.assertRaises(ValueError):
            benchmark.report_metrics_version({"metrics_version": True})
        replay = benchmark.paired_report(
            {"a": {s: fixture() for s in (17, 42, 73)}},
            reference="a",
            draws=2,
            metrics_version=1,
        )
        self.assertEqual(replay["metrics_version"], 1)
        self.assertEqual(replay["models"]["a"]["roc_auc"]["estimate"], 0.75)

    def test_cluster_draws_keep_all_images_with_multiplicity(self):
        f = fixture()
        f["group_id"] = ["n1", "n2", "p1", "p1"]
        oracle = np.random.default_rng(2026)
        draws = list(benchmark.group_draws(f, draws=12, seed=2026))
        for indices in draws:
            neg = oracle.choice(["n1", "n2"], 2, replace=True)
            pos = oracle.choice(["p1"], 1, replace=True)
            expected = np.concatenate(
                [np.flatnonzero(f.group_id == g) for g in [*neg, *pos]]
            )
            np.testing.assert_array_equal(indices, expected)
            self.assertEqual(list(indices).count(2), list(indices).count(3))
            self.assertEqual(set(f.iloc[indices].target), {0, 1})
        with self.assertRaises(ValueError):
            list(benchmark.group_draws(f.iloc[:2], draws=12))
        with self.assertRaises(ValueError):
            list(benchmark.group_draws(f.assign(group_id="mixed"), draws=12))

    def test_shared_draws_mean_seed_metrics_and_separate_variability(self):
        a = {s: fixture() for s in (17, 42, 73)}
        a[73] = fixture().assign(prob_melanoma=[0.9, 0.6, 0.65, 0.2])
        # Seed AUCs .75, .75, .25: averaging probabilities gives a different AUC.
        r = benchmark.paired_report({"a": a, "b": a}, reference="b", draws=12)
        m = r["models"]["a"]["roc_auc"]
        self.assertAlmostEqual(m["estimate"], 7 / 12)
        self.assertEqual(m["seed_range"], [0.25, 0.75])
        self.assertAlmostEqual(m["seed_sd"], np.std([0.75, 0.75, 0.25], ddof=1))
        self.assertEqual(r["differences"]["a"]["roc_auc"]["interval"], [0, 0])
        # Each class has one cluster: cluster draws always reproduce the cohort.
        self.assertEqual(m["interval"], [m["estimate"], m["estimate"]])
        permuted = {s: f.iloc[::-1] for s, f in a.items()}
        self.assertEqual(
            r, benchmark.paired_report({"a": permuted, "b": a}, reference="b", draws=12)
        )

    def test_missing_seed_or_misaligned_identity_fails(self):
        a = {s: fixture() for s in (17, 42, 73)}
        for bad in [
            {17: fixture()},
            {**a, 73: fixture().assign(target=[1, 0, 1, 1])},
            {**a, 42: fixture().assign(image_id="duplicate")},
        ]:
            with self.assertRaises(ValueError):
                benchmark.paired_report({"a": a, "b": bad}, reference="a", draws=12)

    def test_undefined_resamples_are_counted_not_silently_dropped(self):
        f = fixture().assign(group_id=["n1", "n2", "p1", "p2"])
        a = {s: f for s in (17, 42, 73)}
        r = benchmark.paired_report({"a": a}, reference="a", draws=100)
        m = r["models"]["a"]["precision"]
        self.assertIsNone(m["interval"])
        self.assertGreater(m["undefined_draws"], 0)
        self.assertEqual(m["estimate"], 1)


class BenchmarkPlanTests(unittest.TestCase):
    def test_fixed_schedule_accounts_for_every_seed_and_candidate(self):
        jobs = benchmark.planned_runs()
        self.assertEqual(len(jobs), 27)
        self.assertEqual(len({j["run_id"] for j in jobs}), 27)
        self.assertEqual(sum(j["candidate_count"] for j in jobs), 45)
        self.assertEqual(
            sum(j["candidate_count"] * j["config"].get("epochs", 0) for j in jobs), 720
        )
        self.assertEqual({j["seed"] for j in jobs}, {17, 42, 73})
        for job in jobs:
            if job["pipeline"] == "deep":
                self.assertEqual(job["config"]["learning_rates"], [0.001, 0.0003])
                self.assertEqual(job["config"]["mc_samples"], 30)

    def test_argument_list_preserves_paths_and_uses_declared_grid(self):
        job = next(
            j
            for j in benchmark.planned_runs()
            if j["stratum"] == "efficientnet_full_unweighted"
        )
        command = benchmark.training_command(
            job,
            python="/python",
            metadata="/data/a b.csv",
            images="/data/images",
            manifest="/data/m.json",
            runs="/out",
            source="isic2018_task3",
            device="cpu",
        )
        self.assertEqual(command[command.index("--metadata-path") + 1], "/data/a b.csv")
        self.assertIn("--pretrained", command)
        self.assertIn("--fine-tune-backbone", command)
        self.assertEqual(
            command[
                command.index("--learning-rates") + 1 : command.index(
                    "--learning-rates"
                )
                + 3
            ],
            ["0.001", "0.0003"],
        )


class BenchmarkReviewTests(unittest.TestCase):
    def test_log_loss_matches_existing_boundary_convention(self):
        from src import calibration

        f = fixture().assign(prob_melanoma=[0.0, 1.0, 0.0, 1.0])
        expected = calibration.probability_report(f.target, f.prob_melanoma)["log_loss"]
        self.assertEqual(benchmark.endpoints(f)["log_loss"], expected)


class PairedOracleTests(unittest.TestCase):
    def test_nondegenerate_intervals_match_direct_pair_count_auc(self):
        f = pd.DataFrame(
            {
                "image_id": list("abcdef"),
                "group_id": ["n1", "n1", "n2", "p1", "p2", "p2"],
                "target": [0, 0, 0, 1, 1, 1],
                "prediction": [0, 0, 1, 0, 1, 1],
            }
        )
        scores = {
            17: [0.1, 0.4, 0.6, 0.3, 0.7, 0.9],
            42: [0.4, 0.2, 0.1, 0.8, 0.6, 0.3],
            73: [0.9, 0.6, 0.7, 0.1, 0.8, 0.3],
        }
        a = {s: f.assign(prob_melanoma=p) for s, p in scores.items()}
        b = {s: f.assign(prob_melanoma=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) for s in scores}

        def direct_auc(frame):
            pos = frame.loc[frame.target == 1, "prob_melanoma"].to_numpy()[:, None]
            neg = frame.loc[frame.target == 0, "prob_melanoma"].to_numpy()[None, :]
            return np.mean((pos > neg) + 0.5 * (pos == neg))

        rng = np.random.default_rng(2026)
        values, differences = [], []
        for _ in range(61):
            groups = [*rng.choice(["n1", "n2"], 2), *rng.choice(["p1", "p2"], 2)]
            indices = np.concatenate([np.flatnonzero(f.group_id == g) for g in groups])
            values.append(np.mean([direct_auc(a[s].iloc[indices]) for s in scores]))
            differences.append(
                np.mean(
                    [
                        direct_auc(a[s].iloc[indices]) - direct_auc(b[s].iloc[indices])
                        for s in scores
                    ]
                )
            )
        report = benchmark.paired_report({"a": a, "b": b}, reference="b", draws=61)
        np.testing.assert_allclose(
            report["models"]["a"]["roc_auc"]["interval"],
            np.quantile(values, [0.025, 0.975]),
        )
        np.testing.assert_allclose(
            report["differences"]["a"]["roc_auc"]["interval"],
            np.quantile(differences, [0.025, 0.975]),
        )


if __name__ == "__main__":
    unittest.main()
