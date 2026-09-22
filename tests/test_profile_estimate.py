"""Arithmetic oracles for measured full-job timing estimates."""

import math
import unittest

from src.profile_estimate import estimate_job, pack_jobs


def phase(steady, startup):
    return {
        "steady_seconds_per_batch": steady,
        "startup_seconds_per_pass": startup,
    }


def profile():
    return {
        "train": phase(2, 3),
        "selection_loss": phase(1, 4),
        "selection_probability": phase(1.5, 5),
        "mc": phase(0.5, 2),
    }


class EstimateJobTests(unittest.TestCase):
    def test_counts_every_pipeline_pass_and_amortizes_startup_per_pass(self):
        estimate = estimate_job(
            profile(),
            {"train": 5, "selection": 3, "calibration": 2, "development": 4},
            batch_size=2,
            epochs=3,
            candidates=2,
            mc_samples=4,
        )
        expected = {
            "train": (3, 6, 36, 18, 54),
            "selection_loss": (2, 6, 12, 24, 36),
            "selection_probability": (2, 6, 18, 30, 48),
            "mc_calibration": (1, 4, 2, 8, 10),
            "mc_development": (2, 4, 4, 8, 12),
        }
        for name, values in expected.items():
            item = estimate["breakdown"][name]
            self.assertEqual(
                (
                    item["batches_per_pass"],
                    item["passes"],
                    item["steady_seconds"],
                    item["startup_seconds"],
                    item["total_seconds"],
                ),
                values,
            )
        self.assertEqual(estimate["total_seconds"], 160)
        self.assertFalse(estimate["concurrency_speedup_assumed"])
        self.assertTrue(estimate["excludes"])

    def test_default_schedule_matches_actual_invocation_counts(self):
        estimate = estimate_job(
            profile(),
            {
                "train": 5491,
                "selection": 1366,
                "calibration": 922,
                "development": 1395,
            },
        )
        breakdown = estimate["breakdown"]
        self.assertEqual(
            (breakdown["train"]["batches_per_pass"], breakdown["train"]["passes"]),
            (86, 40),
        )
        self.assertEqual(
            (
                breakdown["selection_loss"]["batches_per_pass"],
                breakdown["selection_loss"]["passes"],
            ),
            (22, 40),
        )
        self.assertEqual(
            (
                breakdown["selection_probability"]["batches_per_pass"],
                breakdown["selection_probability"]["passes"],
            ),
            (22, 40),
        )
        self.assertEqual(
            (
                breakdown["mc_calibration"]["batches_per_pass"],
                breakdown["mc_calibration"]["passes"],
            ),
            (15, 30),
        )
        self.assertEqual(
            (
                breakdown["mc_development"]["batches_per_pass"],
                breakdown["mc_development"]["passes"],
            ),
            (22, 30),
        )

    def test_profile_counts_and_controls_are_strictly_validated(self):
        counts = {"train": 5, "selection": 3, "calibration": 2, "development": 4}
        bad_profiles = []
        for key, value in [
            ("steady_seconds_per_batch", 0),
            ("steady_seconds_per_batch", math.nan),
            ("startup_seconds_per_pass", -1),
            ("startup_seconds_per_pass", math.inf),
        ]:
            bad = profile()
            bad["train"] = {**bad["train"], key: value}
            bad_profiles.append(bad)
        bad_profiles.extend(
            [
                {**profile(), "extra": phase(1, 0)},
                {key: value for key, value in profile().items() if key != "mc"},
                {**profile(), "train": {**profile()["train"], "extra": 1}},
            ]
        )
        for bad in bad_profiles:
            with self.subTest(profile=bad), self.assertRaises(ValueError):
                estimate_job(bad, counts)

        for bad_counts, kwargs in [
            ({**counts, "train": 0}, {}),
            ({**counts, "extra": 1}, {}),
            (counts, {"batch_size": True}),
            (counts, {"epochs": 0}),
            (counts, {"candidates": 1.5}),
            (counts, {"mc_samples": 1}),
        ]:
            with (
                self.subTest(counts=bad_counts, kwargs=kwargs),
                self.assertRaises(ValueError),
            ):
                estimate_job(profile(), bad_counts, **kwargs)


class PackJobsTests(unittest.TestCase):
    def test_packs_only_the_next_complete_jobs_in_supplied_order(self):
        result = pack_jobs(
            [
                {"id": "first", "seconds": 4},
                {"id": "too-long", "seconds": 8},
                {"id": "later-short", "seconds": 1},
            ],
            available_seconds=12,
            reserve_seconds=2,
        )
        self.assertEqual(result["selected"], [{"id": "first", "seconds": 4}])
        self.assertEqual(
            result["deferred"],
            [
                {"id": "too-long", "seconds": 8},
                {"id": "later-short", "seconds": 1},
            ],
        )
        self.assertEqual(result["used_seconds"], 4)
        self.assertEqual(result["remaining_seconds"], 6)
        self.assertEqual(result["reason"], "next_job_does_not_fit")
        self.assertFalse(result["concurrency_speedup_assumed"])

    def test_exact_fit_is_included_and_empty_queue_is_valid(self):
        exact = pack_jobs(
            [{"id": "job", "seconds": 8}],
            available_seconds=10,
            reserve_seconds=2,
        )
        self.assertEqual([job["id"] for job in exact["selected"]], ["job"])
        self.assertEqual(exact["remaining_seconds"], 0)
        self.assertIsNone(exact["reason"])
        self.assertEqual(
            pack_jobs([], 10, 2)["selected"],
            [],
        )

    def test_reserve_can_block_all_work_without_partial_execution(self):
        result = pack_jobs(
            [{"id": "job", "seconds": 1}],
            available_seconds=2,
            reserve_seconds=3,
        )
        self.assertEqual(result["selected"], [])
        self.assertEqual(result["deferred"], [{"id": "job", "seconds": 1}])
        self.assertEqual(result["usable_seconds"], 0)
        self.assertEqual(result["reason"], "reserve_exceeds_available")

    def test_rejects_unmeasured_concurrency_and_invalid_job_estimates(self):
        valid = [{"id": "job", "seconds": 1}]
        bad_jobs = [
            [{"id": "job", "seconds": 0}],
            [{"id": "job", "seconds": math.nan}],
            [{"id": "job", "seconds": 1, "extra": 1}],
            [{"id": "same", "seconds": 1}, {"id": "same", "seconds": 2}],
        ]
        for jobs in bad_jobs:
            with self.subTest(jobs=jobs), self.assertRaises(ValueError):
                pack_jobs(jobs, 10, 0)
        for kwargs in [
            {"available_seconds": math.inf, "reserve_seconds": 0},
            {"available_seconds": 10, "reserve_seconds": -1},
            {"available_seconds": 10, "reserve_seconds": 0, "concurrency": 2},
            {"available_seconds": 10, "reserve_seconds": 0, "concurrency": True},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                pack_jobs(valid, **kwargs)


if __name__ == "__main__":
    unittest.main()
