"""Phase fits must expose startup costs and refuse unsupported extrapolations."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.profile_runner import (
    cache_projection,
    fit_phase_rates,
    warm_persistent_loaders,
)


class ProfileRunnerTests(unittest.TestCase):
    def test_two_lengths_recover_known_startup_and_batch_rate(self):
        rows = [
            {
                "phase": p,
                "batches": n,
                "repeat": repeat,
                "seconds_per_pass": 2 + n * 0.5,
            }
            for p in ["train", "selection_loss", "selection_probability", "mc"]
            for n in [6, 24]
            for repeat in range(2)
        ]
        result = fit_phase_rates(rows)
        self.assertEqual(
            result["rates"]["train"],
            {"steady_seconds_per_batch": 0.5, "startup_seconds_per_pass": 2.0},
        )
        self.assertEqual(result["quality"]["train"]["repeat_spread_seconds"], 0)

    def test_flat_or_missing_timing_does_not_produce_fake_positive_speed(self):
        for rows in (
            [{"phase": "train", "repeat": 0, "batches": 6, "seconds_per_pass": 1}],
            [
                {"phase": "train", "repeat": 0, "batches": n, "seconds_per_pass": 1}
                for n in [6, 24]
            ],
        ):
            with self.assertRaises(ValueError):
                fit_phase_rates(rows)

    def test_unstable_per_repeat_slopes_cannot_win_screening(self):
        rows = [
            {
                "phase": phase,
                "batches": n,
                "repeat": repeat,
                "seconds_per_pass": seconds,
            }
            for phase in ["train", "selection_loss", "selection_probability", "mc"]
            for repeat, small, large in [(0, 1, 51), (1, 100, 51)]
            for n, seconds in [(6, small), (24, large)]
        ]
        with self.assertRaisesRegex(ValueError, "unstable"):
            fit_phase_rates(rows)

    def test_persistent_worker_cold_start_is_observed_once_outside_phase_fit(self):
        samplers = [SimpleNamespace(count=100), SimpleNamespace(count=100)]
        with patch(
            "scripts.profile_runner.time.perf_counter",
            side_effect=[0, 5, 5, 6, 10, 17, 17, 19],
        ):
            observed = warm_persistent_loaders([[1, 2], [1, 2]], samplers, 3)
        self.assertEqual([s.count for s in samplers], [6, 6])
        self.assertEqual([r["extra_startup_seconds"] for r in observed], [4, 5])
        self.assertEqual([r["warm_first_batch_seconds"] for r in observed], [1, 2])

    def test_two_slots_reserve_two_jobs_before_launch(self):
        from scripts.profile_runner import DEFAULT_CONFIG, required_headroom

        one = required_headroom(DEFAULT_CONFIG, 1)
        two = required_headroom(DEFAULT_CONFIG, 2)
        self.assertEqual(two["host_free"] - one["host_free"], 3 * 1024**3)
        self.assertEqual(two["gpu_free"] - one["gpu_free"], 4 * 1024**3)
        self.assertEqual(two["disk_free"], DEFAULT_CONFIG["reserve_disk_bytes"])

    def test_full_cohort_cache_does_not_reuse_small_subset_memory(self):
        got = cache_projection(
            cache_bytes=100,
            unique_contents=10,
            full_images=1000,
            processes=2,
            workers=1,
        )
        self.assertEqual(got["pixel_bytes"], 40000)
        self.assertFalse(got["includes_object_or_model_overhead"])
