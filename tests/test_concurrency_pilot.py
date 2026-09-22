"""Concurrency claims require overlap and measured aggregate throughput."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.concurrency_pilot import capacity_reason, summarize_group, summarize_trials


class ConcurrencyPilotTests(unittest.TestCase):
    def test_cpu_oversubscription_is_explicit_but_memory_reserves_hold(self):
        cfg = {
            "threads_per_job": 1,
            "loader_workers": 0,
            "allow_cpu_oversubscription": True,
            "reserve_host_bytes": 2,
            "reserve_device_bytes": 1,
            "reserve_disk_bytes": 1,
        }
        available = {"cpu_count": 2, "host_free": 12, "gpu_free": 12, "disk_free": 20}
        self.assertIsNone(capacity_reason(cfg, available, 3, 3, 3))
        self.assertEqual(capacity_reason(cfg, available, 4, 3, 3), "host_memory")
        self.assertEqual(
            capacity_reason(
                {**cfg, "allow_cpu_oversubscription": False}, available, 3, 3, 3
            ),
            "cpu_count",
        )
        self.assertEqual(
            capacity_reason(cfg, {**available, "gpu_free": 8}, 3, 3, 3), "device_memory"
        )

    def test_nonoverlapping_jobs_cannot_claim_parallel_speedup(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            summarize_group([self.sample(0, 10), self.sample(11, 21)])
        measured = summarize_group([self.sample(0, 10), self.sample(1, 11)])
        self.assertEqual(measured["elapsed_seconds"], 11)
        self.assertEqual(measured["overlap_seconds"], 9)
        self.assertAlmostEqual(measured["pilot_workloads_per_second"], 2 / 11)

    def test_two_jobs_sharing_gpu_need_observed_speedup(self):
        trials = [
            {"slots": 1, "measurement": {"pilot_workloads_per_second": 0.1}},
            {"slots": 1, "measurement": {"pilot_workloads_per_second": 0.1}},
            {"slots": 2, "measurement": {"pilot_workloads_per_second": 0.15}},
            {"slots": 2, "measurement": {"pilot_workloads_per_second": 0.16}},
            {"slots": 3, "measurement": {"pilot_workloads_per_second": 0.09}},
        ]
        result = summarize_trials(trials, repeats=2)
        self.assertEqual(result["best_observed_slots"], 2)
        self.assertAlmostEqual(result["levels"]["2"]["speedup_vs_one"], 1.55)
        self.assertFalse(result["levels"]["3"]["enough_repeats"])
        self.assertFalse(result["full_run_authorized"])

    def test_repeated_generated_content_keeps_unique_mc_identity(self):
        from scripts.resource_pilot import sample

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sample.json"
            sample(
                argparse.Namespace(
                    device="cpu",
                    workload="cnn",
                    workers=0,
                    threads=1,
                    batches=3,
                    batch_size=2,
                    image_size=32,
                    dataset_batches=1,
                    ready_file=None,
                    start_file=None,
                    output=output,
                )
            )
            result = json.loads(output.read_text())
            self.assertTrue(result["loss_finite"])
            self.assertLess(result["unique_images"], result["image_paths"])
            self.assertGreater(
                result["workload_finished_unix_seconds"],
                result["workload_started_unix_seconds"],
            )

    @staticmethod
    def sample(start, end):
        return {
            "workload_started_unix_seconds": start,
            "workload_finished_unix_seconds": end,
            "loss_finite": True,
        }
