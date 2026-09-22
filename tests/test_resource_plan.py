"""Deterministic oracles for the bounded resource-planning contract."""

import math
import unittest

from src import resource_plan

GiB = 1024**3


def config(**updates):
    value = {
        "version": 1,
        "preferred_devices": ["mps:0", "cpu"],
        "wall_budget_seconds": 10_000,
        "pilot_budget_seconds": 300,
        "paid_budget": 0,
        "reserve_host_bytes": 2 * GiB,
        "reserve_disk_bytes": 5 * GiB,
        "reserve_device_bytes": 1 * GiB,
        "max_concurrent_jobs": 8,
        "threads_per_job": 2,
        "loader_workers_candidates": [0, 2],
        "measurement_max_age_seconds": 120,
        "as_of_unix_seconds": 1_000,
        "jobs": {"deep": 6},
    }
    value.update(updates)
    return value


def inventory(*devices, **updates):
    value = {
        "measured_at_unix_seconds": 950,
        "available_cpu_count": 16,
        "available_host_bytes": 20 * GiB,
        "available_disk_bytes": 50 * GiB,
        "devices": list(devices),
    }
    value.update(updates)
    return value


def device(device_id, kind, *, acquired=True, free=10 * GiB, total=12 * GiB):
    return {
        "id": device_id,
        "kind": kind,
        "acquired": acquired,
        "free_bytes": free,
        "total_bytes": total,
    }


def observation(seconds, host, device_bytes, disk=GiB, *, measured_at=950):
    return {
        "measured_at_unix_seconds": measured_at,
        "seconds_per_job": seconds,
        "peak_host_bytes": host,
        "peak_device_bytes": device_bytes,
        "required_disk_bytes": disk,
    }


class ConfigValidationTests(unittest.TestCase):
    def test_normalizes_json_worker_keys_without_mutating_input(self):
        original = config(loader_workers_candidates=[2, 0])
        normalized = resource_plan.validate_config(original)
        self.assertEqual(normalized["loader_workers_candidates"], [2, 0])
        self.assertIsNot(normalized, original)
        self.assertIsNot(normalized["jobs"], original["jobs"])

    def test_rejects_unknown_keys_paid_compute_and_invalid_bounds(self):
        bad_values = [
            config(surprise=True),
            config(paid_budget=0.01),
            config(wall_budget_seconds=0),
            config(pilot_budget_seconds=10_001),
            config(preferred_devices=[]),
            config(preferred_devices=["mps:0", "mps:0"]),
            config(loader_workers_candidates=[0, 0]),
            config(jobs={"deep": 0}),
            config(version=True),
        ]
        for value in bad_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                resource_plan.validate_config(value)


class CapacityPlanTests(unittest.TestCase):
    def test_cpu_capacity_has_measured_loader_choice_and_no_speedup_claim(self):
        cfg = config(preferred_devices=["cpu"], max_concurrent_jobs=10)
        inv = inventory(device("cpu", "cpu"))
        measurements = {
            "deep": {
                "cpu": {
                    "0": observation(12, 3 * GiB, 0),
                    "2": observation(10, 3 * GiB, 0),
                }
            }
        }
        plan = resource_plan.plan_capacity(cfg, inv, measurements)
        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["selected_device"], "cpu")
        self.assertEqual(plan["selected_loader_workers"], 2)
        # min(CPU 16 // (2 + 2), host 18 // 3, configured 10)
        self.assertEqual(plan["attainable_parallelism"], 4)
        self.assertEqual(plan["per_device_slots"], {"cpu": 4})
        self.assertEqual(plan["sequential_makespan_seconds"], 60)
        self.assertEqual(plan["parallel_makespan_seconds"], 60)
        self.assertFalse(plan["parallel_speedup_assumed"])

    def test_mps_counts_host_and_device_peaks_against_shared_ram(self):
        cfg = config(preferred_devices=["mps:0"], loader_workers_candidates=[0])
        inv = inventory(device("mps:0", "mps"), available_host_bytes=8 * GiB)
        measurements = {"deep": {"mps:0": {"0": observation(20, 3 * GiB, 4 * GiB)}}}
        plan = resource_plan.plan_capacity(cfg, inv, measurements)
        self.assertEqual(plan["status"], "blocked")
        self.assertEqual(plan["per_device_slots"], {"mps:0": 0})
        self.assertIn("no_usable_device", plan["reasons"])
        self.assertIn("insufficient_shared_mps_memory", plan["device_reasons"]["mps:0"])

    def test_cuda_is_one_job_per_acquired_device_and_respects_vram(self):
        cfg = config(preferred_devices=["cuda:0"], loader_workers_candidates=[0])
        ok = inventory(device("cuda:0", "cuda", free=5 * GiB))
        measured = {"deep": {"cuda:0": {"0": observation(20, 2 * GiB, 4 * GiB)}}}
        plan = resource_plan.plan_capacity(cfg, ok, measured)
        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["attainable_parallelism"], 1)

        too_large = {"deep": {"cuda:0": {"0": observation(20, 2 * GiB, 4 * GiB + 1)}}}
        plan = resource_plan.plan_capacity(cfg, ok, too_large)
        self.assertEqual(plan["status"], "blocked")
        self.assertIn("insufficient_device_memory", plan["device_reasons"]["cuda:0"])

    def test_unacquired_or_missing_hardware_never_becomes_ready(self):
        cfg = config(preferred_devices=["mps:0"], loader_workers_candidates=[0])
        measured = {"deep": {"mps:0": {"0": observation(20, GiB, GiB)}}}
        for inv in [
            inventory(device("mps:0", "mps", acquired=False)),
            inventory(device("cpu", "cpu")),
        ]:
            with self.subTest(inv=inv):
                plan = resource_plan.plan_capacity(cfg, inv, measured)
                self.assertEqual(plan["status"], "blocked")
                self.assertEqual(plan["attainable_parallelism"], 0)

    def test_only_successful_fresh_measured_worker_choices_are_considered(self):
        cfg = config(preferred_devices=["cpu"])
        inv = inventory(device("cpu", "cpu"))
        # Worker 0 can be omitted after a failed pilot; worker 2 is measured.
        measured = {"deep": {"cpu": {"2": observation(9, 2 * GiB, 0)}}}
        plan = resource_plan.plan_capacity(cfg, inv, measured)
        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["selected_loader_workers"], 2)

        stale = {"deep": {"cpu": {"2": observation(9, 2 * GiB, 0, measured_at=800)}}}
        plan = resource_plan.plan_capacity(cfg, inv, stale)
        self.assertEqual(plan["status"], "blocked")
        self.assertIn("missing_fresh_measurement", plan["device_reasons"]["cpu"])

    def test_disk_and_wall_budget_block_a_full_run(self):
        cfg = config(
            preferred_devices=["cpu"],
            loader_workers_candidates=[0],
            wall_budget_seconds=50,
            pilot_budget_seconds=10,
        )
        inv = inventory(device("cpu", "cpu"), available_disk_bytes=10 * GiB)
        measured = {"deep": {"cpu": {"0": observation(10, GiB, 0)}}}
        plan = resource_plan.plan_capacity(cfg, inv, measured)
        self.assertEqual(plan["status"], "blocked")
        self.assertIn("insufficient_disk", plan["reasons"])
        self.assertIn("wall_budget_exceeded", plan["reasons"])

    def test_zero_nan_infinite_and_unknown_measurement_fields_are_rejected(self):
        cfg = config(preferred_devices=["cpu"], loader_workers_candidates=[0])
        inv = inventory(device("cpu", "cpu"))
        for update in [
            {"seconds_per_job": 0},
            {"seconds_per_job": math.nan},
            {"peak_host_bytes": math.inf},
            {"peak_device_bytes": -1},
            {"unknown": 1},
        ]:
            record = observation(10, GiB, 0)
            record.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                resource_plan.plan_capacity(cfg, inv, {"deep": {"cpu": {"0": record}}})

    def test_falls_back_only_to_an_explicitly_allowed_acquired_device(self):
        cfg = config(loader_workers_candidates=[0])
        inv = inventory(device("mps:0", "mps", acquired=False), device("cpu", "cpu"))
        measured = {
            "deep": {
                "mps:0": {"0": observation(5, GiB, GiB)},
                "cpu": {"0": observation(20, GiB, 0)},
            }
        }
        plan = resource_plan.plan_capacity(cfg, inv, measured)
        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["selected_device"], "cpu")
        self.assertIn("device_not_acquired", plan["device_reasons"]["mps:0"])

    def test_reports_capacity_for_each_allowed_device_but_selects_preference(self):
        cfg = config(loader_workers_candidates=[0])
        inv = inventory(device("mps:0", "mps"), device("cpu", "cpu"))
        measured = {
            "deep": {
                "mps:0": {"0": observation(5, GiB, GiB)},
                "cpu": {"0": observation(20, GiB, 0)},
            }
        }
        plan = resource_plan.plan_capacity(cfg, inv, measured)
        self.assertEqual(plan["selected_device"], "mps:0")
        self.assertEqual(plan["per_device_slots"], {"mps:0": 1, "cpu": 6})


if __name__ == "__main__":
    unittest.main()
