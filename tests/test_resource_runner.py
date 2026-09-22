"""Full execution requires a matching, explicit workload manifest."""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.resource_runner import (
    execute,
    validate_jobs,
    validate_profile,
    validate_resource_flags,
)
from src.resource_plan import plan_capacity
from tests.test_resource_plan import GiB, config, device, inventory, observation


class RunnerAdmissionTests(unittest.TestCase):
    def test_duplicate_equal_and_abbreviated_flags_cannot_override_resources(self):
        prefix = ["python", "train.py", "--device", "cuda", "--num-workers", "0"]
        for suffix in (
            ["--device", "cpu"],
            ["--device=cpu"],
            ["--dev", "cpu"],
            ["--num-workers=8"],
            ["--num-wo", "8"],
        ):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                validate_resource_flags(prefix + suffix, "cuda", 0)
        validate_resource_flags(prefix, "cuda", 0)

    def test_larger_frozen_batch_is_rejected_before_input_reads(self):
        from src import benchmark, run_contract

        root = Path(__file__).resolve().parents[1]
        declared = next(j for j in benchmark.planned_runs() if j["pipeline"] == "deep")
        declared["config"] = {**declared["config"], "batch_size": 256}
        report = {
            "pilot_profile": {
                "batch_size": 64,
                "image_size": 128,
                "epochs": 20,
                "candidates": 2,
                "mc_samples": 30,
                "counts": {
                    "train": 5491,
                    "selection": 1366,
                    "calibration": 922,
                    "development": 1395,
                },
                "deep_source_sha256": run_contract.sha256(root / "src/deep.py"),
            },
            "decision": {"selected_device": "cuda:0", "selected_loader_workers": 0},
        }
        plan = {"device": "cuda", "jobs": [declared]}
        with (
            patch("src.benchmark_registry.load_plan", return_value=plan),
            patch("src.benchmark_registry.check_evaluator_source"),
        ):
            with self.assertRaisesRegex(ValueError, "measured profile"):
                validate_profile(
                    report, [{"id": declared["run_id"]}], Path("plan.json")
                )

    def test_fresh_admission_executes_and_records_a_real_child(self):
        now = time.time()
        cfg = config(
            preferred_devices=["cpu"],
            jobs={"cnn": 1},
            loader_workers_candidates=[0],
            as_of_unix_seconds=now,
        )
        inv = inventory(device("cpu", "cpu"), measured_at_unix_seconds=now)
        measurements = {"cnn": {"cpu": {"0": observation(1, GiB, 0, measured_at=now)}}}
        report = {
            "config": cfg,
            "measurements": measurements,
            "decision": plan_capacity(cfg, inv, measurements),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            jobs = [
                {
                    "id": "sample",
                    "workload": "cnn",
                    "cwd": tmp,
                    "argv": [
                        sys.executable,
                        "-c",
                        "print('ran')",
                        "--device",
                        "cpu",
                        "--num-workers",
                        "0",
                    ],
                }
            ]
            with (
                patch("scripts.resource_runner.acquire_for_run", return_value=inv),
                patch("scripts.resource_runner.validate_profile"),
                patch(
                    "scripts.resource_runner.host_available_bytes",
                    return_value=20 * GiB,
                ),
            ):
                execute(report, jobs, out)
            self.assertEqual(
                json.loads((out / "state.json").read_text())["status"], "completed"
            )
            self.assertEqual((out / "sample.log").read_text().strip(), "ran")

    def test_stale_measurement_prevents_any_job_launch(self):
        now = time.time()
        cfg = config(
            preferred_devices=["cpu"],
            jobs={"cnn": 1},
            loader_workers_candidates=[0],
            as_of_unix_seconds=now,
        )
        inv = inventory(device("cpu", "cpu"), measured_at_unix_seconds=now)
        measurements = {
            "cnn": {"cpu": {"0": observation(1, GiB, 0, measured_at=now - 1000)}}
        }
        report = {
            "config": cfg,
            "measurements": measurements,
            "decision": {
                "status": "ready",
                "selected_device": "cpu",
                "selected_loader_workers": 0,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            jobs = [
                {
                    "id": "sample",
                    "workload": "cnn",
                    "cwd": tmp,
                    "argv": [
                        sys.executable,
                        "-c",
                        "raise AssertionError('must not run')",
                    ],
                }
            ]
            with (
                patch("scripts.resource_runner.acquire_for_run", return_value=inv),
                patch("scripts.resource_runner.validate_profile"),
                patch("scripts.resource_runner.subprocess.Popen") as popen,
            ):
                with self.assertRaisesRegex(ValueError, "fresh resource check"):
                    execute(report, jobs, Path(tmp) / "out")
                popen.assert_not_called()

    def test_wrong_workload_count_blocks_before_execution(self):
        jobs = [
            {
                "id": "a",
                "workload": "cnn",
                "argv": ["python", "train.py"],
                "cwd": "/tmp",
            }
        ]
        with self.assertRaisesRegex(ValueError, "counts"):
            validate_jobs({"jobs": {"cnn": 2}}, jobs)

    def test_duplicate_and_path_ids_rejected(self):
        job = {
            "id": "a",
            "workload": "cnn",
            "argv": ["python", "train.py"],
            "cwd": "/tmp",
        }
        with self.assertRaises(ValueError):
            validate_jobs({"jobs": {"cnn": 2}}, [job, job])
        with self.assertRaises(ValueError):
            validate_jobs({"jobs": {"cnn": 1}}, [{**job, "id": "../escape"}])

    def test_accepts_exact_explicit_commands(self):
        job = {
            "id": "cnn-17",
            "workload": "cnn",
            "argv": ["python", "train.py"],
            "cwd": "/tmp",
        }
        self.assertEqual(validate_jobs({"jobs": {"cnn": 1}}, [job]), [job])


if __name__ == "__main__":
    unittest.main()
