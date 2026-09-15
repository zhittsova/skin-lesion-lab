"""Frozen schedule and saved-run provenance gates."""

import tempfile
import unittest
from pathlib import Path

from src import benchmark_registry


class RegistryTests(unittest.TestCase):
    def test_frozen_registry_cannot_be_overwritten_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "plan.json"
            payload = {"jobs": [], "purpose": "development"}
            benchmark_registry.save_plan(p, payload)
            self.assertEqual(benchmark_registry.load_plan(p), payload)
            with self.assertRaises(FileExistsError):
                benchmark_registry.save_plan(p, payload)
            p.write_text(p.read_text().replace("development", "confirmation"))
            with self.assertRaisesRegex(ValueError, "hash"):
                benchmark_registry.load_plan(p)

    def test_run_gate_rejects_wrong_config_source_inputs_and_old_run(self):
        job = {
            "run_id": "prevalence-17",
            "pipeline": "classical",
            "config": {"model": "prevalence", "seed": 17},
        }
        plan = {
            "created_utc": "2026-09-15T00:00:00+00:00",
            "source": {"commit": "frozen"},
            "inputs": {"metadata": "abc"},
            "dataset_source": "ham10000",
            "device": "cpu",
        }
        run = {
            "created_utc": "2026-09-15T01:00:00+00:00",
            "source": plan["source"],
            "inputs": plan["inputs"],
            "run_id": job["run_id"],
            "pipeline": "classical_prevalence",
            "config": {**job["config"], "source": "ham10000"},
        }
        benchmark_registry.check_scheduled_run(plan, job, run)
        for key, bad in [
            ("source", {"commit": "changed"}),
            ("inputs", {"metadata": "bad"}),
            ("created_utc", "2026-09-14T01:00:00+00:00"),
            ("config", {**run["config"], "seed": 73}),
        ]:
            with self.assertRaises(ValueError):
                benchmark_registry.check_scheduled_run(plan, job, {**run, key: bad})


class RegistryReviewTests(unittest.TestCase):
    def test_changed_evaluator_cannot_report_a_frozen_plan(self):
        from unittest.mock import patch

        with patch(
            "src.benchmark_registry.run_contract._source_record",
            return_value=({"commit": "changed"}, b"", []),
        ):
            with self.assertRaises(ValueError):
                benchmark_registry.check_evaluator_source(
                    {"source": {"commit": "frozen"}}
                )

    def test_missing_development_rows_cannot_count_as_completed(self):
        import json
        from unittest.mock import patch

        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "trial"
            (run / "inputs").mkdir(parents=True)
            record = {"status": "completed", "artifacts": {}}
            (run / "run.json").write_text(json.dumps(record))
            (run / "inputs/split-manifest.json").write_text(
                json.dumps({"partitions": {"development": ["dev"]}})
            )
            plan = {
                "jobs": [
                    {
                        "run_id": "trial",
                        "stratum": "model",
                        "seed": 17,
                        "candidate_count": 1,
                    }
                ]
            }
            frame = pd.DataFrame({"role": ["selection"], "image_id": ["select"]})
            with (
                patch(
                    "src.benchmark_registry.run_contract.validate_run",
                    return_value=(record, frame),
                ),
                patch("src.benchmark_registry.check_scheduled_run"),
            ):
                status, _ = benchmark_registry.inventory(plan, root)
            self.assertFalse(status["complete"])
            self.assertEqual(status["jobs"][0]["status"], "invalid")
