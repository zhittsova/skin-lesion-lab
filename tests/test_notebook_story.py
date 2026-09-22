"""Notebook input and provenance contracts."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.notebook_story import (
    accepted_report,
    summary_rows,
    synthetic_demo,
    synthetic_report,
)


class NotebookStoryTests(unittest.TestCase):
    def accepted_fixture(self, directory, change=None):
        """Use generated predictions and a test-only digest as the reader oracle."""
        report = synthetic_report()
        report.update(
            purpose="development",
            plan_sha256="fixture-plan",
            counts={"images": 1395, "groups": 1037, "classes": {"0": 1219, "1": 176}},
        )
        report["registry"]["complete"] = True
        if change is not None:
            change(report)
        path = Path(directory) / "fixture-report.json"
        content = json.dumps(report, sort_keys=True).encode()
        path.write_bytes(content)
        return path, hashlib.sha256(content).hexdigest()

    def test_published_notebooks_are_clean_and_compile(self):
        import json

        root = Path(__file__).resolve().parents[1]
        for path in sorted((root / "notebooks").glob("*.ipynb")):
            notebook = json.loads(path.read_text())
            for cell in notebook["cells"]:
                self.assertFalse(cell.get("attachments"), path.name)
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"], path.name)
                    self.assertFalse(cell["outputs"], path.name)
                    compile("".join(cell["source"]), str(path), "exec")

    def test_accepted_report_rejects_tampered_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark-report.json"
            path.write_text('{"purpose":"development"}')
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                accepted_report(path)

    def test_generated_accepted_fixture_checks_digest_plan_and_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            path, digest = self.accepted_fixture(directory)
            with (
                patch("src.notebook_story.ACCEPTED_SHA256", digest),
                patch("src.notebook_story.ACCEPTED_PLAN_SHA256", "fixture-plan"),
            ):
                report = accepted_report(path)
                self.assertEqual(len(summary_rows(report, "roc_auc")), 2)
                self.assertEqual(len(report["registry"]["jobs"]), 6)
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    accepted_report(path)

    def test_generated_fixture_rejects_wrong_plan_and_registry_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            for change in (
                lambda report: report.update(plan_sha256="wrong-plan"),
                lambda report: report["registry"]["jobs"][0].update(seed=999),
                lambda report: report["registry"]["jobs"][0].update(run_id=""),
                lambda report: report["registry"]["jobs"].append(
                    dict(report["registry"]["jobs"][0], stratum="orphan")
                ),
            ):
                path, digest = self.accepted_fixture(directory, change)
                with (
                    patch("src.notebook_story.ACCEPTED_SHA256", digest),
                    patch("src.notebook_story.ACCEPTED_PLAN_SHA256", "fixture-plan"),
                ):
                    with self.assertRaises(ValueError):
                        accepted_report(path)

    def test_synthetic_demo_is_explicit_and_uses_pipeline_endpoints(self):
        demo = synthetic_demo()
        self.assertEqual(demo["kind"], "illustrative synthetic data")
        self.assertEqual(demo["run_id"], "synthetic-example-not-a-run")
        self.assertEqual(demo["counts"], {"records": 6, "groups": 4})
        self.assertEqual(demo["ungrouped_roc_auc"], 1.0)
        self.assertEqual(demo["grouped_roc_auc"], 0.5)

    def test_summary_rows_reject_missing_run_provenance(self):
        report = {
            "models": {
                "logistic": {
                    "roc_auc": {
                        "estimate": 0.7,
                        "interval": [0.6, 0.8],
                        "seed_sd": 0.01,
                    }
                }
            },
            "registry": {"jobs": []},
            "seeds": [17, 42, 73],
        }
        with self.assertRaisesRegex(ValueError, "registry jobs"):
            summary_rows(report, "roc_auc")


if __name__ == "__main__":
    unittest.main()
