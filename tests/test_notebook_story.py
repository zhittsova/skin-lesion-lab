"""Notebook input and provenance contracts."""

import tempfile
import unittest
from pathlib import Path

from src.notebook_story import accepted_report, summary_rows, synthetic_demo


class NotebookStoryTests(unittest.TestCase):
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

    def test_synthetic_demo_is_explicit_and_uses_pipeline_endpoints(self):
        demo = synthetic_demo()
        self.assertEqual(demo["kind"], "illustrative synthetic data")
        self.assertEqual(demo["run_id"], "synthetic-example-not-a-run")
        self.assertEqual(demo["counts"], {"images": 6, "groups": 4})
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
        with self.assertRaisesRegex(ValueError, "run IDs"):
            summary_rows(report, "roc_auc")

    def test_accepted_report_links_all_model_seeds(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "runs/benchmark-s09-recovery-20260921/benchmark-report.json"
        if not path.exists():
            self.skipTest("private S09 report is not distributed")
        report = accepted_report(path)
        rows = summary_rows(report, "roc_auc")
        self.assertEqual(len(rows), 9)
        self.assertEqual(sum(len(row["run_ids"]) for row in rows), 27)
        self.assertAlmostEqual(
            next(
                row["estimate"]
                for row in rows
                if row["model"] == "efficientnet_full_unweighted"
            ),
            0.8906,
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
