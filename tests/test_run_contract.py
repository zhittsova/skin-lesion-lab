"""Run records must reject stale, partial and ambiguous experiment outputs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from src import splitting
from tests.test_splitting import cohort


class RunContractTests(unittest.TestCase):
    def setUp(self):
        from src import run_contract

        self.contract = run_contract
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = splitting.create_manifest(cohort(40))
        self.setUp_inputs()

    def setUp_inputs(self):
        self.metadata = self.root / "metadata.csv"
        self.metadata.write_text("synthetic metadata\n")
        self.split = self.root / "split.json"
        splitting.save_manifest(self.manifest, self.split)
        self.lock = self.root / "uv.lock"
        self.lock.write_text("synthetic lock\n")
        self.inputs = {
            "metadata": self.metadata,
            "split_manifest": self.split,
            "dependency_lock": self.lock,
        }

    def start(self, run_id="trial-a", pipeline="classical_gmm"):
        return self.contract.RunRecord.start(
            self.root / "runs",
            run_id=run_id,
            pipeline=pipeline,
            config={"seed": 42, "source": "ham10000", "metadata_path": self.metadata},
            inputs=self.inputs,
        )

    def complete(self, pipeline="classical_gmm"):
        run = self.start(run_id=pipeline, pipeline=pipeline)
        files = {}
        by_id = {row["image_id"]: row for row in self.manifest["rows"]}
        for role in ("train", "development"):
            path = run.path / "results" / "tables" / f"predictions_{role}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            ids = self.manifest["partitions"][role]
            pd.DataFrame(
                {
                    "image_id": ids,
                    "target": [by_id[i]["target"] for i in ids],
                    "prob_melanoma": [0.8 if by_id[i]["target"] else 0.2 for i in ids],
                    "prediction": [by_id[i]["target"] for i in ids],
                }
            ).to_csv(path, index=False)
            files[role] = path
        summary = run.path / "results" / "metrics_summary.json"
        summary.write_text(
            json.dumps({"cost_matrix": {"false_negative": 10, "false_positive": 1}})
        )
        run.finish(self.manifest, files)
        return run

    def test_classical_comparators_recompute_cost_and_map_reports(self):
        for pipeline in ("classical_prevalence", "classical_logistic"):
            with self.subTest(pipeline=pipeline):
                run = self.complete(pipeline)
                report = self.contract.recompute_report(run.path)
                self.assertEqual(
                    set(report["metrics"]["development"]),
                    {"cost_threshold", "map_threshold"},
                )
                self.assertIn("development", report["expected_calibration_error"])

    def test_gmm_failure_reasons_are_diagnostic(self):
        examples = {
            "GMM did not converge": "gmm_nonconvergence",
            "logistic fit did not converge": "logistic_nonconvergence",
            "max_components must be positive": "invalid_gmm_configuration",
            "n_components exceeds class rows": "invalid_gmm_configuration",
            "reg_covar must be positive": "invalid_gmm_configuration",
            "max_iter must be positive": "invalid_gmm_configuration",
            "invalid GMM covariance_type": "invalid_gmm_configuration",
        }
        for message, expected in examples.items():
            with self.subTest(message=message):
                self.assertEqual(
                    self.contract._failure_reason(ValueError(message)), expected
                )

    def test_nonfinite_configuration_is_recorded_safely(self):
        for name, value in (("nan", float("nan")), ("inf", float("inf"))):
            with self.subTest(name=name):
                run = self.contract.RunRecord.start(
                    self.root / "runs",
                    run_id=f"nonfinite-{name}",
                    pipeline="classical_gmm",
                    config={"gmm_reg_covar": value},
                    inputs=self.inputs,
                )
                self.assertEqual(run.record["config"]["gmm_reg_covar"], name)
                self.assertEqual(
                    json.loads((run.path / "run.json").read_text())["status"],
                    "running",
                )

    def test_round_trip_and_prediction_only_metrics(self):
        run = self.complete()
        record, predictions = self.contract.validate_run(run.path)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(set(predictions.role), {"train", "development"})
        self.assertTrue(predictions.group_id.notna().all())
        self.assertTrue(predictions.lesion_id.notna().all())
        self.metadata.unlink()
        metrics = self.contract.recompute_metrics(run.path)
        self.assertEqual(
            metrics["development"]["tp"] + metrics["development"]["tn"],
            len(self.manifest["partitions"]["development"]),
        )
        self.assertNotIn(str(self.root), json.dumps(record))

    def test_duplicate_id_and_failed_run_cannot_validate(self):
        run = self.start()
        with self.assertRaises(FileExistsError):
            self.start()
        with self.assertRaisesRegex(ValueError, "completed"):
            self.contract.validate_run(run.path)
        run.fail(ValueError("private path: " + str(self.root)))
        with self.assertRaisesRegex(ValueError, "completed"):
            self.contract.validate_run(run.path)
        self.assertNotIn(str(self.root), (run.path / "run.json").read_text())

    def test_missing_and_corrupt_artifacts_and_hash_mismatch(self):
        for change in ("delete", "corrupt_csv", "corrupt_json", "hash", "array"):
            with self.subTest(change=change):
                run = self.complete()
                prediction = run.path / "predictions.csv"
                if change == "delete":
                    prediction.unlink()
                elif change == "corrupt_csv":
                    prediction.write_text("broken\n")
                elif change == "corrupt_json":
                    (run.path / "run.json").write_text("{")
                elif change == "hash":
                    prediction.write_bytes(prediction.read_bytes() + b"\n")
                else:
                    array = run.path / "results" / "wrong.npy"
                    array.write_bytes(b"not an array")
                with self.assertRaises(
                    (ValueError, FileNotFoundError, json.JSONDecodeError)
                ):
                    self.contract.validate_run(run.path)
                # Each case gets a fresh run root.
                self.temp = tempfile.TemporaryDirectory()
                self.addCleanup(self.temp.cleanup)
                self.root = Path(self.temp.name)
                self.setUp_inputs()

    def test_content_validation_rejects_rehashed_stale_predictions_and_invalid_array(
        self,
    ):
        run = self.complete()
        file = run.path / "predictions.csv"
        predictions = pd.read_csv(file)
        predictions.loc[0, "group_id"] = "wrong"
        predictions.to_csv(file, index=False)
        record = json.loads((run.path / "run.json").read_text())
        record["artifacts"]["predictions.csv"] = self.contract.sha256(file)
        (run.path / "run.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "grouping"):
            self.contract.validate_run(run.path)

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.setUp_inputs()
        run = self.start()
        invalid = run.path / "bad.npy"
        np.save(invalid, np.array([float("nan")]))
        ids = self.manifest["partitions"]["development"]
        by_id = {row["image_id"]: row for row in self.manifest["rows"]}
        file = run.path / "raw.csv"
        pd.DataFrame(
            {
                "image_id": ids,
                "target": [by_id[i]["target"] for i in ids],
                "prob_melanoma": [0.5] * len(ids),
                "prediction": [1] * len(ids),
            }
        ).to_csv(file, index=False)
        run.finish(self.manifest, {"development": file})
        with self.assertRaisesRegex(ValueError, "array"):
            self.contract.validate_run(run.path)

    def test_input_hash_mismatch_and_manifest_change(self):
        run = self.complete()
        self.metadata.write_text("changed metadata\n")
        with self.assertRaisesRegex(ValueError, "input hash mismatch"):
            self.contract.validate_run(
                run.path, input_paths={"metadata": self.metadata}
            )
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.setUp_inputs()
        run = self.start()
        self.split.write_text("changed manifest")
        with self.assertRaisesRegex(ValueError, "input changed"):
            run.finish(self.manifest, {})

    def test_source_change_during_run_blocks_completion(self):
        run = self.start()
        changed = dict(run.record["source"], commit="changed")
        with patch.object(
            self.contract, "_source_record", return_value=(changed, b"", [])
        ):
            with self.assertRaisesRegex(ValueError, "source changed"):
                run.finish(self.manifest, {})

    def test_rehashed_manifest_with_false_split_hash_is_rejected(self):
        run = self.complete()
        path = run.path / "inputs" / "split-manifest.json"
        manifest = json.loads(path.read_text())
        manifest["rows"][0]["lesion_id"] = "changed"
        path.write_text(json.dumps(manifest))
        record_path = run.path / "run.json"
        record = json.loads(record_path.read_text())
        digest = self.contract.sha256(path)
        record["inputs"]["split_manifest"] = digest
        record["artifacts"]["inputs/split-manifest.json"] = digest
        record_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "content hash"):
            self.contract.validate_run(run.path)

    def test_rehashed_producer_csv_cannot_disagree_with_shared_predictions(self):
        run = self.complete()
        file = run.path / "results" / "tables" / "predictions_development.csv"
        frame = pd.read_csv(file)
        frame.loc[0, "group_id"] = "wrong"
        frame.to_csv(file, index=False)
        record_path = run.path / "run.json"
        record = json.loads(record_path.read_text())
        record["artifacts"]["results/tables/predictions_development.csv"] = (
            self.contract.sha256(file)
        )
        record_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "producer predictions"):
            self.contract.validate_run(run.path)

    def test_failure_midwrite_cannot_be_completed(self):
        run = self.start()
        (run.path / "results").mkdir()
        (run.path / "results" / "half.csv").write_text("half")
        run.fail(RuntimeError("write interrupted"))
        with self.assertRaisesRegex(ValueError, "completed"):
            self.contract.validate_run(run.path)
        resumed = self.contract.RunRecord.start(
            self.root / "runs",
            run_id="retry-b",
            pipeline="classical_gmm",
            config={"seed": 42},
            inputs=self.inputs,
            resume_from="trial-a",
        )
        self.assertEqual(
            json.loads((resumed.path / "run.json").read_text())["resume_from"],
            "trial-a",
        )

    def test_interrupted_running_record_can_be_retried_without_reuse(self):
        run = self.start()
        partial = run.path / "partial.csv"
        partial.write_text("half")
        retry = self.contract.RunRecord.start(
            self.root / "runs",
            run_id="retry-c",
            pipeline="classical_gmm",
            config={"seed": 42},
            inputs=self.inputs,
            resume_from="trial-a",
        )
        self.assertEqual(
            json.loads((retry.path / "run.json").read_text())["resume_from"], "trial-a"
        )
        self.assertFalse((retry.path / "partial.csv").exists())
        self.assertEqual(
            json.loads((run.path / "run.json").read_text())["status"], "running"
        )

    def test_start_failure_records_safe_stage_and_reason(self):
        self.metadata.unlink()
        with self.assertRaises(FileNotFoundError):
            self.start()
        failed = json.loads((self.root / "runs" / "trial-a" / "run.json").read_text())
        self.assertEqual(failed["failure"]["stage"], "hash_inputs")
        self.assertEqual(failed["failure"]["reason"], "missing_file")
        self.assertNotIn(str(self.root), json.dumps(failed))

    def test_finalize_failure_records_safe_stage_and_reason(self):
        run = self.start()
        run.fail(ValueError("input changed during run: metadata"), stage="finalize")
        failed = json.loads((run.path / "run.json").read_text())
        self.assertEqual(failed["failure"]["stage"], "finalize")
        self.assertEqual(failed["failure"]["reason"], "input_changed")


if __name__ == "__main__":
    unittest.main()
