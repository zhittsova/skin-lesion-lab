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

    def complete(
        self, pipeline="classical_gmm", *, run_id=None, probability_by_target=None
    ):
        run = self.start(run_id=run_id or pipeline, pipeline=pipeline)
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
                    "prob_melanoma": [
                        (probability_by_target or {0: 0.2, 1: 0.8})[by_id[i]["target"]]
                        for i in ids
                    ],
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

    def test_saved_v1_summary_replay_is_explicit_and_new_metrics_are_null(self):
        run = self.complete(probability_by_target={0: 0.1, 1: 0.2})
        old = self.contract.recompute_report(run.path)
        self.assertEqual(old["metrics_version"], 1)
        self.assertEqual(old["metrics"]["development"]["map_threshold"]["precision"], 0)
        record_path = run.path / "run.json"
        record = json.loads(record_path.read_text())
        summary_path = run.path / "results/metrics_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["metrics_version"] = 2
        summary_path.write_text(json.dumps(summary))
        record["config"]["metrics_version"] = 2
        record["config_sha256"] = splitting.canonical_hash(record["config"])
        record["artifacts"]["results/metrics_summary.json"] = self.contract.sha256(
            summary_path
        )
        record_path.write_text(json.dumps(record))
        new = self.contract.recompute_report(run.path)
        self.assertIsNone(new["metrics"]["development"]["map_threshold"]["precision"])
        json.dumps(new, allow_nan=False)
        record["config"]["metrics_version"] = 99
        record["config_sha256"] = splitting.canonical_hash(record["config"])
        record_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "metrics version"):
            self.contract.validate_run(run.path)

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

    def test_lexical_identifiers_survive_complete_run_and_reload(self):
        frame = cohort(40)
        replacements = {
            "I0_000": "001",
            "I0_001": "NA",
            "I1_000": "000",
            "I1_001": "NULL",
        }
        frame["isic_id"] = frame.isic_id.replace(replacements)
        frame["lesion_id"] = frame.lesion_id.replace({"L0_000": "002", "L1_000": "NA"})
        self.manifest = splitting.create_manifest(frame)
        self.split.unlink()
        splitting.save_manifest(self.manifest, self.split)
        run = self.complete(
            run_id="000",
            probability_by_target={0: 0.12345678901234568, 1: 0.8765432109876543},
        )
        record, predictions = self.contract.validate_run(run.path)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["run_id"], "000")
        self.assertTrue(set(replacements.values()).issubset(set(predictions.image_id)))
        self.assertTrue({"002", "NA"}.issubset(set(predictions.lesion_id)))
        self.assertIn("001", set(predictions.group_id))
        self.assertIsInstance(predictions.prob_melanoma.iloc[0], np.float64)
        self.assertEqual(
            predictions.loc[predictions.target == 0, "prob_melanoma"].iloc[0],
            0.12345678901234568,
        )
        self.assertIn("development", self.contract.recompute_metrics(run.path))

    def test_reader_keeps_na_like_text_and_rejects_bad_prediction_fields(self):
        lexical = self.root / "lexical.csv"
        pd.DataFrame(
            {"image_id": ["000", "NA", "NULL", "N/A"], "prob_melanoma": [0.1] * 4}
        ).to_csv(lexical, index=False)
        read = self.contract._read_csv(lexical)
        self.assertEqual(read.image_id.tolist(), ["000", "NA", "NULL", "N/A"])
        self.assertTrue(pd.api.types.is_float_dtype(read.prob_melanoma))

        ids = self.manifest["partitions"]["development"]
        targets = {row["image_id"]: row["target"] for row in self.manifest["rows"]}
        for case in ("missing_image_id", "malformed_probability"):
            with self.subTest(case=case):
                run = self.start(run_id=case)
                raw = pd.DataFrame(
                    {
                        "image_id": ids,
                        "target": [targets[image] for image in ids],
                        "prob_melanoma": [0.5] * len(ids),
                        "prediction": [targets[image] for image in ids],
                    }
                )
                if case == "missing_image_id":
                    raw = raw.drop(columns="image_id")
                else:
                    raw["prob_melanoma"] = raw["prob_melanoma"].astype(object)
                    raw.loc[0, "prob_melanoma"] = "bad"
                path = run.path / "raw.csv"
                raw.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    run.finish(self.manifest, {"development": path})
                self.assertEqual(
                    json.loads((run.path / "run.json").read_text())["status"], "running"
                )

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
