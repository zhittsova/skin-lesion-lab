"""Strict releases must contain complete fitted runs, even with fresh valid hashes."""

import copy
import json
import pickle
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from src import external, external_inference, external_release, run_contract
from tests.external_fixtures import ReleaseFixture, digest, write


class ReleaseRunContractTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.fixture = ReleaseFixture(self.root)

    @contextmanager
    def valid_run(self, name="logistic-17"):
        release = copy.deepcopy(self.fixture.release)
        path = self.root / release["runs"][name]["path"]
        original = {
            p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()
        }
        record, _ = run_contract.validate_run(path)
        external_release.verify_release(
            self.root / "release.json", self.fixture.save(), self.root
        )
        try:
            yield path, record
        finally:
            shutil.rmtree(path)
            for relative, content in original.items():
                (path / relative).parent.mkdir(parents=True, exist_ok=True)
                (path / relative).write_bytes(content)
            self.fixture.release = release
            self.fixture.save()

    def repin(self, path, record, changed=()):
        for relative in changed:
            record["artifacts"][relative] = digest(path / relative)
            self.fixture.pin(str((path / relative).relative_to(self.root)))
        write(path / "run.json", record)
        self.fixture.pin(str((path / "run.json").relative_to(self.root)))
        self.fixture.release["runs"][path.name]["run_sha256"] = digest(
            path / "run.json"
        )

    def omit(self, path, record, relative):
        (path / relative).unlink()
        del record["artifacts"][relative]
        del self.fixture.release["files"][str((path / relative).relative_to(self.root))]
        self.repin(path, record)

    def reject(self, path, ordinary_error):
        with self.assertRaisesRegex((ValueError, KeyError, OSError), ordinary_error):
            run_contract.validate_run(path)
        kwargs = self.fixture.kwargs()
        # Each declared byte agrees with its new ledger; the contract must reject.
        for relative, expected in self.fixture.release["files"].items():
            self.assertEqual(digest(self.root / relative), expected)
        with patch.object(
            external_release, "check_estimator", wraps=external_release.check_estimator
        ) as loader:
            with self.assertRaises(ValueError):
                external_release.verify_release(
                    kwargs["release_path"], kwargs["release_sha256"], self.root
                )
            loader.assert_not_called()
            with patch.object(external_inference, "predict_raw") as predictor:
                with self.assertRaises(ValueError):
                    external_inference.evaluate_run(**kwargs)
                predictor.assert_not_called()
            loader.assert_not_called()
        self.assertFalse(kwargs["output"].exists())

    def test_omitted_calibration_producer_with_retained_reference(self):
        with self.valid_run() as (path, record):
            self.omit(path, record, record["prediction_files"]["calibration"])
            self.reject(path, "missing producer prediction artifact")

    def test_existing_unregistered_producer_reference_and_extra_file(self):
        for reference in (True, False):
            with self.subTest(reference=reference), self.valid_run() as (path, record):
                relative = "results/unregistered.csv"
                shutil.copyfile(
                    path / record["prediction_files"]["calibration"], path / relative
                )
                if reference:
                    record["prediction_files"]["calibration"] = relative
                # Even a release-ledger entry cannot substitute for the run ledger.
                self.fixture.pin(str((path / relative).relative_to(self.root)))
                self.repin(path, record)
                self.reject(path, "unregistered artifact")

    def test_required_producer_key_cannot_be_omitted(self):
        with self.valid_run() as (path, record):
            del record["prediction_files"]["calibration"]
            self.repin(path, record)
            self.reject(path, "calibration")

    def test_shared_and_producer_predictions_must_agree(self):
        self.fixture.add_run("cnn-scratch-unweighted", 17)
        for name in ("logistic-17", "cnn-scratch-unweighted-17"):
            with self.subTest(name=name), self.valid_run(name) as (path, record):
                relative = record["prediction_files"]["calibration"]
                producer = run_contract._read_csv(path / relative)
                producer.loc[0, "prob_melanoma"] = 0.45
                producer.to_csv(path / relative, index=False)
                self.repin(path, record, [relative])
                self.reject(path, "producer predictions disagree")

    def test_required_source_and_shared_artifacts_cannot_be_removed(self):
        self.fixture.add_run("cnn-scratch-unweighted", 17)
        for name in ("logistic-17", "cnn-scratch-unweighted-17"):
            for relative in (
                "inputs/source.diff",
                "inputs/untracked/src/generated_fixture.py",
                "predictions.csv",
            ):
                with (
                    self.subTest(name=name, relative=relative),
                    self.valid_run(name) as (path, record),
                ):
                    self.omit(path, record, relative)
                    self.reject(path, "No such file")

    def test_deep_array_and_schema_omissions(self):
        self.fixture.add_run("cnn-scratch-unweighted", 17)
        for defect in ("file", "schema", "shape", "content"):
            with (
                self.subTest(defect=defect),
                self.valid_run("cnn-scratch-unweighted-17") as (path, record),
            ):
                relative = "arrays/small_cnn_development_mc_probabilities.npy"
                if defect == "file":
                    del record["array_schemas"][relative]
                    self.omit(path, record, relative)
                    error = "No such file"
                elif defect == "schema":
                    del record["array_schemas"][relative]
                    self.repin(path, record)
                    error = "array artifact set mismatch"
                else:
                    values = np.load(path / relative, allow_pickle=False)
                    values = values[:, :1] if defect == "shape" else values + 0.01
                    np.save(path / relative, values)
                    record["array_schemas"][relative]["shape"] = list(values.shape)
                    self.repin(path, record, [relative])
                    error = (
                        "deep array sample count"
                        if defect == "shape"
                        else "deep array content"
                    )
                self.reject(path, error)

    def test_deep_metadata_and_split_omissions(self):
        self.fixture.add_run("cnn-scratch-unweighted", 17)
        for relative in (
            "results/deep_training_metadata.json",
            "inputs/split-manifest.json",
        ):
            with (
                self.subTest(relative=relative),
                self.valid_run("cnn-scratch-unweighted-17") as (path, record),
            ):
                self.omit(path, record, relative)
                self.reject(path, "No such file")

    def test_policy_provenance_and_summary_must_agree_with_saved_scores(self):
        for defect in ("score", "fit_digest", "summary"):
            with self.subTest(defect=defect), self.valid_run() as (path, record):
                if defect == "score":
                    relative = record["prediction_files"]["calibration"]
                    producer = run_contract._read_csv(path / relative)
                    producer.loc[0, "raw_score"] = 0.4
                    producer.to_csv(path / relative, index=False)
                    error = "decision policy"
                else:
                    relative = (
                        "models/decision_policy.json"
                        if defect == "fit_digest"
                        else "results/metrics_summary.json"
                    )
                    value = json.loads((path / relative).read_text())
                    if defect == "fit_digest":
                        value["fit"]["data_sha256"] = "0" * 64
                        estimator = "models/logistic_model.pkl"
                        with (path / estimator).open("rb") as handle:
                            saved = pickle.load(handle)
                        saved["decision_policy"] = value
                        with (path / estimator).open("wb") as handle:
                            pickle.dump(saved, handle)
                        self.repin(path, record, [estimator])
                    else:
                        value["cost_matrix"]["cost_threshold"] = 0.123
                    write(path / relative, value)
                    error = (
                        "calibration fit data digest"
                        if defect == "fit_digest"
                        else "summary operating points"
                    )
                self.repin(path, record, [relative])
                self.reject(path, error)

    def test_completion_rechecks_unregistered_fitted_dependency(self):
        kwargs = self.fixture.kwargs()
        original = external_inference.predict_raw

        def predict_then_mutate(*args, **options):
            scores = original(*args, **options)
            (self.root / "fitted/logistic-17/extra.txt").write_text(
                "added during prediction"
            )
            return scores

        with patch.object(
            external_inference, "predict_raw", side_effect=predict_then_mutate
        ) as predictor:
            with self.assertRaisesRegex(ValueError, "unregistered artifact"):
                external_inference.evaluate_run(**kwargs)
            predictor.assert_called_once()
        status = json.loads((kwargs["output"] / "run.json").read_text())
        self.assertEqual(status["status"], "failed")

    def test_complete_controls_include_real_tiny_training_pipelines(self):
        from scripts.smoke_experiment import make_cohort, run_cli

        generated = self.root / "training"
        generated.mkdir()
        metadata, images = make_cohort(generated)
        manifest = generated / "manifest.json"
        common = [
            "--source",
            "ham10000",
            "--metadata-path",
            str(metadata),
            "--images-dir",
            str(images),
            "--split-manifest",
            str(manifest),
        ]
        run_cli(generated, "freeze", ["freeze_splits.py", *common])
        for family, seed, entrypoint, options in (
            ("logistic", 42, "train_pipeline.py", ["--model", "logistic"]),
            (
                "cnn-scratch-unweighted",
                73,
                "train_deep_pipeline.py",
                [
                    "--epochs",
                    "1",
                    "--mc-samples",
                    "2",
                    "--image-size",
                    "32",
                    "--batch-size",
                    "8",
                    "--device",
                    "cpu",
                ],
            ),
        ):
            name = f"{family}-{seed}"
            run_cli(
                generated,
                name,
                [
                    entrypoint,
                    *common,
                    "--runs-dir",
                    str(self.root / "fitted"),
                    "--run-id",
                    name,
                    "--seed",
                    str(seed),
                    *options,
                ],
            )
            self.fixture.register_run(family, seed)
            with self.valid_run(name) as (_, record):
                self.assertEqual(record["status"], "completed")
                self.assertEqual(
                    set(record["prediction_roles"]),
                    {"train", "selection", "calibration", "development"}
                    if family == "logistic"
                    else {"calibration", "development"},
                )
                split = json.loads(
                    (
                        self.root / "fitted" / name / "inputs/split-manifest.json"
                    ).read_text()
                )
                self.assertEqual(
                    set(split["partitions"]),
                    {"train", "selection", "calibration", "development"},
                )

    def test_report_rechecks_contract_at_entry_and_before_write(self):
        for family in external_release.FAMILIES:
            for seed in external_release.SEEDS:
                if (family, seed) != ("logistic", 17):
                    self.fixture.add_run(family, seed)
        for name in self.fixture.release["runs"]:
            external_inference.evaluate_run(
                **self.fixture.kwargs(name, output="inference/" + name)
            )
        common = self.fixture.kwargs()
        for key in ("run_id", "images_dir", "output"):
            del common[key]
        common.update(
            runs_dir=self.root / "inference", output_path=self.root / "accepted.json"
        )
        external_inference.write_report(
            **{**common, "output_path": self.root / "control.json"}
        )
        # A reassembled release and rebound execution envelopes isolate the contract
        # defect from otherwise stale release hashes in saved external runs.
        with self.valid_run() as (path, record):
            self.omit(path, record, record["prediction_files"]["calibration"])
            common["release_sha256"] = self.fixture.save()
            for name in self.fixture.release["runs"]:
                status_path = self.root / "inference" / name / "run.json"
                status = json.loads(status_path.read_text())
                status["release_sha256"] = common["release_sha256"]
                status["fitted_identity"] = self.fixture.release["runs"][name]
                write(status_path, status)
            with self.assertRaisesRegex(
                ValueError, "missing producer prediction artifact"
            ):
                external_inference.write_report(**common)
            self.assertFalse(common["output_path"].exists())
        common["release_sha256"] = self.fixture.save()
        for name in self.fixture.release["runs"]:
            status_path = self.root / "inference" / name / "run.json"
            status = json.loads(status_path.read_text())
            status["release_sha256"] = common["release_sha256"]
            status["fitted_identity"] = self.fixture.release["runs"][name]
            write(status_path, status)
        original = external.paired_report

        def report_then_mutate(*args, **options):
            report = original(*args, **options)
            (self.root / "fitted/logistic-17/extra.txt").write_text(
                "added during reporting"
            )
            return report

        with patch.object(
            external, "paired_report", side_effect=report_then_mutate
        ) as reporter:
            with self.assertRaisesRegex(ValueError, "unregistered artifact"):
                external_inference.write_report(**common)
            reporter.assert_called_once()
        self.assertFalse(common["output_path"].exists())
