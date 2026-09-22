"""Release closure, identity, lexical output and actual tiny inference controls."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from src import (
    calibration,
    classical,
    deep,
    external,
    external_inference,
    external_release,
    splitting,
)
from tests.external_fixtures import ReleaseFixture, digest, write


class ExternalReleaseTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.fixture = ReleaseFixture(self.root)

    def reject_before_prediction(self):
        with patch.object(external_inference, "predict_raw") as predictor:
            with self.assertRaises(ValueError):
                external_inference.evaluate_run(**self.fixture.kwargs())
            predictor.assert_not_called()
        self.assertFalse((self.root / "output").exists())

    def test_complete_release_has_real_estimator_policy_and_lexical_replay(self):
        with (
            patch.object(classical, "fit_logistic", side_effect=AssertionError("fit")),
            patch.object(
                calibration, "fit_policy", side_effect=AssertionError("calibration fit")
            ),
            patch.object(
                deep, "train_one_epoch", side_effect=AssertionError("training")
            ),
        ):
            kwargs = self.fixture.kwargs()
            status = external_inference.evaluate_run(**kwargs)
        self.assertEqual(status["status"], "completed")
        observed = self.replay(kwargs)
        self.assertEqual(
            observed.image_id.tolist(),
            sorted(["001", "000", "NA", "NULL", "7", "mixed_A"]),
        )
        self.assertEqual(observed.group_id.tolist(), observed.image_id.tolist())
        self.assertEqual(observed.prob_melanoma.dtype.kind, "f")
        with self.assertRaises(FileExistsError):
            external_inference.evaluate_run(**kwargs)
        raw = np.load(self.root / "output/raw_scores.npy")
        policy = json.loads(
            (self.root / "fitted/logistic-17/models/decision_policy.json").read_text()
        )
        # Independent v1 equation, without the production policy application helper.
        clipped = np.clip(raw, 1e-10, 1 - 1e-10)
        z = (
            policy["calibrator"]["slope"] * (np.log(clipped) - np.log1p(-clipped))
            + policy["calibrator"]["intercept"]
        )
        expected = np.empty_like(z)
        expected[z >= 0] = 1 / (1 + np.exp(-z[z >= 0]))
        ez = np.exp(z[z < 0])
        expected[z < 0] = ez / (1 + ez)
        np.testing.assert_array_equal(observed.prob_melanoma, expected)
        np.testing.assert_array_equal(
            observed.prediction, expected >= policy["decision"]["threshold"]
        )
        self.assertEqual(digest(self.root / "release.json"), kwargs["release_sha256"])

    def replay(self, kwargs, *, legacy=False):
        return external_inference.validate_predictions(
            kwargs["output"],
            self.fixture.release,
            kwargs["release_sha256"],
            self.fixture.manifest,
            kwargs["manifest_sha256"],
            self.root,
            expected_run_id=kwargs["run_id"],
            audit_sha256=kwargs["audit_sha256"],
            legacy=legacy,
        )

    def test_omitted_required_dependencies_and_unrelated_marker_fail_before_inference(
        self,
    ):
        original = copy.deepcopy(self.fixture.release)
        required = [
            "fitted/logistic-17/run.json",
            "fitted/logistic-17/models/logistic_model.pkl",
            "fitted/logistic-17/models/decision_policy.json",
            "fitted/logistic-17/inputs/split-manifest.json",
            "environment.json",
            "pyproject.toml",
            "uv.lock",
            ".python-version",
            "protocol.md",
            "docs/evaluation-protocol.md",
            "scripts/evaluate_external.py",
            "src/external.py",
            "src/external_inference.py",
            "src/run_contract.py",
            "src/features.py",
            "src/evaluation.py",
            "src/bayes.py",
        ]
        for name in required:
            with self.subTest(omitted=name):
                self.fixture.release = copy.deepcopy(original)
                del self.fixture.release["files"][name]
                self.reject_before_prediction()
        self.fixture.release = copy.deepcopy(original)
        (self.root / "marker").write_bytes(b"not the referenced estimator")
        self.fixture.release["files"] = {"marker": digest(self.root / "marker")}
        self.reject_before_prediction()

    def test_mutation_missing_file_and_bad_digest_fail_before_inference(self):
        for name in list(self.fixture.release["files"]):
            with self.subTest(mutated=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                self.reject_before_prediction()
                path.write_bytes(original)
        path = self.root / "fitted/logistic-17/models/logistic_model.pkl"
        original = path.read_bytes()
        path.unlink()
        self.reject_before_prediction()
        path.write_bytes(original)
        self.fixture.release["files"][str(path.relative_to(self.root))] = "bad"
        self.reject_before_prediction()

    def test_unsupported_or_legacy_schema_cannot_authorize_inference(self):
        for version in (None, 1, 3, True, "2"):
            with self.subTest(version=version):
                self.fixture.release["schema_version"] = version
                self.reject_before_prediction()

    def test_every_declared_identity_and_reference_is_checked(self):
        entry = self.fixture.release["runs"]["logistic-17"]
        for key, wrong in {
            "run_id": "wrong",
            "family": "efficientnet-full-unweighted",
            "seed": 73,
            "pipeline": "deep_efficientnet_b0",
            "architecture": "efficientnet_b0",
            "pretrained": True,
            "mode": "full",
            "loss_strategy": "pos_weight",
            "positive_weight": 2,
            "policy_version": 2,
            "config_sha256": "0" * 64,
            "run_sha256": "0" * 64,
            "estimator": "marker",
            "policy": "marker",
            "preprocessing": {"kind": "none"},
            "device": "mps",
        }.items():
            with self.subTest(key=key):
                before = entry[key]
                entry[key] = wrong
                self.reject_before_prediction()
                entry[key] = before

    def test_self_hashed_relabeling_cannot_override_fitted_facts(self):
        path = self.root / "fitted/logistic-17/run.json"
        original = json.loads(path.read_text())
        for key, value in (("seed", 73), ("model", "gmm"), ("run_id", "renamed")):
            with self.subTest(key=key):
                record = copy.deepcopy(original)
                record["config"][key] = value
                record["config_sha256"] = splitting.canonical_hash(record["config"])
                write(path, record)
                self.fixture.pin(str(path.relative_to(self.root)))
                self.fixture.release["runs"]["logistic-17"].update(
                    run_sha256=digest(path), config_sha256=record["config_sha256"]
                )
                self.reject_before_prediction()

    def test_runtime_source_environment_and_protocol_cannot_be_rebound(self):
        original = copy.deepcopy(self.fixture.release)
        for name in (
            "src/features.py",
            "scripts/evaluate_external.py",
            "docs/evaluation-protocol.md",
            "uv.lock",
        ):
            path = self.root / name
            old = path.read_bytes()
            path.write_bytes(old + b"\n")
            self.fixture.pin(name)
            with self.subTest(name=name):
                self.reject_before_prediction()
            path.write_bytes(old)
            self.fixture.release = copy.deepcopy(original)
        env_path = self.root / "environment.json"
        env = json.loads(env_path.read_text())
        env["python"] = "0.0.0"
        write(env_path, env)
        self.fixture.pin("environment.json")
        self.reject_before_prediction()

    def test_repeated_reference_duplicate_json_key_and_unsafe_path_rejected(self):
        original = copy.deepcopy(self.fixture.release)
        self.fixture.release["source_files"].append("src/external.py")
        self.reject_before_prediction()
        self.fixture.release = copy.deepcopy(original)
        self.fixture.release["runs"]["second"] = copy.deepcopy(
            self.fixture.release["runs"]["logistic-17"]
        )
        self.reject_before_prediction()
        self.fixture.release = copy.deepcopy(original)
        for name in ("../outside", "/absolute", "src/../src/external.py"):
            self.fixture.release["files"][name] = "0" * 64
            self.reject_before_prediction()
            del self.fixture.release["files"][name]
        path = self.root / "duplicate.json"
        path.write_text('{"schema_version":2,"schema_version":1}')
        with self.assertRaisesRegex(ValueError, "repeated"):
            external.verify_files(path, digest(path), self.root)

    def test_completion_rechecks_all_inputs_and_never_marks_mutated_run_completed(self):
        for name in (
            "protocol.md",
            "environment.json",
            "src/features.py",
            "fitted/logistic-17/models/decision_policy.json",
            "manifest.json",
            "audit.json",
            "001.jpg",
        ):
            path = self.root / name
            original = path.read_bytes()

            def mutate(*args):
                path.write_bytes(original + b" ")
                return np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

            with (
                self.subTest(name=name),
                patch.object(external_inference, "predict_raw", side_effect=mutate),
            ):
                kwargs = self.fixture.kwargs(
                    output="failed-" + str(len(list(self.root.glob("failed-*"))))
                )
                with self.assertRaises(ValueError):
                    external_inference.evaluate_run(**kwargs)
                self.assertEqual(
                    json.loads((kwargs["output"] / "run.json").read_text())["status"],
                    "failed",
                )
            path.write_bytes(original)

    def test_fitted_partition_overlap_is_rejected_before_prediction(self):
        path = self.root / "fitted/logistic-17/inputs/split-manifest.json"
        value = json.loads(path.read_text())
        value["partitions"]["train"][0] = "001"
        write(path, value)
        rec_path = path.parents[1] / "run.json"
        rec = json.loads(rec_path.read_text())
        rec["artifacts"]["inputs/split-manifest.json"] = digest(path)
        write(rec_path, rec)
        self.fixture.pin(str(path.relative_to(self.root)))
        self.fixture.pin(str(rec_path.relative_to(self.root)))
        self.fixture.release["runs"]["logistic-17"]["run_sha256"] = digest(rec_path)
        self.reject_before_prediction()

    def test_matrix_requires_each_actual_family_seed_once(self):
        runs = {
            f"{family}-{seed}": {"family": family, "seed": seed}
            for family in ("logistic", "efficientnet-full-unweighted")
            for seed in (17, 42, 73)
        }
        external_release.validate_matrix(runs)
        variants = [
            dict(list(runs.items())[:-1]),
            {**runs, "duplicate": runs["logistic-17"]},
            {**runs, "extra": {"family": "gmm", "seed": 17}},
        ]
        for value in variants:
            with self.assertRaises(ValueError):
                external_release.validate_matrix(value)

    def test_deep_identity_modes_and_loss_have_valid_controls(self):
        for family in (
            "cnn-scratch-unweighted",
            "cnn-scratch-pos_weight",
            "efficientnet-full-unweighted",
            "efficientnet-head-unweighted",
        ):
            self.fixture.add_run(family, 17)
        external.verify_files(
            self.root / "release.json", self.fixture.save(), self.root
        )
        name = "cnn-scratch-unweighted-17"
        with (
            patch.object(calibration, "fit_policy", side_effect=AssertionError("fit")),
            patch.object(deep, "train_one_epoch", side_effect=AssertionError("train")),
        ):
            kwargs = self.fixture.kwargs(name, output="deep-output")
            external_inference.evaluate_run(**kwargs)
            self.replay(kwargs)
        entry = self.fixture.release["runs"]["efficientnet-full-unweighted-17"]
        for key, value in (
            ("mode", "head"),
            ("architecture", "small_cnn"),
            ("loss_strategy", "pos_weight"),
            ("positive_weight", 2.0),
            ("seed", 73),
        ):
            with self.subTest(key=key):
                old = entry[key]
                entry[key] = value
                self.reject_before_prediction()
                entry[key] = old
        training_name = entry["preprocessing"]["artifact"]
        del self.fixture.release["files"][training_name]
        self.reject_before_prediction()

    def test_future_gmm_policy_retains_finite_log_and_ranking_scores(self):
        self.fixture.add_run("gmm", 17, policy_version=2)
        kwargs = self.fixture.kwargs("gmm-17", output="gmm-output")
        with patch.object(calibration, "fit_policy", side_effect=AssertionError("fit")):
            external_inference.evaluate_run(**kwargs)
            frame = self.replay(kwargs)
        self.assertTrue(np.isfinite(frame.calibration_score).all())
        np.testing.assert_array_equal(frame.ranking_score, frame.calibration_score)

    def test_full_matrix_tiny_cli_flow_and_reporting_reject_mixtures(self):
        import os
        import subprocess
        import sys

        for family in ("logistic", "efficientnet-full-unweighted"):
            for seed in (17, 42, 73):
                if (family, seed) != ("logistic", 17):
                    self.fixture.add_run(family, seed)
        common = dict(
            root=self.root,
            release_path=self.root / "release.json",
            release_sha256=self.fixture.save(),
            manifest_path=self.root / "manifest.json",
            manifest_sha256=digest(self.root / "manifest.json"),
            audit_path=self.root / "audit.json",
            audit_sha256=digest(self.root / "audit.json"),
            runs_dir=self.root / "runs",
        )
        for name in self.fixture.release["runs"]:
            if name == "logistic-17":
                # Real CLI from a portable source snapshot, without .git or private files.
                command = [
                    sys.executable,
                    "-c",
                    "import runpy,sys,torch; torch.set_num_threads(1); p=sys.argv.pop(1); runpy.run_path(p,run_name='__main__')",
                    str(self.root / "scripts/evaluate_external.py"),
                    "run",
                    "--release",
                    str(common["release_path"]),
                    "--release-sha256",
                    common["release_sha256"],
                    "--manifest",
                    str(common["manifest_path"]),
                    "--manifest-sha256",
                    common["manifest_sha256"],
                    "--audit",
                    str(common["audit_path"]),
                    "--audit-sha256",
                    common["audit_sha256"],
                    "--images-dir",
                    str(self.root),
                    "--run-id",
                    name,
                    "--runs-dir",
                    str(common["runs_dir"]),
                ]
                result = subprocess.run(
                    command,
                    cwd=self.root,
                    env={**os.environ},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            else:
                external_inference.evaluate_run(
                    **self.fixture.kwargs(name, output=f"runs/{name}")
                )
        # Actual fitted facts, complete real estimator/checkpoint loading, and saved output replay.
        report = external_inference.write_report(**common, draws=5)
        self.assertEqual(len(report["registry"]), 6)
        self.assertTrue(report["strict_release_accepted"])
        self.assertEqual(
            set(report["models"]), {"logistic", "efficientnet-full-unweighted"}
        )
        with self.assertRaisesRegex(ValueError, "report settings"):
            external_inference.write_report(
                **common, output_path=self.root / "wrong-settings.json", draws=2
            )
        # The same generated scores can be inspected under an explicit historical
        # envelope, but that envelope cannot pass the strict execution gate.
        strict_release = copy.deepcopy(self.fixture.release)
        historical = copy.deepcopy(strict_release)
        historical["schema_version"] = 1
        historical["protocol_sha256"] = digest(self.root / "protocol.md")
        saved_bytes = {}
        for name, entry in historical["runs"].items():
            run_path = self.root / "runs" / name
            for filename in ("run.json", "policy-report.json"):
                saved_bytes[run_path / filename] = (run_path / filename).read_bytes()
            status = json.loads((run_path / "run.json").read_text())
            policy = json.loads((self.root / entry["policy"]).read_text())
            raw = np.load(run_path / "raw_scores.npy")
            cohort = sorted(
                self.fixture.manifest["rows"], key=lambda row: row["image_id"]
            )
            value = calibration.policy_report(
                policy,
                np.array([r["target"] for r in cohort]),
                raw,
                variances=raw.var(axis=0, ddof=1) if raw.ndim == 2 else None,
                metrics_version=1,
            )
            write(run_path / "policy-report.json", value)
            status["schema_version"] = 1
            status["artifacts"]["policy-report.json"] = digest(
                run_path / "policy-report.json"
            )
            write(run_path / "run.json", status)
        self.fixture.release = historical
        legacy_digest = self.fixture.save()
        for name in historical["runs"]:
            path = self.root / "runs" / name / "run.json"
            status = json.loads(path.read_text())
            status["release_sha256"] = legacy_digest
            write(path, status)
        legacy = external_inference.write_report(
            **{**common, "release_sha256": legacy_digest},
            output_path=self.root / "legacy-inspection.json",
            legacy_source_root=self.root,
            legacy_protocol_path=self.root / "protocol.md",
            draws=2,
        )
        self.assertFalse(legacy["strict_release_accepted"])
        self.assertEqual(legacy["contract"], "legacy-v1-inspection")
        self.reject_before_prediction()
        for path, value in saved_bytes.items():
            path.write_bytes(value)
        self.fixture.release = strict_release
        self.fixture.save()
        for key, bad in (
            ("release_sha256", "0" * 64),
            ("fitted_identity", {}),
            ("audit_sha256", "0" * 64),
            ("run_id", "wrong"),
        ):
            path = self.root / "runs/logistic-17/run.json"
            original = path.read_bytes()
            value = json.loads(original)
            value[key] = bad
            write(path, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                external_inference.write_report(
                    **common, output_path=self.root / "bad-report.json", draws=2
                )
            self.assertFalse((self.root / "bad-report.json").exists())
            path.write_bytes(original)
        original = copy.deepcopy(self.fixture.release)
        del self.fixture.release["runs"]["logistic-73"]
        with self.assertRaisesRegex(ValueError, "matrix"):
            external_inference.write_report(
                **{**common, "release_sha256": self.fixture.save()},
                output_path=self.root / "bad-report.json",
                draws=5,
            )
        self.fixture.release = original
        self.fixture.save()
        # A completion-time mutation of a saved score must also prevent report creation.
        original_reporter = external.paired_report
        score_path = self.root / "runs/logistic-17/raw_scores.npy"

        def mutate_after_read(*args, **kwargs):
            result = original_reporter(*args, **kwargs)
            score_path.write_bytes(score_path.read_bytes() + b"changed")
            return result

        with (
            patch.object(external, "paired_report", side_effect=mutate_after_read),
            self.assertRaises(ValueError),
        ):
            external_inference.write_report(
                **common, output_path=self.root / "bad-report.json", draws=2
            )
        self.assertFalse((self.root / "bad-report.json").exists())

    def test_estimator_metadata_cannot_be_relabelled_by_self_hashes(self):
        self.fixture.add_run("cnn-scratch-unweighted", 17)
        entry = self.fixture.release["runs"]["cnn-scratch-unweighted-17"]
        path = self.root / entry["estimator"]
        original = path.read_bytes()
        for key, bad in (
            ("seed", 73),
            ("architecture", "efficientnet_b0"),
            ("fine_tune_backbone", True),
            ("loss_strategy", "pos_weight"),
            ("pos_weight", 2.0),
        ):
            with self.subTest(key=key):
                state = torch.load(path, weights_only=True)
                state[key] = bad
                torch.save(state, path)
                # Re-pin the bytes, including the policy/checkpoint ledger, so the identity
                # check must inspect the actual checkpoint rather than notice a stale hash.
                policy_path = self.root / entry["policy"]
                policy = json.loads(policy_path.read_text())
                policy["model"]["checkpoint_sha256"] = digest(path)
                write(policy_path, policy)
                training_path = self.root / entry["preprocessing"]["artifact"]
                training = json.loads(training_path.read_text())
                training["checkpoint"]["sha256"] = digest(path)
                write(training_path, training)
                rec_path = self.root / entry["path"] / "run.json"
                rec = json.loads(rec_path.read_text())
                for changed in (path, policy_path, training_path):
                    rec["artifacts"][str(changed.relative_to(rec_path.parent))] = (
                        digest(changed)
                    )
                    self.fixture.pin(str(changed.relative_to(self.root)))
                write(rec_path, rec)
                self.fixture.pin(str(rec_path.relative_to(self.root)))
                entry["run_sha256"] = digest(rec_path)
                self.reject_before_prediction()
                path.write_bytes(original)

    def test_group_lexical_values_missing_ids_and_numeric_corruption(self):
        import pandas as pd

        manifest = self.fixture.manifest
        manifest["rows"][0]["group_id"] = "N/A"
        manifest["rows"][1]["group_id"] = "患者"
        manifest["cohort_sha256"] = splitting.canonical_hash(
            {k: v for k, v in manifest.items() if k != "cohort_sha256"}
        )
        write(self.root / "manifest.json", manifest)
        write(
            self.root / "audit.json",
            {
                "clear": True,
                "manifest_sha256": digest(self.root / "manifest.json"),
                "exact": [],
                "near": [],
            },
        )
        kwargs = self.fixture.kwargs()
        external_inference.evaluate_run(**kwargs)
        frame = self.replay(kwargs)
        self.assertIn("N/A", frame.group_id.tolist())
        self.assertIn("患者", frame.group_id.tolist())
        original = (self.root / "output/predictions.csv").read_bytes()
        for column, bad in (
            ("image_id", ""),
            ("group_id", ""),
            ("prob_melanoma", "bad"),
            ("prob_melanoma", "NaN"),
            ("image_id", "001"),
        ):
            with self.subTest(column=column, bad=bad):
                changed = pd.read_csv(
                    self.root / "output/predictions.csv",
                    dtype=str,
                    keep_default_na=False,
                )
                changed.loc[0, column] = bad
                changed.to_csv(self.root / "output/predictions.csv", index=False)
                path = self.root / "output/run.json"
                status = json.loads(path.read_text())
                status["artifacts"]["predictions.csv"] = digest(
                    self.root / "output/predictions.csv"
                )
                write(path, status)
                with self.assertRaisesRegex(ValueError, "predictions"):
                    self.replay(kwargs)
                (self.root / "output/predictions.csv").write_bytes(original)

    def test_saved_scores_and_report_are_checked_before_completion(self):
        original_verify = external.verify_files
        for artifact in ("raw_scores.npy", "policy-report.json"):
            kwargs = self.fixture.kwargs(output="tampered-" + artifact)

            def verify_then_mutate(*args, **options):
                result = original_verify(*args, **options)
                path = kwargs["output"] / artifact
                if path.exists():
                    if artifact.endswith(".npy"):
                        values = np.load(path)
                        values[0] += 0.01
                        np.save(path, values)
                    else:
                        value = json.loads(path.read_text())
                        value["changed"] = True
                        write(path, value)
                return result

            with (
                self.subTest(artifact=artifact),
                patch.object(external, "verify_files", side_effect=verify_then_mutate),
            ):
                with self.assertRaises(ValueError):
                    external_inference.evaluate_run(**kwargs)
                self.assertEqual(
                    json.loads((kwargs["output"] / "run.json").read_text())["status"],
                    "failed",
                )

    def test_incomplete_policy_fails_before_prediction_even_when_repinned(self):
        import pickle

        entry = self.fixture.release["runs"]["logistic-17"]
        policy_path = self.root / entry["policy"]
        estimator_path = self.root / entry["estimator"]
        policy = json.loads(policy_path.read_text())
        del policy["calibrator"]
        write(policy_path, policy)
        with estimator_path.open("rb") as handle:
            saved = pickle.load(handle)
        saved["decision_policy"] = policy
        with estimator_path.open("wb") as handle:
            pickle.dump(saved, handle)
        rec_path = self.root / entry["path"] / "run.json"
        record = json.loads(rec_path.read_text())
        for path in (policy_path, estimator_path):
            record["artifacts"][str(path.relative_to(rec_path.parent))] = digest(path)
            self.fixture.pin(str(path.relative_to(self.root)))
        write(rec_path, record)
        self.fixture.pin(str(rec_path.relative_to(self.root)))
        entry["run_sha256"] = digest(rec_path)
        self.reject_before_prediction()

    def test_gmm_class_seed_offsets_cannot_be_relabelled(self):
        import pickle

        self.fixture.add_run("gmm", 17, policy_version=2)
        entry = self.fixture.release["runs"]["gmm-17"]
        path = self.root / entry["estimator"]
        with path.open("rb") as handle:
            saved = pickle.load(handle)
        self.assertEqual([saved["fitted"][k].random_state for k in (0, 1)], [17, 18])
        saved["fitted"][1].random_state = 17
        with path.open("wb") as handle:
            pickle.dump(saved, handle)
        self.fixture.pin(entry["estimator"])
        rec_path = self.root / entry["path"] / "run.json"
        record = json.loads(rec_path.read_text())
        record["artifacts"]["models/bayesian_gmm_model.pkl"] = digest(path)
        write(rec_path, record)
        self.fixture.pin(str(rec_path.relative_to(self.root)))
        entry["run_sha256"] = digest(rec_path)
        self.reject_before_prediction()
