"""Frozen estimator and checkpoint inference on generated external images."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from src import calibration, classical, deep, external_inference, run_contract


class ExternalInferenceTests(unittest.TestCase):
    def test_saved_estimators_predict_without_fit_and_keep_checkpoint_bytes(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models").mkdir()
            ids = ["external-a", "external-b"]
            for i, name in enumerate(ids):
                Image.new("RGB", (32, 32), (20 + 100 * i, 35, 75)).save(
                    root / f"{name}.jpg"
                )
            x = np.random.default_rng(5).random((8, 128))
            y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
            fitted, _ = classical.fit_logistic(x, y, x, y)
            with (root / "models/logistic_model.pkl").open("wb") as f:
                pickle.dump({"fitted": fitted}, f)
            deep.set_seed(17)
            model = deep.build_model("small_cnn")
            checkpoint = root / "models/small_cnn_mc_dropout.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": "small_cnn",
                    "image_size": 32,
                    "dropout": 0.3,
                    "seed": 17,
                },
                checkpoint,
            )
            digest = run_contract.sha256(checkpoint)
            config = {
                "architecture": "small_cnn",
                "dropout": 0.3,
                "image_size": 32,
                "seed": 17,
                "batch_size": 2,
                "mc_samples": 2,
                "num_workers": 0,
            }
            with (
                patch.object(
                    classical, "fit_logistic", side_effect=AssertionError("fit called")
                ),
                patch.object(
                    calibration,
                    "fit_policy",
                    side_effect=AssertionError("calibration fit called"),
                ),
                patch.object(
                    deep,
                    "train_one_epoch",
                    side_effect=AssertionError("training called"),
                ),
            ):
                logistic = external_inference.predict_raw(
                    root,
                    {"pipeline": "classical_logistic", "config": {"seed": 17}},
                    ids,
                    root,
                    "cpu",
                )
                first = external_inference.predict_raw(
                    root,
                    {"pipeline": "deep_small_cnn", "config": config},
                    ids,
                    root,
                    "cpu",
                )
                second = external_inference.predict_raw(
                    root,
                    {"pipeline": "deep_small_cnn", "config": config},
                    ids,
                    root,
                    "cpu",
                )
            self.assertEqual(logistic.shape, (2,))
            self.assertEqual(first.shape, (2, 2))
            np.testing.assert_array_equal(first, second)
            self.assertTrue(np.isfinite(first).all())
            self.assertEqual(run_contract.sha256(checkpoint), digest)

    def test_evaluation_writes_validated_run_and_rejects_stale_or_reused_inputs(self):
        import json

        import pandas as pd
        from src import splitting

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = root / "fitted"
            (model_root / "models").mkdir(parents=True)
            rows = []
            for i in range(4):
                image_id = f"external{i}"
                Image.new("RGB", (8, 8), (i * 25, 20, 35)).save(
                    root / f"{image_id}.jpg"
                )
                rows.append(
                    {
                        "image_id": image_id,
                        "group_id": f"p{i}",
                        "target": i // 2,
                        "reason": "retained",
                        "image_sha256": run_contract.sha256(root / f"{image_id}.jpg"),
                    }
                )
            manifest = {"schema_version": 1, "purpose": "external", "rows": rows}
            manifest["cohort_sha256"] = splitting.canonical_hash(manifest)
            (root / "manifest.json").write_text(json.dumps(manifest))
            manifest_hash = run_contract.sha256(root / "manifest.json")
            policy = calibration.fit_policy(
                np.array([0, 0, 1, 1]),
                np.array([0.1, 0.2, 0.7, 0.9]),
                image_ids=["fit-a", "fit-b", "fit-c", "fit-d"],
                split_hash="frozen",
            )
            (model_root / "models/decision_policy.json").write_text(json.dumps(policy))
            config = {"seed": 17}
            record = {
                "status": "completed",
                "pipeline": "classical_logistic",
                "config": config,
                "config_sha256": splitting.canonical_hash(config),
                "environment": {"software": {}},
            }
            (model_root / "run.json").write_text(json.dumps(record))
            release = {
                "schema_version": 1,
                "runs": {"logistic-17": {"path": "fitted"}},
                "files": {
                    str(p.relative_to(root)): run_contract.sha256(p)
                    for p in model_root.rglob("*")
                    if p.is_file()
                },
            }
            (root / "release.json").write_text(json.dumps(release))
            release_hash = run_contract.sha256(root / "release.json")
            audit = {
                "clear": True,
                "manifest_sha256": manifest_hash,
                "exact": [],
                "near": [],
            }
            (root / "audit.json").write_text(json.dumps(audit))
            kwargs = dict(
                root=root,
                release_path=root / "release.json",
                release_sha256=release_hash,
                manifest_path=root / "manifest.json",
                manifest_sha256=manifest_hash,
                audit_path=root / "audit.json",
                audit_sha256=run_contract.sha256(root / "audit.json"),
                images_dir=root,
                run_id="logistic-17",
                output=root / "output",
            )
            with patch.object(
                external_inference,
                "predict_raw",
                return_value=np.array([0.1, 0.2, 0.7, 0.9]),
            ):
                external_inference.evaluate_run(**kwargs)
            validated = external_inference.validate_predictions(
                root / "output",
                release,
                release_hash,
                manifest,
                manifest_hash,
                root,
                expected_run_id="logistic-17",
                audit_sha256=kwargs["audit_sha256"],
            )
            self.assertEqual(validated.target.tolist(), [0, 0, 1, 1])
            with self.assertRaisesRegex(ValueError, "run identity"):
                external_inference.validate_predictions(
                    root / "output",
                    release,
                    release_hash,
                    manifest,
                    manifest_hash,
                    root,
                    expected_run_id="logistic-42",
                    audit_sha256=kwargs["audit_sha256"],
                )
            with self.assertRaises(FileExistsError):
                external_inference.evaluate_run(**kwargs)
            frame = pd.read_csv(root / "output/predictions.csv")
            frame.loc[0, "target"] = 1
            frame.to_csv(root / "output/predictions.csv", index=False)
            state = json.loads((root / "output/run.json").read_text())
            state["artifacts"]["predictions.csv"] = run_contract.sha256(
                root / "output/predictions.csv"
            )
            (root / "output/run.json").write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "predictions"):
                external_inference.validate_predictions(
                    root / "output",
                    release,
                    release_hash,
                    manifest,
                    manifest_hash,
                    root,
                    expected_run_id="logistic-17",
                    audit_sha256=kwargs["audit_sha256"],
                )
            with self.assertRaisesRegex(ValueError, "external input hash"):
                external_inference.evaluate_run(
                    **{**kwargs, "manifest_sha256": "0" * 64, "output": root / "bad"}
                )

    def test_reporting_requires_matching_audit_and_run_identity(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "clear": True,
                        "manifest_sha256": "manifest",
                        "exact": [],
                        "near": [],
                    }
                )
            )
            digest = run_contract.sha256(audit)
            external_inference.validate_audit(audit, digest, "manifest")
            for audit_digest, manifest_digest in [
                ("wrong", "manifest"),
                (digest, "changed"),
            ]:
                with self.assertRaisesRegex(ValueError, "audit"):
                    external_inference.validate_audit(
                        audit, audit_digest, manifest_digest
                    )
            audit.write_text(
                json.dumps(
                    {
                        "clear": True,
                        "manifest_sha256": "manifest",
                        "exact": [],
                        "near": [{"unresolved": True}],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "audit"):
                external_inference.validate_audit(
                    audit, run_contract.sha256(audit), "manifest"
                )
