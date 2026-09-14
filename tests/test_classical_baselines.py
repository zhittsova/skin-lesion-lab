"""Deterministic contracts for the classical reference models."""

import contextlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from PIL import Image
from src import bayes, classical, data, features, gmm, run_contract, splitting


class ClassicalModelTests(unittest.TestCase):
    def test_prevalence_is_training_prevalence_and_majority(self):
        fitted = classical.fit_prevalence(np.array([0, 0, 0, 1]))
        self.assertEqual(fitted["prevalence"], 0.25)
        self.assertEqual(fitted["majority_class"], 0)
        np.testing.assert_array_equal(
            classical.predict_prevalence(fitted, 3), [0.25, 0.25, 0.25]
        )
        with self.assertRaisesRegex(ValueError, "binary training labels"):
            classical.fit_prevalence(np.array([0, 2]))

    def test_logistic_separates_and_round_trips_on_seeded_data(self):
        rng = np.random.default_rng(17)
        train_x = np.r_[rng.normal(-2, 0.2, (16, 2)), rng.normal(2, 0.2, (16, 2))]
        train_y = np.r_[np.zeros(16, dtype=int), np.ones(16, dtype=int)]
        selection_x = np.array([[-2.1, -1.8], [1.8, 2.2]])
        selection_y = np.array([0, 1])
        model, info = classical.fit_logistic(
            train_x, train_y, selection_x, selection_y, seed=17
        )
        self.assertEqual(info["selected_c"], 0.01)  # Equal AUC chooses smaller C.
        scores = classical.predict_logistic(model, selection_x)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(scores[0] < 0.5 < scores[1])
        np.testing.assert_allclose(
            scores,
            classical.predict_logistic(pickle.loads(pickle.dumps(model)), selection_x),
            atol=1e-12,
        )

    def test_logistic_scaler_ignores_heldout_perturbation(self):
        train_x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        train_y = np.array([0, 0, 1, 1])
        selection_y = np.array([0, 1])
        first, _ = classical.fit_logistic(
            train_x, train_y, np.array([[-1.5], [1.5]]), selection_y, seed=42
        )
        second, _ = classical.fit_logistic(
            train_x, train_y, np.array([[-150.0], [150.0]]), selection_y, seed=42
        )
        np.testing.assert_array_equal(first["scaler"].mean_, second["scaler"].mean_)
        np.testing.assert_array_equal(first["scaler"].scale_, second["scaler"].scale_)

    def test_gmm_rejects_invalid_settings_and_bounds_components(self):
        x = np.array([[-1.0], [-0.5], [0.5], [1.0]])
        y = np.array([0, 0, 1, 1])
        for bad in (0, -1, 1.5):
            with self.subTest(max_components=bad):
                with self.assertRaisesRegex(ValueError, "max_components"):
                    gmm.train_class_gmms_auto(x, y, max_components=bad)
        with self.assertRaisesRegex(ValueError, "reg_covar"):
            gmm.train_class_gmms_auto(x, y, reg_covar=0)
        with self.assertRaisesRegex(ValueError, "n_components"):
            gmm.train_class_gmm(x[y == 0], 3)
        models, info = gmm.train_class_gmms_auto(
            x, y, max_components=10, cv_type="diag", random_state=17
        )
        self.assertEqual(set(models), {0, 1})
        for value in info.values():
            self.assertEqual(value["bic_info"]["component_range"], [1, 2])
            self.assertEqual(value["requested_max_components"], 10)
            self.assertEqual(value["effective_max_components"], 2)

    def test_gmm_nonconvergence_is_explicit(self):
        x = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
        with self.assertRaisesRegex(ValueError, "did not converge"):
            gmm.select_optimal_components(
                x, max_components=2, cv_type="diag", max_iter=1
            )

    def test_duplicate_features_can_converge_with_initialization_warning(self):
        x = np.zeros((4, 1))
        selected, info = gmm.select_optimal_components(
            x, max_components=2, cv_type="diag", random_state=17
        )
        self.assertIn(selected, [1, 2])
        self.assertTrue(np.isfinite(info["bic_scores"]).all())
        self.assertTrue(info["initialization_warnings"]["2"])


class ClassicalRunTests(unittest.TestCase):
    def test_three_models_share_manifest_and_reload_scores(self):
        import train_pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            rows = []
            rng = np.random.default_rng(73)
            for label in (0, 1):
                for index in range(20):
                    image_id = f"I{label}_{index:03d}"
                    center = [220, 30, 30] if label else [30, 220, 30]
                    pixels = np.clip(rng.normal(center, 6, (16, 16, 3)), 0, 255).astype(
                        np.uint8
                    )
                    Image.fromarray(pixels).save(images / f"{image_id}.jpg")
                    rows.append(
                        {
                            "image_id": image_id,
                            "lesion_id": f"L{label}_{index // 2:03d}",
                            "dx": "mel" if label else "nv",
                        }
                    )
            metadata = root / "metadata.csv"
            pd.DataFrame(rows).to_csv(metadata, index=False)
            frame, _, _ = data.prepare_dataset(
                str(metadata),
                str(images),
                source="ham10000",
                attrition_path=root / "attrition.json",
            )
            manifest = splitting.create_manifest(frame)
            manifest_file = root / "manifest.json"
            splitting.save_manifest(manifest, manifest_file)
            expected = manifest["partitions"]["development"]
            for model in ("prevalence", "logistic", "gmm"):
                args = [
                    "train_pipeline.py",
                    "--source",
                    "ham10000",
                    "--metadata-path",
                    str(metadata),
                    "--images-dir",
                    str(images),
                    "--split-manifest",
                    str(manifest_file),
                    "--runs-dir",
                    str(root / "runs"),
                    "--run-id",
                    f"{model}-seed-73",
                    "--model",
                    model,
                    "--seed",
                    "73",
                ]
                if model == "gmm":
                    args += [
                        "--gmm-max-components",
                        "1",
                        "--gmm-covariance-type",
                        "diag",
                    ]
                with (
                    patch("sys.argv", args),
                    patch.object(train_pipeline, "plots", Mock()),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    train_pipeline.main()
                run = root / "runs" / f"{model}-seed-73"
                record, shared = run_contract.validate_run(run)
                self.assertEqual(record["status"], "completed")
                self.assertEqual(record["pipeline"], f"classical_{model}")
                self.assertEqual(record["split_hash"], manifest["split_hash"])
                self.assertEqual(
                    shared.loc[shared.role == "development", "image_id"].tolist(),
                    expected,
                )
                producer = pd.read_csv(
                    run / "results" / "tables" / "predictions_development.csv"
                )
                self.assertTrue(np.isfinite(producer.prob_melanoma).all())
                self.assertTrue(producer.prob_melanoma.between(0, 1).all())
                if model == "prevalence":
                    self.assertEqual(shared.prob_melanoma.nunique(), 1)
                else:
                    self.assertGreater(
                        producer.loc[producer.target == 1, "prob_melanoma"].mean(),
                        producer.loc[producer.target == 0, "prob_melanoma"].mean(),
                    )
                model_file = (
                    "bayesian_gmm_model.pkl" if model == "gmm" else f"{model}_model.pkl"
                )
                with (run / "models" / model_file).open("rb") as handle:
                    fitted = pickle.load(handle)
                if model == "prevalence":
                    reloaded = classical.predict_prevalence(
                        fitted["fitted"], len(expected)
                    )
                    self.assertEqual(
                        fitted["fitted"]["prevalence"],
                        float(shared.loc[shared.role == "train", "target"].mean()),
                    )
                else:
                    hsv = [
                        data.load_image_hsv(str(images / f"{image_id}.jpg"), (256, 256))
                        for image_id in expected
                    ]
                    raw = features.compute_hsv_histograms_batch(hsv)
                    if model == "logistic":
                        reloaded = classical.predict_logistic(fitted["fitted"], raw)
                    else:
                        transformed = features.apply_standardization(
                            raw, fitted["train_mean"], fitted["train_std"]
                        )
                        likelihoods = gmm.compute_class_likelihoods(
                            fitted["fitted"], transformed
                        )
                        reloaded = bayes.get_melanoma_probability(
                            bayes.compute_posterior_probabilities(
                                likelihoods, fitted["class_priors"]
                            )
                        )
                np.testing.assert_allclose(
                    reloaded, producer.prob_melanoma.to_numpy(), atol=1e-12
                )
                summary = json.loads(
                    (run / "results" / "metrics_summary.json").read_text()
                )
                self.assertEqual(summary["split"]["split_hash"], manifest["split_hash"])

            invalid_args = [
                "train_pipeline.py",
                "--source",
                "ham10000",
                "--metadata-path",
                str(metadata),
                "--images-dir",
                str(images),
                "--split-manifest",
                str(manifest_file),
                "--runs-dir",
                str(root / "runs"),
                "--run-id",
                "invalid-gmm",
                "--model",
                "gmm",
                "--gmm-max-components",
                "0",
            ]
            with (
                patch("sys.argv", invalid_args),
                patch.object(train_pipeline, "plots", Mock()),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(ValueError, "max_components"),
            ):
                train_pipeline.main()
            failed = json.loads(
                (root / "runs" / "invalid-gmm" / "run.json").read_text()
            )
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["failure"]["stage"], "training")
            self.assertEqual(failed["failure"]["reason"], "invalid_gmm_configuration")

            for value in ("nan", "inf"):
                with (
                    self.subTest(reg_covar=value),
                    patch(
                        "sys.argv",
                        [
                            *invalid_args[:-2],
                            "--run-id",
                            f"invalid-reg-{value}",
                            "--gmm-reg-covar",
                            value,
                        ],
                    ),
                    patch.object(train_pipeline, "plots", Mock()),
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaisesRegex(ValueError, "reg_covar"),
                ):
                    train_pipeline.main()
                failed = json.loads(
                    (root / "runs" / f"invalid-reg-{value}" / "run.json").read_text()
                )
                self.assertEqual(failed["status"], "failed")
                self.assertEqual(
                    failed["failure"]["reason"], "invalid_gmm_configuration"
                )
                self.assertEqual(failed["config"]["gmm_reg_covar"], value)


if __name__ == "__main__":
    unittest.main()
