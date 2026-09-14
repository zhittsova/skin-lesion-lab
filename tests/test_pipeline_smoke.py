"""Synthetic CPU integration of the shared allocation and output membership."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from src import data, gmm, splitting


class SharedManifestSmokeTests(unittest.TestCase):
    def test_both_pipelines_and_summary_preserve_the_same_development_ids(self):
        import summarize_results
        import train_deep_pipeline
        import train_pipeline

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with (
            tempfile.TemporaryDirectory() as temporary,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            rows = []
            rng = np.random.default_rng(123)
            for label in (0, 1):
                for i in range(20):
                    image_id = f"I{label}_{i:03d}"
                    pixels = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(images / f"{image_id}.jpg")
                    rows.append(
                        {
                            "image_id": image_id,
                            "lesion_id": f"L{label}_{i // 2:03d}",
                            "dx": "mel" if label else "nv",
                        }
                    )
            metadata = root / "metadata.csv"
            pd.DataFrame(rows).to_csv(metadata, index=False)
            frame, _, _ = data.prepare_dataset(
                str(metadata),
                str(images),
                source="ham10000",
                attrition_path=root / "audit.json",
            )
            manifest = splitting.create_manifest(frame)
            path = root / "manifest.json"
            splitting.save_manifest(manifest, path)
            shared = [
                "--source",
                "ham10000",
                "--metadata-path",
                str(metadata),
                "--images-dir",
                str(images),
                "--split-manifest",
                str(path),
            ]
            original_fit = gmm.train_class_gmms_auto

            def tiny_fit(x, y, **kwargs):
                # Limit only this fixture's search cost; use the real fitter and scorer.
                return original_fit(
                    x, y, max_components=1, cv_type="diag", random_state=42
                )

            classical = root / "classical"
            models = root / "classical_models"
            with (
                patch(
                    "sys.argv",
                    [
                        "train_pipeline.py",
                        *shared,
                        "--results-dir",
                        str(classical),
                        "--models-dir",
                        str(models),
                    ],
                ),
                patch.object(train_pipeline, "plots", Mock()),
                patch.object(gmm, "train_class_gmms_auto", side_effect=tiny_fit),
            ):
                train_pipeline.main()
            deep_results = root / "deep"
            with (
                patch(
                    "sys.argv",
                    [
                        "train_deep_pipeline.py",
                        *shared,
                        "--results-dir",
                        str(deep_results),
                        "--models-dir",
                        str(root / "deep_models"),
                        "--runs-dir",
                        str(root / "deep_runs"),
                        "--epochs",
                        "1",
                        "--mc-samples",
                        "2",
                        "--image-size",
                        "32",
                        "--batch-size",
                        "8",
                        "--seed",
                        "73",
                    ],
                ),
                patch.object(
                    train_deep_pipeline.deep,
                    "get_default_device",
                    return_value=torch.device("cpu"),
                ),
                patch.object(train_deep_pipeline, "plots", Mock()),
                patch.object(train_deep_pipeline, "plot_uncertainty_distribution"),
                patch.object(train_deep_pipeline, "plot_mean_vs_epistemic_uncertainty"),
            ):
                train_deep_pipeline.main()
            for location, filename in (
                (classical, "predictions_development.csv"),
                (deep_results, "small_cnn_predictions_development.csv"),
            ):
                predictions = pd.read_csv(location / "tables" / filename)
                self.assertEqual(
                    predictions.image_id.tolist(), manifest["partitions"]["development"]
                )
            for location, filename in (
                (classical, "metrics_summary.json"),
                (deep_results, "small_cnn_metrics_summary.json"),
            ):
                summary = json.loads((location / filename).read_text())
                self.assertEqual(summary["split"]["split_hash"], manifest["split_hash"])
                self.assertEqual(set(summary["split"]["splits"]), set(splitting.ROLES))
            with (
                patch(
                    "sys.argv",
                    [
                        "summarize_results.py",
                        *shared,
                        "--results-dir",
                        str(classical),
                        "--model-path",
                        str(models / "bayesian_gmm_model.pkl"),
                    ],
                ),
                patch.object(summarize_results, "plots", Mock()),
            ):
                summarize_results.main()
            self.assertEqual(
                json.loads((classical / "metrics_summary.json").read_text())["split"][
                    "split_hash"
                ],
                manifest["split_hash"],
            )
