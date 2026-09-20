"""Compare cached and uncached training through the public pipeline."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scripts.smoke_experiment import make_cohort
from src import data, splitting


class CachedPipelineTests(unittest.TestCase):
    def test_cache_preserves_candidate_checkpoints_and_mc_predictions(self):
        import train_deep_pipeline as pipeline

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, images = make_cohort(root)
            frame, _, _ = data.prepare_dataset(
                str(metadata),
                str(images),
                source="ham10000",
                attrition_path=root / "attrition.json",
            )
            manifest = splitting.create_manifest(frame)
            split = root / "split.json"
            splitting.save_manifest(manifest, split)
            for mode in ("none", "resized"):
                argv = [
                    "train_deep_pipeline.py",
                    "--source",
                    "ham10000",
                    "--metadata-path",
                    str(metadata),
                    "--images-dir",
                    str(images),
                    "--split-manifest",
                    str(split),
                    "--runs-dir",
                    str(root / "runs"),
                    "--run-id",
                    mode,
                    "--device",
                    "cpu",
                    "--image-size",
                    "32",
                    "--epochs",
                    "2",
                    "--batch-size",
                    "8",
                    "--mc-samples",
                    "3",
                    "--learning-rates",
                    "0.001",
                    "0.0003",
                    "--seed",
                    "17",
                    "--image-cache",
                    mode,
                ]
                with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    pipeline.main()
            baseline, cached = (root / "runs" / mode for mode in ("none", "resized"))
            searches = [
                json.loads((p / "results/deep_candidate_search.json").read_text())
                for p in (baseline, cached)
            ]
            self.assertEqual(searches[0]["winner"], searches[1]["winner"])
            for left, right in zip(
                searches[0]["candidates"], searches[1]["candidates"], strict=True
            ):
                self.assertEqual(left["history"], right["history"])
                states = [
                    torch.load(p / item["checkpoint"], weights_only=True)[
                        "model_state_dict"
                    ]
                    for p, item in ((baseline, left), (cached, right))
                ]
                for key in states[0]:
                    self.assertTrue(torch.equal(states[0][key], states[1][key]), key)
            for role in ("calibration", "development"):
                filename = f"small_cnn_{role}_mc_probabilities.npy"
                np.testing.assert_array_equal(
                    np.load(baseline / "arrays" / filename),
                    np.load(cached / "arrays" / filename),
                )
            cache = json.loads((cached / "results/image_cache.json").read_text())
            self.assertEqual(cache["mode"], "resized")
            self.assertEqual(cache["image_count"], 40)
            self.assertEqual(cache["pixel_bytes"], 40 * 32 * 32 * 3)
            self.assertEqual(cache["builds"], 1)
            record = json.loads((cached / "run.json").read_text())
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["config"]["image_cache"], "resized")

    def test_cached_worker_copies_and_invalid_caps_are_rejected(self):
        import train_deep_pipeline as pipeline

        for extra, message in (
            (["--num-workers", "1"], "num_workers"),
            (["--image-cache-max-bytes", "0"], "cache"),
        ):
            with (
                self.subTest(extra=extra),
                patch(
                    "sys.argv",
                    [
                        "train_deep_pipeline.py",
                        "--source",
                        "ham10000",
                        "--split-manifest",
                        "unused.json",
                        "--image-cache",
                        "resized",
                        *extra,
                    ],
                ),
            ):
                args = pipeline.parse_args()
                with self.assertRaisesRegex(ValueError, message):
                    pipeline.validate_training_config(args)
