"""Parity, identity, and memory-bound tests for profiling datasets."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from src import deep
from src.profile_data import ProfileDataset


def write_jpeg(path, seed, size=(43, 37)):
    pixels = np.random.default_rng(seed).integers(
        0, 256, (size[1], size[0], 3), dtype=np.uint8
    )
    Image.fromarray(pixels).save(path, quality=91)


class ProfileDatasetTests(unittest.TestCase):
    def test_cached_modes_match_project_dataset_across_train_eval_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jpeg(root / "a.jpg", 1)
            write_jpeg(root / "b.jpg", 2)
            records = [
                {"image_id": "a", "filename": "a.jpg", "label": 0},
                {"image_id": "b", "filename": "b.jpg", "label": 1},
            ]
            for train in (False, True):
                baseline = deep.SkinLesionImageDataset(
                    ["a", "b"],
                    [0, 1],
                    root,
                    deep.build_transforms(32, train),
                )
                for mode in ("none", "decoded", "resized"):
                    dataset = ProfileDataset(
                        records, root, image_size=32, train=train, cache_mode=mode
                    )
                    for seed in (7, 19):
                        for index in range(2):
                            deep.set_seed(seed + index)
                            expected = baseline[index]
                            deep.set_seed(seed + index)
                            actual = dataset[index]
                            torch.testing.assert_close(
                                actual[0], expected[0], rtol=0, atol=0
                            )
                            self.assertEqual(actual[1:], expected[1:])

    def test_hardlink_aliases_keep_identity_and_share_cached_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jpg"
            first, second = root / "alias-1.jpg", root / "alias-2.jpg"
            write_jpeg(source, 3, size=(40, 35))
            os.link(source, first)
            os.link(source, second)
            dataset = ProfileDataset(
                [
                    {"image_id": "first", "filename": first.name, "label": 0},
                    {"image_id": "second", "filename": second.name, "label": 1},
                ],
                root,
                image_size=32,
                train=False,
                cache_mode="decoded",
            )
            self.assertEqual(len(dataset._cache), 1)
            self.assertEqual(dataset.cache_bytes, 40 * 35 * 3)
            first_item, second_item = dataset[0], dataset[1]
            torch.testing.assert_close(first_item[0], second_item[0], rtol=0, atol=0)
            self.assertEqual((first_item[2], second_item[2]), ("first", "second"))
            self.assertEqual((first_item[3], second_item[3]), (str(first), str(second)))

    def test_stochastic_access_never_mutates_cached_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jpeg(root / "a.jpg", 4)
            for mode in ("decoded", "resized"):
                dataset = ProfileDataset(
                    [{"image_id": "a", "filename": "a.jpg", "label": 0}],
                    root,
                    image_size=32,
                    train=True,
                    cache_mode=mode,
                )
                cached = next(iter(dataset._cache.values()))
                before = cached.tobytes()
                augmented = []
                for seed in range(4):
                    deep.set_seed(seed)
                    augmented.append(dataset[0][0])
                self.assertEqual(cached.tobytes(), before)
                self.assertTrue(
                    any(not torch.equal(augmented[0], item) for item in augmented[1:])
                )

    def test_budget_refusal_happens_before_pixel_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jpeg(root / "a.jpg", 5, size=(100, 80))
            record = [{"image_id": "a", "filename": "a.jpg", "label": 0}]
            with (
                patch(
                    "PIL.Image.Image.convert",
                    side_effect=AssertionError("pixel decode must not start"),
                ),
                self.assertRaises(MemoryError),
            ):
                ProfileDataset(
                    record,
                    root,
                    image_size=32,
                    train=False,
                    cache_mode="decoded",
                    cache_max_bytes=100 * 80 * 3 - 1,
                )

    def test_instrumentation_is_optional_and_reports_all_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jpeg(root / "a.jpg", 6)
            records = [{"image_id": "a", "filename": "a.jpg", "label": 1}]
            plain = ProfileDataset(records, root, 32, False, instrument=False)
            timed = ProfileDataset(records, root, 32, False, instrument=True)
            self.assertEqual(len(plain[0]), 4)
            item = timed[0]
            self.assertEqual(len(item), 5)
            self.assertEqual(
                set(item[4]),
                {"decode_seconds", "resize_seconds", "augment_seconds"},
            )
            self.assertTrue(all(value >= 0 for value in item[4].values()))

    def test_records_and_paths_are_strictly_validated(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            root, outside = Path(tmp), Path(out)
            write_jpeg(root / "a.jpg", 7)
            write_jpeg(outside / "outside.jpg", 8)
            (root / "escape.jpg").symlink_to(outside / "outside.jpg")
            valid = [{"image_id": "a", "filename": "a.jpg", "label": 0}]
            bad_records = [
                [],
                [{**valid[0], "extra": 1}],
                [{**valid[0], "label": True}],
                [{**valid[0], "filename": "../a.jpg"}],
                [{**valid[0], "filename": "escape.jpg"}],
                [valid[0], {**valid[0], "filename": "a.jpg"}],
            ]
            for records in bad_records:
                with self.subTest(records=records), self.assertRaises(ValueError):
                    ProfileDataset(records, root, 32, False)
            for kwargs in [
                {"image_size": 31},
                {"train": 1},
                {"cache_mode": "other"},
                {"instrument": 1},
                {"cache_max_bytes": 0},
            ]:
                values = {
                    "image_size": 32,
                    "train": False,
                    "cache_mode": "none",
                    "instrument": False,
                    "cache_max_bytes": 1024,
                    **kwargs,
                }
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    ProfileDataset(valid, root, **values)


if __name__ == "__main__":
    unittest.main()
