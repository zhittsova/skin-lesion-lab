"""Tests for the bounded resized-image cache used by deep training."""

import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from src import deep
from torchvision import transforms


def write_jpeg(path: Path, seed: int, size=(43, 37)) -> None:
    pixels = np.random.default_rng(seed).integers(
        0, 256, (size[1], size[0], 3), dtype=np.uint8
    )
    Image.fromarray(pixels).save(path, quality=91)


class ResizedImageCacheTests(unittest.TestCase):
    def test_cache_matches_uncached_eval_and_train_across_repeated_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jpeg(root / "a.jpg", 1)
            write_jpeg(root / "b.jpg", 2, size=(39, 51))
            cache = deep.ResizedImageCache(
                ["a", "b"], root, image_size=32, max_bytes=2 * 32 * 32 * 3
            )

            for train in (False, True):
                transform = deep.build_transforms(32, train=train)
                uncached = deep.SkinLesionImageDataset(
                    ["a", "b"], [0, 1], root, transform
                )
                cached = deep.SkinLesionImageDataset(
                    ["a", "b"], [0, 1], root, transform, resized_cache=cache
                )

                for epoch_seed in (7, 19):
                    deep.set_seed(epoch_seed)
                    expected = [uncached[index] for index in (0, 1, 0)]
                    expected_python_state = random.getstate()
                    expected_numpy_state = np.random.get_state()
                    expected_rng_state = torch.get_rng_state().clone()

                    deep.set_seed(epoch_seed)
                    actual = [cached[index] for index in (0, 1, 0)]
                    actual_python_state = random.getstate()
                    actual_numpy_state = np.random.get_state()
                    actual_rng_state = torch.get_rng_state().clone()

                    for expected_item, actual_item in zip(
                        expected, actual, strict=True
                    ):
                        torch.testing.assert_close(
                            actual_item[0], expected_item[0], rtol=0, atol=0
                        )
                        self.assertEqual(actual_item[1:], expected_item[1:])
                    torch.testing.assert_close(
                        actual_rng_state, expected_rng_state, rtol=0, atol=0
                    )
                    self.assertEqual(actual_python_state, expected_python_state)
                    self.assertEqual(actual_numpy_state[0], expected_numpy_state[0])
                    np.testing.assert_array_equal(
                        actual_numpy_state[1], expected_numpy_state[1]
                    )
                    self.assertEqual(actual_numpy_state[2:], expected_numpy_state[2:])

    def test_each_access_uses_a_copy_of_cached_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jpeg(root / "a.jpg", 3)
            cache = deep.ResizedImageCache(
                ["a"], root, image_size=32, max_bytes=32 * 32 * 3
            )
            cached = cache._images["a"]
            before = cached.tobytes()

            mutating_transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.Lambda(
                        lambda image: (image.paste((0, 0, 0), (0, 0, 8, 8)), image)[1]
                    ),
                    transforms.ToTensor(),
                ]
            )
            dataset = deep.SkinLesionImageDataset(
                ["a"], [0], root, mutating_transform, resized_cache=cache
            )
            first = dataset[0][0]
            second = dataset[0][0]

            self.assertEqual(cached.tobytes(), before)
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_budget_refusal_happens_before_image_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jpeg(root / "a.jpg", 4)
            with (
                patch("src.image_cache.Image.open") as image_open,
                self.assertRaisesRegex(MemoryError, "3072 bytes"),
            ):
                deep.ResizedImageCache(
                    ["a"], root, image_size=32, max_bytes=32 * 32 * 3 - 1
                )
            image_open.assert_not_called()

    def test_cache_contract_rejects_root_size_id_and_transform_mismatches(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as other,
        ):
            root, other_root = Path(tmp), Path(other)
            write_jpeg(root / "a.jpg", 5)
            write_jpeg(other_root / "a.jpg", 6)
            cache = deep.ResizedImageCache(
                ["a"], root, image_size=32, max_bytes=32 * 32 * 3
            )
            cases = [
                (["a"], other_root, deep.build_transforms(32, False)),
                (["a"], root, deep.build_transforms(48, False)),
                (["missing"], root, deep.build_transforms(32, False)),
                (
                    ["a"],
                    root,
                    transforms.Compose([transforms.ToTensor()]),
                ),
                (
                    ["a"],
                    root,
                    transforms.Compose(
                        [
                            transforms.Resize(
                                (32, 32),
                                interpolation=transforms.InterpolationMode.NEAREST,
                            ),
                            transforms.ToTensor(),
                        ]
                    ),
                ),
            ]
            for image_ids, images_dir, transform in cases:
                with self.subTest(transform=transform), self.assertRaises(ValueError):
                    deep.SkinLesionImageDataset(
                        image_ids,
                        [0],
                        images_dir,
                        transform,
                        resized_cache=cache,
                    )

    def test_cache_counts_every_cohort_item_without_inode_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jpeg(root / "a.jpg", 7)
            os.link(root / "a.jpg", root / "b.jpg")
            cache = deep.ResizedImageCache(
                ["a", "b"], root, image_size=32, max_bytes=2 * 32 * 32 * 3
            )

            self.assertEqual(len(cache), 2)
            self.assertEqual(cache.image_count, 2)
            self.assertEqual(cache.cache_bytes, 2 * 32 * 32 * 3)
            self.assertEqual(cache.image_size, 32)
            self.assertEqual(cache.max_bytes, 2 * 32 * 32 * 3)
            self.assertGreaterEqual(cache.cache_build_seconds, 0.0)
            self.assertEqual(set(cache._images), {"a", "b"})
            self.assertIsNot(cache._images["a"], cache._images["b"])

    def test_cache_inputs_are_strictly_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jpeg(root / "a.jpg", 8)
            invalid = [
                ([], root, 32, 1),
                ("a", root, 32, 10_000),
                (["a", "a"], root, 32, 10_000),
                (["../a"], root, 32, 10_000),
                (["a"], root, 31, 10_000),
                (["a"], root, 32, 0),
                (["a"], root, 32, True),
            ]
            for image_ids, images_dir, image_size, max_bytes in invalid:
                with self.subTest(image_ids=image_ids), self.assertRaises(ValueError):
                    deep.ResizedImageCache(
                        image_ids,
                        images_dir,
                        image_size=image_size,
                        max_bytes=max_bytes,
                    )


if __name__ == "__main__":
    unittest.main()
