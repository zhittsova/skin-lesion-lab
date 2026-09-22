"""Dataset variants for measuring JPEG, resize, and augmentation bottlenecks."""

import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src import deep

_RECORD_KEYS = {"image_id", "filename", "label"}
_CACHE_MODES = {"none", "decoded", "resized"}


class ProfileDataset(Dataset):
    """Load project images with optional inode-deduplicated RGB pixel caches.

    ``none`` opens and decodes the JPEG on every access. ``decoded`` caches raw
    RGB pixels. ``resized`` also applies the deterministic leading Resize once.
    Every cached access transforms a copy so stochastic transforms cannot alter
    shared pixels. Instrumented items add timing fields for materialization,
    resize, and all transforms after resize.
    """

    def __init__(
        self,
        records,
        images_dir: Path,
        image_size: int,
        train: bool,
        cache_mode="none",
        instrument=False,
        cache_max_bytes=512 * 1024**2,
    ):
        if not isinstance(records, list) or not records:
            raise ValueError("records must be a nonempty list")
        if (
            isinstance(image_size, bool)
            or not isinstance(image_size, int)
            or image_size < 32
        ):
            raise ValueError("image_size must be an integer >= 32")
        if not isinstance(train, bool):
            raise ValueError("train must be boolean")
        if cache_mode not in _CACHE_MODES:
            raise ValueError("cache_mode must be none, decoded, or resized")
        if not isinstance(instrument, bool):
            raise ValueError("instrument must be boolean")
        if (
            isinstance(cache_max_bytes, bool)
            or not isinstance(cache_max_bytes, int)
            or cache_max_bytes < 1
        ):
            raise ValueError("cache_max_bytes must be a positive integer")

        supplied_root = Path(images_dir)
        try:
            root = supplied_root.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as error:
            raise ValueError("images_dir must exist") from error
        if not root.is_dir():
            raise ValueError("images_dir must be a directory")

        entries = []
        image_ids, alias_paths = set(), set()
        for record in records:
            if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
                raise ValueError("records require exactly image_id, filename, label")
            image_id, filename, label = (
                record["image_id"],
                record["filename"],
                record["label"],
            )
            if not isinstance(image_id, str) or not image_id or image_id in image_ids:
                raise ValueError("image IDs must be nonempty and unique")
            relative = Path(filename) if isinstance(filename, str) else None
            if (
                relative is None
                or not filename
                or relative.name != filename
                or len(relative.parts) != 1
            ):
                raise ValueError("filename must be one relative filename")
            if (
                isinstance(label, bool)
                or not isinstance(label, int)
                or label not in (0, 1)
            ):
                raise ValueError("labels must be integer zero or one")
            alias = supplied_root / filename
            try:
                resolved = alias.resolve(strict=True)
            except (FileNotFoundError, RuntimeError) as error:
                raise ValueError(f"image does not exist: {filename}") from error
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise ValueError("image path escapes images_dir or is not a file")
            alias_key = str(alias.absolute())
            if alias_key in alias_paths:
                raise ValueError("image paths must be unique")
            stat = alias.stat()
            entries.append(
                {
                    "image_id": image_id,
                    "label": label,
                    "path": alias.absolute(),
                    "cache_key": (stat.st_dev, stat.st_ino),
                }
            )
            image_ids.add(image_id)
            alias_paths.add(alias_key)

        self.records = entries
        self.cache_mode = cache_mode
        self.instrument = instrument
        self.transform = deep.build_transforms(image_size, train)
        self._resize = self.transform.transforms[0]
        self._after_resize = transforms.Compose(self.transform.transforms[1:])
        self._cache = {}
        self.cache_bytes = 0
        self.cache_build_seconds = 0.0

        if cache_mode != "none":
            began = time.perf_counter()
            representatives = {}
            for entry in entries:
                representatives.setdefault(entry["cache_key"], entry["path"])
            sizes = {}
            total = 0
            for key, path in representatives.items():
                with Image.open(path) as source:
                    width, height = source.size
                size = (
                    image_size * image_size * 3
                    if cache_mode == "resized"
                    else width * height * 3
                )
                sizes[key] = size
                total += size
            if total > cache_max_bytes:
                raise MemoryError(
                    f"cache needs {total} bytes, above limit {cache_max_bytes}"
                )
            for key, path in representatives.items():
                with Image.open(path) as source:
                    image = source.convert("RGB")
                if cache_mode == "resized":
                    image = self._resize(image)
                self._cache[key] = image
            self.cache_bytes = sum(sizes.values())
            self.cache_build_seconds = time.perf_counter() - began

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        began = time.perf_counter()
        if self.cache_mode == "none":
            with Image.open(record["path"]) as source:
                image = source.convert("RGB")
        else:
            image = self._cache[record["cache_key"]].copy()
        decode_seconds = time.perf_counter() - began

        if self.instrument:
            began = time.perf_counter()
            if self.cache_mode == "resized":
                resized = image
                resize_seconds = 0.0
            else:
                resized = self._resize(image)
                resize_seconds = time.perf_counter() - began
            began = time.perf_counter()
            tensor = self._after_resize(resized)
            augment_seconds = time.perf_counter() - began
        elif self.cache_mode == "resized":
            tensor = self._after_resize(image)
        else:
            tensor = self.transform(image)

        item = (
            tensor,
            torch.tensor(record["label"], dtype=torch.float32),
            record["image_id"],
            str(record["path"]),
        )
        if not self.instrument:
            return item
        return (
            *item,
            {
                "decode_seconds": decode_seconds,
                "resize_seconds": resize_seconds,
                "augment_seconds": augment_seconds,
            },
        )
