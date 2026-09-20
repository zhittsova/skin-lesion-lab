"""Bounded in-memory cache for resized training images."""

import time
from collections.abc import Sequence
from pathlib import Path

from PIL import Image
from torchvision import transforms


class ResizedImageCache:
    """Decode and resize a fixed image cohort once for the current process.

    ``cache_bytes`` measures the RGB pixel payload. Python and Pillow object
    overhead is outside that figure, so the limit is a payload bound rather
    than an RSS limit.
    """

    def __init__(
        self,
        image_ids: Sequence[str],
        images_dir: Path,
        image_size: int,
        max_bytes: int,
    ) -> None:
        if isinstance(image_ids, (str, bytes)):
            raise ValueError("image_ids must be a nonempty sequence")
        try:
            cohort = list(image_ids)
        except TypeError as error:
            raise ValueError("image_ids must be a nonempty sequence") from error
        if not cohort:
            raise ValueError("image_ids must be a nonempty sequence")
        if any(
            not isinstance(image_id, str)
            or not image_id
            or Path(image_id).name != image_id
            or len(Path(image_id).parts) != 1
            for image_id in cohort
        ):
            raise ValueError("image IDs must be nonempty path-free strings")
        if len(set(cohort)) != len(cohort):
            raise ValueError("image IDs must be unique")
        if isinstance(image_size, bool) or not isinstance(image_size, int):
            raise ValueError("image_size must be an integer >= 32")
        if image_size < 32:
            raise ValueError("image_size must be an integer >= 32")
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 1
        ):
            raise ValueError("max_bytes must be a positive integer")

        supplied_root = Path(images_dir)
        try:
            root = supplied_root.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as error:
            raise ValueError("images_dir must exist") from error
        if not root.is_dir():
            raise ValueError("images_dir must be a directory")

        cache_bytes = len(cohort) * image_size * image_size * 3
        if cache_bytes > max_bytes:
            raise MemoryError(
                f"cache needs {cache_bytes} bytes, above limit {max_bytes}"
            )

        paths: dict[str, Path] = {}
        for image_id in cohort:
            alias = supplied_root / f"{image_id}.jpg"
            try:
                resolved = alias.resolve(strict=True)
            except (FileNotFoundError, RuntimeError) as error:
                raise ValueError(f"image does not exist: {image_id}.jpg") from error
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise ValueError("image path escapes images_dir or is not a file")
            paths[image_id] = resolved

        self.image_ids = tuple(cohort)
        self.images_dir = root
        self.image_size = image_size
        self.max_bytes = max_bytes
        self.cache_bytes = cache_bytes
        self._images: dict[str, Image.Image] = {}

        resize = transforms.Resize((image_size, image_size))
        began = time.perf_counter()
        for image_id, path in paths.items():
            with Image.open(path) as source:
                image = source.convert("RGB")
            self._images[image_id] = resize(image)
        self.cache_build_seconds = time.perf_counter() - began

    @property
    def image_count(self) -> int:
        """Return the number of cohort entries held in memory."""
        return len(self._images)

    def __len__(self) -> int:
        return self.image_count

    def prepare_transform(
        self,
        image_ids: Sequence[str],
        images_dir: Path,
        transform: transforms.Compose | None,
    ) -> transforms.Compose:
        """Validate a dataset contract and return its transforms after resize."""
        try:
            dataset_root = Path(images_dir).resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as error:
            raise ValueError("dataset images_dir must exist") from error
        if dataset_root != self.images_dir:
            raise ValueError("resized cache and dataset must use the same images_dir")

        missing = set(image_ids).difference(self._images)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"resized cache is missing image IDs: {names}")

        if not isinstance(transform, transforms.Compose) or not transform.transforms:
            raise ValueError("cached datasets require a Compose starting with Resize")
        leading = transform.transforms[0]
        expected = transforms.Resize((self.image_size, self.image_size))
        if type(leading) is not transforms.Resize or any(
            getattr(leading, field, None) != getattr(expected, field, None)
            for field in ("size", "interpolation", "max_size", "antialias")
        ):
            raise ValueError(
                "cached dataset transform must start with the matching Resize"
            )
        return transforms.Compose(transform.transforms[1:])

    def get(self, image_id: str) -> Image.Image:
        """Return a copy so per-access transforms cannot change cached pixels."""
        try:
            return self._images[image_id].copy()
        except KeyError as error:
            raise KeyError(f"image ID is not cached: {image_id}") from error
