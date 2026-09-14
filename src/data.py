"""Load validated lesion cohorts and their local images."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.cohort import EmptyCohortError, build_cohort


def load_metadata(metadata_path: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    return df


def load_image_rgb(
    img_path: str, target_size: tuple[int, int] | None = None
) -> np.ndarray:
    """OpenCV reads BGR, so change it to RGB."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if target_size is not None:
        img = cv2.resize(
            img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR
        )

    return img


def load_image_hsv(
    img_path: str, target_size: tuple[int, int] | None = None
) -> np.ndarray:
    """Load one image and convert to HSV."""
    img_rgb = load_image_rgb(img_path, target_size)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return img_hsv


def get_image_ids(df: pd.DataFrame) -> np.ndarray:
    return df["isic_id"].values


def get_lesion_ids(df: pd.DataFrame) -> np.ndarray:
    return df["lesion_id"].values


def prepare_dataset(
    metadata_path: str,
    images_dir: str,
    *,
    source: str,
    attrition_path: str | Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    audit_path = Path(attrition_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cohort = build_cohort(metadata_path, images_dir, source=source)
        counts, outcomes = cohort.counts, cohort.outcomes
    except EmptyCohortError as exc:
        counts, outcomes = exc.counts, exc.outcomes
        cohort = None
    audit_path.write_text(
        json.dumps(
            {
                "source": source,
                "metadata_sha256": hashlib.sha256(
                    Path(metadata_path).read_bytes()
                ).hexdigest(),
                "counts": counts,
                "outcomes": outcomes,
            },
            indent=2,
        )
        + "\n"
    )
    if cohort is None:
        raise ValueError("empty cohort")
    df = cohort.frame
    return df, get_image_ids(df), df["target"].to_numpy()


def get_class_statistics(labels: np.ndarray) -> dict[str, int]:
    counts = dict(zip(*np.unique(labels, return_counts=True)))
    return {
        "benign": int(counts.get(0, 0)),
        "melanoma": int(counts.get(1, 0)),
    }
