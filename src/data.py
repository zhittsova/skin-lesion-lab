"""Loading the csv and images."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_metadata(metadata_path: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    return df


def create_binary_labels(df: pd.DataFrame, target_disease: str = "mel") -> np.ndarray:
    """1 means melanoma, 0 means not melanoma."""
    # Different HAM10000 csv files name this a bit differently.
    labels = np.zeros(len(df), dtype=int)

    if target_disease == "mel":
        melanoma_mask = pd.Series(False, index=df.index)

        for column in ["dx", "diagnosis", "diagnosis_1", "diagnosis_2", "diagnosis_3"]:
            if column not in df.columns:
                continue

            values = df[column].astype(str).str.strip()
            melanoma_mask = melanoma_mask | values.str.lower().eq("mel")
            melanoma_mask = melanoma_mask | values.str.contains(
                "melanoma", case=False, na=False
            )

        labels[melanoma_mask] = 1

    return labels


def create_binary_task_mask(df: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Keep melanoma and benign rows, drop the other cancer classes."""
    melanoma_mask = pd.Series(labels == 1, index=df.index)
    benign_mask = pd.Series(False, index=df.index)

    for column in ["diagnosis", "diagnosis_1", "diagnosis_2", "diagnosis_3"]:
        if column in df.columns:
            benign_mask = benign_mask | df[column].astype(str).str.contains(
                "benign", case=False, na=False
            )

    if "dx" in df.columns:
        benign_dx_codes = {"nv", "bkl", "df", "vasc"}
        benign_mask = benign_mask | df["dx"].astype(str).str.lower().isin(
            benign_dx_codes
        )

    if not benign_mask.any():
        benign_mask = pd.Series(labels == 0, index=df.index)

    return (melanoma_mask | benign_mask).to_numpy()


def get_multiclass_codes(df: pd.DataFrame) -> np.ndarray:
    codes = []

    for _, row in df.iterrows():
        if "dx" in df.columns and pd.notna(row.get("dx")):
            codes.append(str(row["dx"]).lower())
            continue

        text = " ".join(
            str(row.get(col, ""))
            for col in ["diagnosis_1", "diagnosis_2", "diagnosis_3"]
        ).lower()

        if "melanoma" in text:
            codes.append("mel")
        elif "nevus" in text:
            codes.append("nv")
        elif "keratosis" in text and "actinic" not in text and "solar" not in text:
            codes.append("bkl")
        elif "basal cell" in text:
            codes.append("bcc")
        elif "actinic" in text or "solar" in text:
            codes.append("akiec")
        elif "vascular" in text:
            codes.append("vasc")
        elif "dermatofibroma" in text or "fibro-histiocytic" in text:
            codes.append("df")
        elif "squamous cell" in text:
            codes.append("scc")
        else:
            codes.append("other")

    return np.array(codes)


def filter_valid_images(df: pd.DataFrame, images_dir: Path) -> pd.DataFrame:
    """Only keep rows where the jpg is actually there."""
    valid_indices = []
    for idx, row in df.iterrows():
        img_path = images_dir / f"{row['isic_id']}.jpg"
        if img_path.exists():
            valid_indices.append(idx)

    return df.loc[valid_indices].reset_index(drop=True)


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
    metadata_path: str, images_dir: str
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = load_metadata(metadata_path)
    images_path = Path(images_dir)

    df = filter_valid_images(df, images_path)

    labels_all = create_binary_labels(df)
    binary_task_mask = create_binary_task_mask(df, labels_all)
    df = df.loc[binary_task_mask].reset_index(drop=True)
    labels = labels_all[binary_task_mask]

    image_ids = get_image_ids(df)

    return df, image_ids, labels


def get_class_statistics(labels: np.ndarray) -> dict[str, int]:
    counts = dict(zip(*np.unique(labels, return_counts=True)))
    return {
        "benign": int(counts.get(0, 0)),
        "melanoma": int(counts.get(1, 0)),
    }
