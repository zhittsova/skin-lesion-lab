"""Split train/val/test by lesion id."""

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_lesion_split(
    lesion_ids: np.ndarray,
    labels: np.ndarray,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split lesions, then get image indexes back."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Sizes must sum to 1.0")

    classes = np.unique(labels)
    if len(classes) != 2 or not set(classes).issubset({0, 1}):
        raise ValueError(
            "Expected binary labels with both classes present before stratified splitting"
        )

    # if a lesion has any melanoma image, call the lesion melanoma
    unique_lesions, inverse = np.unique(lesion_ids, return_inverse=True)
    lesion_labels = np.zeros(len(unique_lesions), dtype=int)
    np.maximum.at(lesion_labels, inverse, labels)

    train_lesions, temp_lesions, train_labels, temp_labels = train_test_split(
        unique_lesions,
        lesion_labels,
        train_size=train_size,
        test_size=1 - train_size,
        stratify=lesion_labels,
        random_state=random_state,
    )

    val_ratio = val_size / (val_size + test_size)
    val_lesions, test_lesions, val_labels, test_labels = train_test_split(
        temp_lesions,
        temp_labels,
        train_size=val_ratio,
        test_size=1 - val_ratio,
        stratify=temp_labels,
        random_state=random_state + 1,
    )

    train_indices = np.where(np.isin(lesion_ids, train_lesions))[0]
    val_indices = np.where(np.isin(lesion_ids, val_lesions))[0]
    test_indices = np.where(np.isin(lesion_ids, test_lesions))[0]

    return train_indices, val_indices, test_indices


def split_dataset(
    lesion_ids: np.ndarray,
    labels: np.ndarray,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    train_idx, val_idx, test_idx = stratified_lesion_split(
        lesion_ids,
        labels,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def get_split_statistics(
    labels: np.ndarray, split_indices: dict[str, np.ndarray]
) -> dict[str, dict[str, int | float]]:
    stats = {}
    for split_name, indices in split_indices.items():
        split_labels = labels[indices]
        counts = dict(zip(*np.unique(split_labels, return_counts=True)))

        benign_count = counts.get(0, 0)
        melanoma_count = counts.get(1, 0)

        stats[split_name] = {
            "benign": int(benign_count),
            "melanoma": int(melanoma_count),
            "total": int(len(split_labels)),
            "melanoma_ratio": float(melanoma_count) / len(split_labels)
            if len(split_labels) > 0
            else 0,
        }

    return stats


def get_split_report(
    labels: np.ndarray, split_indices: dict[str, np.ndarray], lesion_ids: np.ndarray
) -> dict[str, object]:
    """Return split statistics plus a lesion-level leakage check."""
    stats = get_split_statistics(labels, split_indices)

    for split_name, indices in split_indices.items():
        stats[split_name]["unique_lesions"] = int(len(np.unique(lesion_ids[indices])))

    train_lesions = set(lesion_ids[split_indices["train"]])
    val_lesions = set(lesion_ids[split_indices["val"]])
    test_lesions = set(lesion_ids[split_indices["test"]])

    overlaps = {
        "train_val": len(train_lesions & val_lesions),
        "train_test": len(train_lesions & test_lesions),
        "val_test": len(val_lesions & test_lesions),
    }

    return {
        "splits": stats,
        "lesion_overlap_counts": overlaps,
        "leakage_free": all(count == 0 for count in overlaps.values()),
    }


def print_split_summary(
    labels: np.ndarray, split_indices: dict[str, np.ndarray], lesion_ids: np.ndarray
) -> None:
    report = get_split_report(labels, split_indices, lesion_ids)
    stats = report["splits"]
    lines = ["", "split summary"]

    for split_name in ["train", "val", "test"]:
        s = stats[split_name]
        lines.append(
            f"{split_name}: {s['total']} images, "
            f"{s['benign']} benign, {s['melanoma']} melanoma "
            f"({s['melanoma_ratio'] * 100:.1f}% mel)"
        )
        lines.append(f"  lesion ids: {s['unique_lesions']}")

    lines.append("")
    lines.append("leak check")
    overlaps = report["lesion_overlap_counts"]
    lines.append(f"train/val overlap: {overlaps['train_val']}")
    lines.append(f"train/test overlap: {overlaps['train_test']}")
    lines.append(f"val/test overlap: {overlaps['val_test']}")

    if report["leakage_free"]:
        lines.append("no lesion leakage found")
    else:
        lines.append("oops, lesion leakage found")

    print("\n".join(lines))
