"""Some metric functions for the model."""

from collections.abc import Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict[str, float | int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, y_prob)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def compute_average_cost(
    y_true: np.ndarray, y_pred: np.ndarray, cost_fn: float, cost_fp: float
) -> float:
    """Average asymmetric cost with zero cost for correct decisions."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    _, fp, fn, _ = cm.ravel()
    return float((cost_fp * fp + cost_fn * fn) / len(y_true))


def add_average_cost(
    metrics: Mapping[str, float | int], n_samples: int, cost_fn: float, cost_fp: float
) -> dict[str, float | int]:
    """Return a metrics copy with the asymmetric average cost attached."""
    enriched = dict(metrics)
    enriched["average_cost"] = float(
        (cost_fp * metrics["fp"] + cost_fn * metrics["fn"]) / n_samples
    )
    return enriched


def compute_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


def compute_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    return precision, recall, pr_auc


def compute_calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    mean_probs = []
    frequencies = []
    bin_sizes = []

    for i in range(len(bins) - 1):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == len(bins) - 2:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])

        if mask.sum() > 0:
            mean_probs.append(y_prob[mask].mean())
            frequencies.append(y_true[mask].mean())
            bin_sizes.append(mask.sum())
        else:
            mean_probs.append(bin_centers[i])
            frequencies.append(np.nan)
            bin_sizes.append(0)

    return np.array(mean_probs), np.array(frequencies), np.array(bin_sizes)


def compute_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    mean_probs, frequencies, bin_sizes = compute_calibration_curve(
        y_true, y_prob, n_bins
    )

    total_samples = len(y_true)
    ece = 0.0

    for size, prob, freq in zip(bin_sizes, mean_probs, frequencies):
        if size > 0 and not np.isnan(freq):
            ece += (size / total_samples) * np.abs(prob - freq)

    return ece


def print_evaluation_summary(
    metrics: Mapping[str, float | int], split_name: str = "Test"
) -> None:
    auc_text = "nan" if np.isnan(metrics["roc_auc"]) else f"{metrics['roc_auc']:.4f}"
    text = f"""
{split_name} results
confusion matrix: TP={metrics["tp"]}, TN={metrics["tn"]}, FP={metrics["fp"]}, FN={metrics["fn"]}
accuracy={metrics["accuracy"]:.4f}, precision={metrics["precision"]:.4f}, recall/sens={metrics["recall"]:.4f}
specificity={metrics["specificity"]:.4f}, f1={metrics["f1"]:.4f}
roc_auc={auc_text}, brier={metrics["brier_score"]:.4f}
"""
    print(text)
