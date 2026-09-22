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

from src import calibration


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, *, metrics_version=2
) -> dict[str, float | int | None]:
    if type(metrics_version) is not int or metrics_version not in (1, 2):
        raise ValueError("unsupported metrics version")
    y_prob = calibration.probabilities(y_prob)
    y_true = calibration.targets(y_true, len(y_prob))
    y_pred = calibration.targets(y_pred, len(y_prob))
    if metrics_version == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        def ratio(numerator, denominator):
            return float(numerator / denominator) if denominator else None

        return {
            "accuracy": float((tp + tn) / len(y_true)),
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "sensitivity": ratio(tp, tp + fn),
            "specificity": ratio(tn, tn + fp),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "roc_auc": float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2
            else None,
            "brier_score": float(np.mean((y_prob - y_true) ** 2)),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }
    # Bounded historical summary replay; never the default for a new report.
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
    y_true = calibration.targets(y_true, len(y_true))
    y_pred = calibration.targets(y_pred, len(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    _, fp, fn, _ = cm.ravel()
    return add_average_cost(
        {"fp": int(fp), "fn": int(fn)}, len(y_true), cost_fn, cost_fp
    )["average_cost"]


def add_average_cost(
    metrics: Mapping[str, float | int], n_samples: int, cost_fn: float, cost_fp: float
) -> dict[str, float | int]:
    """Return a metrics copy with the asymmetric average cost attached."""
    calibration.cost_threshold(cost_fn, cost_fp)
    if n_samples <= 0:
        raise ValueError("metrics require nonempty inputs")
    enriched = dict(metrics)
    enriched["average_cost"] = float(
        cost_fp * (metrics["fp"] / n_samples) + cost_fn * (metrics["fn"] / n_samples)
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
    rows = calibration.probability_report(y_true, y_prob, n_bins)["reliability_bins"]
    return (
        np.array([row["mean_probability"] if row["count"] else np.nan for row in rows]),
        np.array(
            [row["positive_fraction"] if row["count"] else np.nan for row in rows]
        ),
        np.array([row["count"] for row in rows]),
    )


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
    def formatted(key):
        return format_metric(metrics[key])

    print(f"""
{split_name} results
confusion matrix: TP={metrics["tp"]}, TN={metrics["tn"]}, FP={metrics["fp"]}, FN={metrics["fn"]}
accuracy={formatted("accuracy")}, precision={formatted("precision")}, recall/sens={formatted("recall")}
specificity={formatted("specificity")}, f1={formatted("f1")}
roc_auc={formatted("roc_auc")}, brier={formatted("brier_score")}
""")


def format_metric(value, digits=4):
    return (
        "undefined"
        if value is None or not np.isfinite(value)
        else f"{value:.{digits}f}"
    )
