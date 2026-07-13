"""Plots for the report."""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_style():
    """Make plots look okay."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["lines.linewidth"] = 2


def plot_class_distribution(labels, save_path=None, title="Class distribution"):
    names = ["Benign", "Melanoma"]
    counts = [int(np.sum(labels == 0)), int(np.sum(labels == 1))]
    total = sum(counts)
    percents = [count / total * 100 for count in counts]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, counts, color=["#4c78a8", "#f58518"])

    for bar, count, pct in zip(bars, counts, percents):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n{pct:.1f}%",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel("Images")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, axis="y")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_class_pie(labels, save_path=None, title="Class distribution pie"):
    names = ["Benign", "Melanoma"]
    counts = [int(np.sum(labels == 0)), int(np.sum(labels == 1))]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        counts,
        labels=names,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4c78a8", "#f58518"],
    )
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_named_distribution(names, save_path=None, title="Raw class distribution"):
    order = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df", "scc", "other"]
    counts = {name: int(np.sum(names == name)) for name in order}
    counts = {name: count for name, count in counts.items() if count > 0}

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.keys(), counts.values(), color="#4c78a8")

    for bar, count in zip(bars, counts.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )

    ax.set_ylabel("Images")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, axis="y")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_named_pie(names, save_path=None, title="Raw class distribution pie"):
    order = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df", "scc", "other"]
    counts = {name: int(np.sum(names == name)) for name in order}
    counts = {name: count for name, count in counts.items() if count > 0}

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(
        list(counts.values()),
        labels=list(counts.keys()),
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    save_path: str | None = None,
    title: str = "ROC Curve",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, "b-", lw=2.5, label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    pr_auc: float,
    baseline: float | None = None,
    save_path: str | None = None,
    title: str = "Precision-Recall Curve",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, "r-", lw=2.5, label=f"PR Curve (AUC = {pr_auc:.4f})")

    if baseline is not None:
        ax.axhline(
            y=baseline,
            color="k",
            linestyle="--",
            lw=1.5,
            label=f"No Skill Classifier ({baseline:.3f})",
        )

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray, save_path: str | None = None, title: str = "Confusion Matrix"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False,
        xticklabels=["Benign", "Melanoma"],
        yticklabels=["Benign", "Melanoma"],
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_calibration_curve(
    mean_probs: np.ndarray,
    frequencies: np.ndarray,
    bin_sizes: np.ndarray,
    save_path: str | None = None,
    title: str = "Reliability Diagram",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Perfect Calibration")

    valid = bin_sizes > 0
    mean_probs_valid = mean_probs[valid]
    frequencies_valid = frequencies[valid]
    bin_sizes_valid = bin_sizes[valid]

    scatter = ax.scatter(
        mean_probs_valid,
        frequencies_valid,
        s=bin_sizes_valid * 10,
        alpha=0.6,
        edgecolors="black",
        lw=1.5,
        c=mean_probs_valid,
        cmap="viridis",
    )

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Empirical Frequency (True Positive Rate)")
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Predicted Probability", fontsize=10)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_bic_scores(
    component_range: Sequence[int],
    bic_scores: Sequence[float],
    optimal: int,
    save_path: str | None = None,
    title: str = "BIC Scores for GMM Component Selection",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(component_range, bic_scores, "bo-", lw=2, markersize=8, label="BIC Score")
    ax.axvline(x=optimal, color="r", linestyle="--", lw=2, label=f"Optimal K={optimal}")
    ax.scatter([optimal], [bic_scores[optimal - 1]], color="r", s=150, zorder=5)

    ax.set_xlabel("Number of Components")
    ax.set_ylabel("BIC Score")
    ax.set_title(title)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_and_pr_comparison(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    precision: np.ndarray,
    recall: np.ndarray,
    pr_auc: float,
    save_path: str | None = None,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(fpr, tpr, "b-", lw=2.5, label=f"ROC (AUC={auc_score:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(recall, precision, "r-", lw=2.5, label=f"PR (AUC={pr_auc:.4f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_metric_comparison(
    metrics_dict: Mapping[str, Mapping[str, float | int]], save_path: str | None = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = ["accuracy", "precision", "recall", "specificity", "f1"]
    split_names = list(metrics_dict.keys())

    x = np.arange(len(metric_names))
    width = 0.25

    for i, split_name in enumerate(split_names):
        values = [metrics_dict[split_name].get(m, 0) for m in metric_names]
        ax.bar(x + i * width, values, width, label=split_name)

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.set_title("Metric Comparison Across Splits")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_threshold_comparison(
    metrics_by_point: Mapping[str, Mapping[str, float | int]],
    save_path: str | None = None,
    title: str = "Threshold Comparison on Test Set",
) -> plt.Figure:
    """Compare MAP and cost-sensitive threshold operating points."""
    metric_names = ["recall", "specificity", "precision", "f1", "average_cost"]
    point_names = list(metrics_by_point.keys())
    max_value = max(
        float(metrics_by_point[point_name][metric])
        for point_name in point_names
        for metric in metric_names
    )

    x = np.arange(len(metric_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, point_name in enumerate(point_names):
        offset = (i - (len(point_names) - 1) / 2) * width
        values = [float(metrics_by_point[point_name][metric]) for metric in metric_names]
        bars = ax.bar(x + offset, values, width, label=point_name)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score / average cost")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(["Recall", "Specificity", "Precision", "F1", "Avg. cost"])
    ax.legend()
    ax.set_ylim([0, min(1.5, max(1.0, max_value) * 1.18)])
    ax.grid(True, alpha=0.3, axis="y")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
