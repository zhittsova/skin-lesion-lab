"""Train a CNN baseline and estimate uncertainty with MC dropout."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import roc_curve
from src import data, deep, evaluation, plots, reporting, splitting, uncertainty
from torch import nn
from torch.utils.data import DataLoader

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
BLUE = {"base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"}
ORANGE = {"base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"}
PINK = {"base": "#F390CA", "dark": "#8A3A6F"}


def parse_args() -> argparse.Namespace:
    project_path = Path(__file__).parent
    dataset_path = project_path / "data" / "raw"

    parser = argparse.ArgumentParser(
        description="Train a project-owned deep skin-lesion baseline with MC Dropout."
    )
    parser.add_argument(
        "--metadata-path", type=Path, default=dataset_path / "metadata.csv"
    )
    parser.add_argument(
        "--source", choices=["ham10000", "isic2018_task3"], required=True
    )
    parser.add_argument("--images-dir", type=Path, default=dataset_path)
    parser.add_argument("--results-dir", type=Path, default=project_path / "results")
    parser.add_argument("--runs-dir", type=Path, default=project_path / "runs")
    parser.add_argument("--models-dir", type=Path, default=project_path / "models")
    parser.add_argument(
        "--architecture",
        choices=["small_cnn", "efficientnet_b0"],
        default="small_cnn",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use torchvision pretrained weights for EfficientNet-B0.",
    )
    parser.add_argument(
        "--fine-tune-backbone",
        action="store_true",
        help="Train all EfficientNet-B0 weights instead of only its classifier head.",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cost-fn", type=float, default=10.0)
    parser.add_argument("--cost-fp", type=float, default=1.0)
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        help="Optional stratified split limit for smoke runs.",
    )
    parser.add_argument(
        "--max-val-images",
        type=int,
        default=None,
        help="Optional stratified split limit for smoke runs.",
    )
    parser.add_argument(
        "--max-test-images",
        type=int,
        default=None,
        help="Optional stratified split limit for smoke runs.",
    )
    return parser.parse_args()


def limit_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    max_images: int | None,
    seed: int,
) -> np.ndarray:
    if max_images is None or max_images >= len(indices):
        return indices

    rng = np.random.default_rng(seed)
    selected_parts = []
    remaining = max_images

    for class_label in [0, 1]:
        class_indices = indices[labels[indices] == class_label]
        if len(class_indices) == 0:
            continue
        n_class = max(1, round(max_images * len(class_indices) / len(indices)))
        n_class = min(n_class, len(class_indices), remaining)
        remaining -= n_class
        selected_parts.append(rng.choice(class_indices, size=n_class, replace=False))

    if remaining > 0:
        already_selected = (
            np.concatenate(selected_parts)
            if selected_parts
            else np.array([], dtype=int)
        )
        pool = np.setdiff1d(indices, already_selected, assume_unique=False)
        if len(pool) > 0:
            selected_parts.append(
                rng.choice(pool, size=min(remaining, len(pool)), replace=False)
            )

    return rng.permutation(np.concatenate(selected_parts))


def make_loader(
    image_ids: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    images_dir: Path,
    image_size: int,
    batch_size: int,
    train: bool,
    num_workers: int,
) -> DataLoader:
    dataset = deep.SkinLesionImageDataset(
        image_ids=image_ids[indices],
        labels=labels[indices],
        images_dir=images_dir,
        transform=deep.build_transforms(image_size=image_size, train=train),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def metrics_for_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_fn: float,
    cost_fp: float,
) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = evaluation.compute_classification_metrics(y_true, y_pred, y_prob)
    metrics = evaluation.add_average_cost(metrics, len(y_true), cost_fn, cost_fp)
    metrics["threshold"] = float(threshold)
    return metrics


def find_best_threshold_by_validation_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float,
    cost_fp: float,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(0.0, 1.0, 501):
        metrics = metrics_for_threshold(y_true, y_prob, threshold, cost_fn, cost_fp)
        rows.append(metrics)

    frame = pd.DataFrame(rows)
    best = frame.sort_values(
        ["average_cost", "recall", "specificity"],
        ascending=[True, False, False],
    ).iloc[0]
    return float(best["threshold"]), frame


def save_deep_metrics_tables(
    metrics_by_split: dict[str, dict[str, dict[str, float | int]]],
    ece_by_split: dict[str, float],
    tables_dir: Path,
    architecture: str,
) -> pd.DataFrame:
    metrics_df = pd.DataFrame(
        reporting.build_metrics_rows(metrics_by_split, ece_by_split)
    )
    metrics_df.to_csv(tables_dir / f"{architecture}_metrics_summary.csv", index=False)

    threshold_columns = [
        "split",
        "operating_point",
        "threshold",
        "recall",
        "specificity",
        "precision",
        "f1",
        "fp",
        "fn",
        "tp",
        "tn",
        "average_cost",
    ]
    metrics_df[threshold_columns].to_csv(
        tables_dir / f"{architecture}_threshold_comparison.csv", index=False
    )
    return metrics_df


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
        },
    )


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    ax.set_title("")
    left = ax.get_position().x0
    fig.text(
        left,
        0.985,
        title,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.94,
        subtitle,
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(top=0.83)
    sns.despine(ax=ax)


def build_uncertainty_frame(mc_result: dict, threshold: float) -> pd.DataFrame:
    summary = uncertainty.summarize_mc_dropout_probabilities(
        mc_result["all_probabilities"]
    )
    frame = pd.DataFrame(
        {
            "image_id": mc_result["image_id"],
            "target": mc_result["label"],
            "target_name": np.where(mc_result["label"] == 1, "Melanoma", "Benign"),
            "mean_prob_melanoma": summary["mean_prob_melanoma"],
            "predictive_std": np.sqrt(summary["variance"]),
            "predictive_entropy": summary["predictive_entropy"],
            "mutual_information": summary["mutual_information"],
        }
    )
    frame["prediction"] = (frame["mean_prob_melanoma"] >= threshold).astype(int)
    frame["near_threshold"] = np.abs(frame["mean_prob_melanoma"] - threshold) <= 0.05
    return frame


def plot_uncertainty_distribution(frame: pd.DataFrame, output_path: Path) -> None:
    use_chart_theme()
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=180)
    sns.histplot(
        data=frame,
        x="predictive_std",
        bins=32,
        stat="count",
        color=BLUE["base"],
        edgecolor=BLUE["dark"],
        linewidth=1.0,
        ax=ax,
    )
    median = frame["predictive_std"].median()
    p90 = frame["predictive_std"].quantile(0.9)
    ax.axvline(median, color=TOKENS["ink"], linestyle=":", linewidth=1.2)
    ax.axvline(p90, color=ORANGE["mid"], linestyle="--", linewidth=1.3)
    ax.set_xlabel("Predictive uncertainty: std of MC P(melanoma)")
    ax.set_ylabel("Test images")
    add_chart_header(
        fig,
        ax,
        "Distribution of predictive uncertainties (MC Dropout)",
        f"Real test-set inference; median std={median:.3f}, 90th percentile={p90:.3f}.",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_mean_vs_epistemic_uncertainty(
    frame: pd.DataFrame,
    output_path: Path,
    threshold: float,
) -> None:
    use_chart_theme()
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=180)
    palette = {"Benign": BLUE["mid"], "Melanoma": PINK["dark"]}
    sns.scatterplot(
        data=frame,
        x="mean_prob_melanoma",
        y="mutual_information",
        hue="target_name",
        palette=palette,
        edgecolor=TOKENS["panel"],
        linewidth=0.25,
        alpha=0.72,
        s=26,
        ax=ax,
    )
    ax.axvline(
        threshold,
        color=TOKENS["ink"],
        linestyle="--",
        linewidth=1.2,
        label=f"decision threshold={threshold:.3f}",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("MC predictive mean P(melanoma)")
    ax.set_ylabel("Mutual information (epistemic uncertainty)")
    ax.legend(frameon=False, loc="upper right")
    add_chart_header(
        fig,
        ax,
        "MC Dropout epistemic uncertainty on the test set",
        "Each point is one image; uncertainty is higher when dropout passes disagree.",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_predictions(
    mc_result: dict,
    threshold: float,
    output_path: Path,
) -> pd.DataFrame:
    frame = build_uncertainty_frame(mc_result, threshold=threshold)
    frame["image_path"] = mc_result["image_path"]
    frame["score"] = np.where(
        frame["prediction"] == 1,
        frame["mean_prob_melanoma"],
        1 - frame["mean_prob_melanoma"],
    )
    frame["model_name"] = "Project_Deep_CNN_MC_Dropout"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def main() -> None:
    args = parse_args()
    start_time = time.time()
    deep.set_seed(args.seed)

    figures_dir = args.results_dir / "figures"
    tables_dir = args.results_dir / "tables"
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("Project-owned deep MC Dropout pipeline")
    print("1. loading data")
    df, image_ids, labels = data.prepare_dataset(
        str(args.metadata_path),
        str(args.images_dir),
        source=args.source,
        attrition_path=args.runs_dir / "cohort_attrition.json",
    )
    lesion_ids = data.get_lesion_ids(df)

    split_indices = splitting.split_dataset(
        lesion_ids,
        labels,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        random_state=args.seed,
    )
    split_indices["train"] = limit_indices(
        split_indices["train"], labels, args.max_train_images, args.seed
    )
    split_indices["val"] = limit_indices(
        split_indices["val"], labels, args.max_val_images, args.seed + 1
    )
    split_indices["test"] = limit_indices(
        split_indices["test"], labels, args.max_test_images, args.seed + 2
    )
    split_report = splitting.get_split_report(labels, split_indices, lesion_ids)
    splitting.print_split_summary(labels, split_indices, lesion_ids)

    print("\n2. building dataloaders")
    train_loader = make_loader(
        image_ids,
        labels,
        split_indices["train"],
        args.images_dir,
        args.image_size,
        args.batch_size,
        train=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        image_ids,
        labels,
        split_indices["val"],
        args.images_dir,
        args.image_size,
        args.batch_size,
        train=False,
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        image_ids,
        labels,
        split_indices["test"],
        args.images_dir,
        args.image_size,
        args.batch_size,
        train=False,
        num_workers=args.num_workers,
    )

    print("\n3. training")
    device = deep.get_default_device()
    print(f"device: {device}")
    model = deep.build_model(
        architecture=args.architecture,
        dropout=args.dropout,
        pretrained=args.pretrained,
        freeze_backbone=not args.fine_tune_backbone,
    ).to(device)

    y_train = labels[split_indices["train"]]
    pos_weight = deep.compute_pos_weight(y_train)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    best_score = -float("inf")
    best_checkpoint = args.models_dir / f"{args.architecture}_mc_dropout.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = deep.train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss = deep.evaluate_loss(model, val_loader, criterion, device)
        val_pred = deep.predict_probabilities(model, val_loader, device)
        val_metrics = metrics_for_threshold(
            val_pred["label"],
            val_pred["probability"],
            threshold=0.5,
            cost_fn=args.cost_fn,
            cost_fp=args.cost_fp,
        )
        score = float(val_metrics["roc_auc"])
        if np.isnan(score):
            score = -val_loss

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_metrics["roc_auc"],
            "val_recall_at_0_5": val_metrics["recall"],
            "val_specificity_at_0_5": val_metrics["specificity"],
        }
        history.append(history_row)
        print(
            f"epoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_auc={val_metrics['roc_auc']:.4f}"
        )

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": args.architecture,
                    "image_size": args.image_size,
                    "dropout": args.dropout,
                    "pretrained": args.pretrained,
                    "fine_tune_backbone": args.fine_tune_backbone,
                    "epoch": epoch,
                    "val_auc": val_metrics["roc_auc"],
                },
                best_checkpoint,
            )

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\n4. MC Dropout inference")
    val_mc = deep.predict_with_mc_dropout(
        model=model,
        dataloader=val_loader,
        device=device,
        n_passes=args.mc_samples,
    )
    test_mc = deep.predict_with_mc_dropout(
        model=model,
        dataloader=test_loader,
        device=device,
        n_passes=args.mc_samples,
    )

    np.save(
        args.runs_dir / f"{args.architecture}_val_mc_probabilities.npy",
        val_mc["all_probabilities"],
    )
    np.save(
        args.runs_dir / f"{args.architecture}_test_mc_probabilities.npy",
        test_mc["all_probabilities"],
    )
    np.save(
        args.runs_dir / f"{args.architecture}_test_probability.npy",
        test_mc["mean_probability"],
    )
    np.save(
        args.runs_dir / f"{args.architecture}_test_uncertainty.npy",
        test_mc["uncertainty"],
    )
    np.save(args.runs_dir / f"{args.architecture}_test_y_true.npy", test_mc["label"])

    print("\n5. threshold selection and evaluation")
    formula_threshold = args.cost_fp / (args.cost_fp + args.cost_fn)
    selected_threshold, threshold_sweep = find_best_threshold_by_validation_cost(
        val_mc["label"],
        val_mc["mean_probability"],
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
    )
    threshold_sweep.to_csv(
        tables_dir / f"{args.architecture}_validation_threshold_sweep.csv",
        index=False,
    )

    metrics_by_split = {}
    for split_name, mc_result in {"val": val_mc, "test": test_mc}.items():
        y_true = mc_result["label"]
        y_prob = mc_result["mean_probability"]
        metrics_by_split[split_name] = {
            "cost_formula_threshold": metrics_for_threshold(
                y_true, y_prob, formula_threshold, args.cost_fn, args.cost_fp
            ),
            "validation_cost_threshold": metrics_for_threshold(
                y_true, y_prob, selected_threshold, args.cost_fn, args.cost_fp
            ),
            "map_threshold": metrics_for_threshold(
                y_true, y_prob, 0.5, args.cost_fn, args.cost_fp
            ),
        }

    ece_by_split = {
        "val": evaluation.compute_calibration_error(
            val_mc["label"], val_mc["mean_probability"]
        ),
        "test": evaluation.compute_calibration_error(
            test_mc["label"], test_mc["mean_probability"]
        ),
    }
    save_deep_metrics_tables(
        metrics_by_split, ece_by_split, tables_dir, args.architecture
    )

    print("\n6. saving predictions and plots")
    test_predictions = save_predictions(
        test_mc,
        threshold=selected_threshold,
        output_path=tables_dir / f"{args.architecture}_predictions_test.csv",
    )
    val_predictions = save_predictions(
        val_mc,
        threshold=selected_threshold,
        output_path=tables_dir / f"{args.architecture}_predictions_val.csv",
    )

    fpr, tpr, auc_score = evaluation.compute_roc_curve(
        test_mc["label"], test_mc["mean_probability"]
    )
    plots.plot_roc_curve(
        fpr,
        tpr,
        auc_score,
        save_path=str(figures_dir / f"{args.architecture}_roc_curve.png"),
        title=f"{args.architecture} ROC curve",
    )
    cm_test = evaluation.compute_confusion_matrix(
        test_mc["label"],
        (test_mc["mean_probability"] >= selected_threshold).astype(int),
    )
    plots.plot_confusion_matrix(
        cm_test,
        save_path=str(figures_dir / f"{args.architecture}_confusion_matrix.png"),
        title=f"{args.architecture} confusion matrix",
    )

    plot_uncertainty_distribution(
        test_predictions,
        figures_dir / f"{args.architecture}_mc_dropout_uncertainty_distribution.png",
    )
    plot_mean_vs_epistemic_uncertainty(
        test_predictions,
        figures_dir
        / f"{args.architecture}_mc_dropout_mean_vs_epistemic_uncertainty.png",
        threshold=formula_threshold,
    )

    fpr_table, tpr_table, roc_thresholds = roc_curve(
        test_mc["label"], test_mc["mean_probability"]
    )
    pd.DataFrame(
        {
            "fpr": fpr_table,
            "tpr": tpr_table,
            "threshold": roc_thresholds,
        }
    ).to_csv(tables_dir / f"{args.architecture}_roc_curve_points.csv", index=False)

    summary = {
        "pipeline": "Project-owned deep CNN with MC Dropout",
        "architecture": args.architecture,
        "task": "binary melanoma-vs-benign classification",
        "positive_class": "melanoma",
        "pretrained": args.pretrained,
        "fine_tune_backbone": args.fine_tune_backbone,
        "image_size": args.image_size,
        "dropout": args.dropout,
        "mc_samples": args.mc_samples,
        "split": split_report,
        "cost_matrix": {
            "false_negative": args.cost_fn,
            "false_positive": args.cost_fp,
            "cost_formula_threshold": formula_threshold,
            "selected_threshold_from_validation": selected_threshold,
            "map_threshold": 0.5,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "pos_weight": pos_weight,
            "history": history,
            "best_checkpoint": str(best_checkpoint),
        },
        "metrics": metrics_by_split,
        "expected_calibration_error": ece_by_split,
        "mc_dropout_uncertainty": {
            "test_mean_predictive_std": float(
                test_predictions["predictive_std"].mean()
            ),
            "test_median_predictive_std": float(
                test_predictions["predictive_std"].median()
            ),
            "test_p90_predictive_std": float(
                test_predictions["predictive_std"].quantile(0.9)
            ),
            "test_mean_mutual_information": float(
                test_predictions["mutual_information"].mean()
            ),
        },
        "artifacts": {
            "test_mc_probabilities": str(
                args.runs_dir / f"{args.architecture}_test_mc_probabilities.npy"
            ),
            "test_predictions": str(
                tables_dir / f"{args.architecture}_predictions_test.csv"
            ),
            "uncertainty_distribution": str(
                figures_dir
                / f"{args.architecture}_mc_dropout_uncertainty_distribution.png"
            ),
            "mean_vs_epistemic_uncertainty": str(
                figures_dir
                / f"{args.architecture}_mc_dropout_mean_vs_epistemic_uncertainty.png"
            ),
        },
        "runtime_seconds": time.time() - start_time,
    }
    reporting.save_json(
        summary, args.results_dir / f"{args.architecture}_metrics_summary.json"
    )

    # Keep a clearly named copy for slide-building convenience.
    test_predictions.to_csv(
        tables_dir / "deep_mc_dropout_uncertainty_test.csv", index=False
    )
    val_predictions.to_csv(
        tables_dir / "deep_mc_dropout_uncertainty_val.csv", index=False
    )

    print(
        f"saved summary: {args.results_dir / f'{args.architecture}_metrics_summary.json'}"
    )
    print(
        f"saved real MC probabilities: {args.runs_dir / f'{args.architecture}_test_mc_probabilities.npy'}"
    )
    print(f"saved figures: {figures_dir}")


if __name__ == "__main__":
    main()
