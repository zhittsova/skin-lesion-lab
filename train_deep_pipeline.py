"""Train a CNN baseline and estimate uncertainty with MC dropout."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import roc_curve
from src import (
    data,
    deep,
    evaluation,
    plots,
    reporting,
    run_contract,
    splitting,
    uncertainty,
)
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
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=dataset_path)
    parser.add_argument("--runs-dir", type=Path, default=project_path / "runs")
    parser.add_argument("--run-id", help="Unique run ID; generated when omitted.")
    parser.add_argument(
        "--resume-from", help="Failed run ID to retry in a new directory."
    )
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
    return parser.parse_args()


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
    ax.set_ylabel("Development holdout images")
    add_chart_header(
        fig,
        ax,
        "Distribution of predictive uncertainties (MC Dropout)",
        f"Real development-set inference; median std={median:.3f}, 90th percentile={p90:.3f}.",
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
        "MC Dropout epistemic uncertainty on the development set",
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
    run = run_contract.RunRecord.start(
        args.runs_dir,
        run_id=getattr(args, "run_id", None),
        pipeline=f"deep_{args.architecture}",
        config=vars(args),
        inputs={
            "metadata": args.metadata_path,
            "split_manifest": args.split_manifest,
            "dependency_lock": Path(__file__).with_name("uv.lock"),
        },
        resume_from=getattr(args, "resume_from", None),
    )
    local = copy.copy(args)
    local.results_dir = run.path / "results"
    local.models_dir = run.path / "models"
    local.runs_dir = run.path / "arrays"
    stage = "training"
    try:
        device = _run(local)
        run.record["environment"]["device"] = str(device)
        stage = "finalize"
        manifest = json.loads(args.split_manifest.read_text())
        run.finish(
            manifest,
            {
                role: local.results_dir
                / "tables"
                / f"{args.architecture}_predictions_{role}.csv"
                for role in ("calibration", "development")
            },
        )
    except BaseException as error:
        run.fail(error, stage=stage)
        raise
    print(f"completed run: {run.path}")


def _run(args) -> None:
    if args.epochs <= 0:
        raise ValueError(
            "epochs must be positive; a checkpoint must be selected this run"
        )
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

    split_indices, split_report = splitting.load_development_split(
        df, args.split_manifest
    )
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
    selection_loader = make_loader(
        image_ids,
        labels,
        split_indices["selection"],
        args.images_dir,
        args.image_size,
        args.batch_size,
        train=False,
        num_workers=args.num_workers,
    )
    calibration_loader = make_loader(
        image_ids,
        labels,
        split_indices["calibration"],
        args.images_dir,
        args.image_size,
        args.batch_size,
        train=False,
        num_workers=args.num_workers,
    )
    development_loader = make_loader(
        image_ids,
        labels,
        split_indices["development"],
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
    checkpoint_selected = False
    best_checkpoint = args.models_dir / f"{args.architecture}_mc_dropout.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = deep.train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        selection_loss = deep.evaluate_loss(model, selection_loader, criterion, device)
        selection_pred = deep.predict_probabilities(model, selection_loader, device)
        selection_metrics = metrics_for_threshold(
            selection_pred["label"],
            selection_pred["probability"],
            threshold=0.5,
            cost_fn=args.cost_fn,
            cost_fp=args.cost_fp,
        )
        score = float(selection_metrics["roc_auc"])
        if np.isnan(score):
            score = -selection_loss

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "selection_loss": selection_loss,
            "selection_auc": selection_metrics["roc_auc"],
            "selection_recall_at_0_5": selection_metrics["recall"],
            "selection_specificity_at_0_5": selection_metrics["specificity"],
        }
        history.append(history_row)
        print(
            f"epoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, selection_loss={selection_loss:.4f}, "
            f"selection_auc={selection_metrics['roc_auc']:.4f}"
        )

        if np.isfinite(score) and score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "split_hash": split_report["split_hash"],
                    "cohort_hash": split_report["cohort_hash"],
                    "architecture": args.architecture,
                    "image_size": args.image_size,
                    "dropout": args.dropout,
                    "pretrained": args.pretrained,
                    "fine_tune_backbone": args.fine_tune_backbone,
                    "epoch": epoch,
                    "selection_auc": selection_metrics["roc_auc"],
                },
                best_checkpoint,
            )
            checkpoint_selected = True

    if not checkpoint_selected:
        raise ValueError("no finite checkpoint selected this run")
    checkpoint = torch.load(best_checkpoint, map_location=device)
    if any(
        checkpoint.get(key) != split_report[key]
        for key in ("split_hash", "cohort_hash")
    ):
        raise ValueError("checkpoint does not match the current split manifest")
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\n4. MC Dropout inference")
    calibration_mc = deep.predict_with_mc_dropout(
        model=model,
        dataloader=calibration_loader,
        device=device,
        n_passes=args.mc_samples,
    )

    np.save(
        args.runs_dir / f"{args.architecture}_calibration_mc_probabilities.npy",
        calibration_mc["all_probabilities"],
    )

    print("\n5. threshold selection and evaluation")
    formula_threshold = args.cost_fp / (args.cost_fp + args.cost_fn)
    selected_threshold, threshold_sweep = find_best_threshold_by_validation_cost(
        calibration_mc["label"],
        calibration_mc["mean_probability"],
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
    )
    threshold_sweep.to_csv(
        tables_dir / f"{args.architecture}_calibration_threshold_sweep.csv",
        index=False,
    )

    development_mc = deep.predict_with_mc_dropout(
        model=model,
        dataloader=development_loader,
        device=device,
        n_passes=args.mc_samples,
    )

    np.save(
        args.runs_dir / f"{args.architecture}_development_mc_probabilities.npy",
        development_mc["all_probabilities"],
    )
    np.save(
        args.runs_dir / f"{args.architecture}_development_probability.npy",
        development_mc["mean_probability"],
    )
    np.save(
        args.runs_dir / f"{args.architecture}_development_uncertainty.npy",
        development_mc["uncertainty"],
    )
    np.save(
        args.runs_dir / f"{args.architecture}_development_y_true.npy",
        development_mc["label"],
    )

    metrics_by_split = {}
    for split_name, mc_result in {
        "calibration": calibration_mc,
        "development": development_mc,
    }.items():
        y_true = mc_result["label"]
        y_prob = mc_result["mean_probability"]
        metrics_by_split[split_name] = {
            "cost_formula_threshold": metrics_for_threshold(
                y_true, y_prob, formula_threshold, args.cost_fn, args.cost_fp
            ),
            "calibration_cost_threshold": metrics_for_threshold(
                y_true, y_prob, selected_threshold, args.cost_fn, args.cost_fp
            ),
            "map_threshold": metrics_for_threshold(
                y_true, y_prob, 0.5, args.cost_fn, args.cost_fp
            ),
        }

    ece_by_split = {
        "calibration": evaluation.compute_calibration_error(
            calibration_mc["label"], calibration_mc["mean_probability"]
        ),
        "development": evaluation.compute_calibration_error(
            development_mc["label"], development_mc["mean_probability"]
        ),
    }
    save_deep_metrics_tables(
        metrics_by_split, ece_by_split, tables_dir, args.architecture
    )

    print("\n6. saving predictions and plots")
    development_predictions = save_predictions(
        development_mc,
        threshold=selected_threshold,
        output_path=tables_dir / f"{args.architecture}_predictions_development.csv",
    )
    save_predictions(
        calibration_mc,
        threshold=selected_threshold,
        output_path=tables_dir / f"{args.architecture}_predictions_calibration.csv",
    )

    fpr, tpr, auc_score = evaluation.compute_roc_curve(
        development_mc["label"], development_mc["mean_probability"]
    )
    plots.plot_roc_curve(
        fpr,
        tpr,
        auc_score,
        save_path=str(figures_dir / f"{args.architecture}_roc_curve.png"),
        title=f"{args.architecture} ROC curve",
    )
    cm_development = evaluation.compute_confusion_matrix(
        development_mc["label"],
        (development_mc["mean_probability"] >= selected_threshold).astype(int),
    )
    plots.plot_confusion_matrix(
        cm_development,
        save_path=str(figures_dir / f"{args.architecture}_confusion_matrix.png"),
        title=f"{args.architecture} confusion matrix",
    )

    plot_uncertainty_distribution(
        development_predictions,
        figures_dir / f"{args.architecture}_mc_dropout_uncertainty_distribution.png",
    )
    plot_mean_vs_epistemic_uncertainty(
        development_predictions,
        figures_dir
        / f"{args.architecture}_mc_dropout_mean_vs_epistemic_uncertainty.png",
        threshold=formula_threshold,
    )

    fpr_table, tpr_table, roc_thresholds = roc_curve(
        development_mc["label"], development_mc["mean_probability"]
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
            "selected_threshold_from_calibration": selected_threshold,
            "map_threshold": 0.5,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "pos_weight": pos_weight,
            "history": history,
            "best_checkpoint": f"models/{best_checkpoint.name}",
        },
        "metrics": metrics_by_split,
        "expected_calibration_error": ece_by_split,
        "mc_dropout_uncertainty": {
            "development_mean_predictive_std": float(
                development_predictions["predictive_std"].mean()
            ),
            "development_median_predictive_std": float(
                development_predictions["predictive_std"].median()
            ),
            "development_p90_predictive_std": float(
                development_predictions["predictive_std"].quantile(0.9)
            ),
            "development_mean_mutual_information": float(
                development_predictions["mutual_information"].mean()
            ),
        },
        "artifacts": {
            "development_mc_probabilities": f"arrays/{args.architecture}_development_mc_probabilities.npy",
            "development_predictions": f"results/tables/{args.architecture}_predictions_development.csv",
            "uncertainty_distribution": f"results/figures/{args.architecture}_mc_dropout_uncertainty_distribution.png",
            "mean_vs_epistemic_uncertainty": f"results/figures/{args.architecture}_mc_dropout_mean_vs_epistemic_uncertainty.png",
        },
        "runtime_seconds": time.time() - start_time,
    }
    reporting.save_json(
        summary, args.results_dir / f"{args.architecture}_metrics_summary.json"
    )

    print(
        f"saved summary: {args.results_dir / f'{args.architecture}_metrics_summary.json'}"
    )
    print(
        f"saved real MC probabilities: {args.runs_dir / f'{args.architecture}_development_mc_probabilities.npy'}"
    )
    print(f"saved figures: {figures_dir}")
    return device


if __name__ == "__main__":
    main()
