"""Build report tables from existing pipeline outputs without rerunning images."""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from src import data, evaluation, plots, reporting, splitting


def metrics_from_predictions(
    df: pd.DataFrame, prediction_column: str, cost_fn: float, cost_fp: float
) -> dict[str, float | int]:
    metrics = evaluation.compute_classification_metrics(
        df["target"].to_numpy(),
        df[prediction_column].to_numpy(),
        df["prob_melanoma"].to_numpy(),
    )
    return evaluation.add_average_cost(metrics, len(df), cost_fn, cost_fp)


def main() -> None:
    project_path = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Summarize a trusted local GMM run.")
    parser.add_argument(
        "--metadata-path", type=Path, default=project_path / "data/raw/metadata.csv"
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=project_path / "data/raw")
    parser.add_argument(
        "--source", choices=["ham10000", "isic2018_task3"], required=True
    )
    parser.add_argument("--results-dir", type=Path, default=project_path / "results")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=project_path / "models/bayesian_gmm_model.pkl",
    )
    args = parser.parse_args()
    dataset_path = args.images_dir
    metadata_path = args.metadata_path
    results_path = args.results_dir
    tables_path = results_path / "tables"
    figures_path = results_path / "figures"
    model_path = args.model_path

    with model_path.open("rb") as f:
        model_info = pickle.load(f)

    cost_fn = float(model_info["cost_fn"])
    cost_fp = float(model_info["cost_fp"])

    df, _, labels = data.prepare_dataset(
        str(metadata_path),
        str(dataset_path),
        source=args.source,
        attrition_path=results_path / "cohort_attrition.json",
    )
    split_indices, split_report = splitting.load_development_split(
        df, args.split_manifest
    )
    if model_info.get("split_hash") != split_report["split_hash"]:
        raise ValueError("model split hash does not match manifest")

    metrics_by_split = {}
    ece_by_split = {}
    for split_name in ["train", "selection", "development"]:
        pred_df = pd.read_csv(tables_path / f"predictions_{split_name}.csv")
        expected = df.iloc[split_indices[split_name]]
        if (
            pred_df.image_id.tolist() != expected.isic_id.tolist()
            or pred_df.target.tolist() != expected.target.tolist()
        ):
            raise ValueError("prediction membership or labels do not match manifest")
        cost_metrics = metrics_from_predictions(pred_df, "prediction", cost_fn, cost_fp)
        map_metrics = metrics_from_predictions(
            pred_df, "prediction_map", cost_fn, cost_fp
        )
        metrics_by_split[split_name] = {
            "cost_threshold": cost_metrics,
            "map_threshold": map_metrics,
        }
        ece_by_split[split_name] = evaluation.compute_calibration_error(
            pred_df["target"].to_numpy(),
            pred_df["prob_melanoma"].to_numpy(),
        )

    reporting.save_json(split_report, results_path / "split_summary.json")
    reporting.save_split_tables(split_report, tables_path)
    reporting.save_metrics_tables(metrics_by_split, ece_by_split, tables_path)

    plots.set_style()
    plots.plot_threshold_comparison(
        {
            "cost threshold": metrics_by_split["development"]["cost_threshold"],
            "MAP threshold": metrics_by_split["development"]["map_threshold"],
        },
        save_path=str(figures_path / "threshold_comparison.png"),
    )

    metrics_summary = {
        "pipeline": "Classical Bayesian HSV + GMM skin lesion triage",
        "task": "binary melanoma-vs-benign classification",
        "positive_class": "melanoma",
        "cost_matrix": {
            "false_negative": cost_fn,
            "false_positive": cost_fp,
            "true_positive": 0.0,
            "true_negative": 0.0,
            "cost_threshold": float(model_info["threshold"]),
            "map_threshold": float(model_info["map_threshold"]),
        },
        "class_priors": {
            "benign": float(model_info["class_priors"][0]),
            "melanoma": float(model_info["class_priors"][1]),
        },
        "bic_selection": reporting.selection_summary(model_info["selection_info"]),
        "split": split_report,
        "metrics": metrics_by_split,
        "expected_calibration_error": ece_by_split,
        "artifact_sources": {
            "predictions": "results/tables/predictions_{train,selection,development}.csv",
            "model": "models/bayesian_gmm_model.pkl",
        },
    }
    reporting.save_json(metrics_summary, results_path / "metrics_summary.json")

    print(f"Saved metrics summary to {results_path / 'metrics_summary.json'}")
    print(f"Saved metric tables to {tables_path}")
    print(f"Saved threshold figure to {figures_path / 'threshold_comparison.png'}")


if __name__ == "__main__":
    main()
