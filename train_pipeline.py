"""
Small script for the classical pipeline.

All models use the frozen development allocation and common run contract.
The prevalence reference uses labels only; logistic and GMM use HSV histograms.
"""

import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from src import (
    bayes,
    calibration,
    classical,
    data,
    evaluation,
    features,
    gmm,
    plots,
    reporting,
    run_contract,
    splitting,
)
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    project_path = Path(__file__).parent
    dataset_path = project_path / "data" / "raw"
    parser = argparse.ArgumentParser(
        description="Train a prevalence, HSV logistic, or HSV GMM baseline."
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
        "--model", choices=("gmm", "prevalence", "logistic"), default="gmm"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gmm-max-components", type=int, default=10)
    parser.add_argument(
        "--gmm-covariance-type",
        choices=("full", "tied", "diag", "spherical"),
        default="full",
    )
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-4)
    parser.add_argument("--gmm-max-iter", type=int, default=300)
    parser.add_argument(
        "--resume-from", help="Failed run ID to retry in a new directory."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_kind = getattr(args, "model", "gmm")
    run = run_contract.RunRecord.start(
        args.runs_dir,
        run_id=getattr(args, "run_id", None),
        pipeline=f"classical_{model_kind}",
        config={
            **vars(args),
            "training_seed": getattr(args, "seed", 42),
            "decision_policy_version": 1,
        },
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
    stage = "training"
    try:
        _run(local)
        run.record["environment"]["device"] = "cpu"
        stage = "finalize"
        manifest = json.loads(args.split_manifest.read_text())
        run.finish(
            manifest,
            {
                role: local.results_dir / "tables" / f"predictions_{role}.csv"
                for role in ("train", "selection", "calibration", "development")
            },
        )
    except BaseException as error:
        run.fail(error, stage=stage)
        raise
    print(f"completed run: {run.path}")


def _run(args):
    model_kind = getattr(args, "model", "gmm")
    METADATA_PATH = args.metadata_path
    IMAGES_DIR = args.images_dir
    RESULTS_PATH = args.results_dir
    MODELS_PATH = args.models_dir

    (RESULTS_PATH / "figures").mkdir(parents=True, exist_ok=True)
    (RESULTS_PATH / "tables").mkdir(parents=True, exist_ok=True)
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    print("""
Bayesian melanoma classifier

1. loading data""")
    plots.set_style()
    df, image_ids, labels = data.prepare_dataset(
        str(METADATA_PATH),
        str(IMAGES_DIR),
        source=args.source,
        attrition_path=RESULTS_PATH / "cohort_attrition.json",
    )

    class_stats = data.get_class_statistics(labels)
    print(f"""
dataset loaded: {len(image_ids)} images
benign: {class_stats["benign"]}
melanoma: {class_stats["melanoma"]}""")

    plots.plot_class_distribution(
        labels, save_path=str(RESULTS_PATH / "figures" / "class_distribution.png")
    )
    plots.plot_class_pie(
        labels, save_path=str(RESULTS_PATH / "figures" / "class_distribution_pie.png")
    )

    lesion_ids = data.get_lesion_ids(df)

    print("\n2. split data by lesion id")
    split_indices, split_report = splitting.load_development_split(
        df, args.split_manifest
    )

    splitting.print_split_summary(labels, split_indices, lesion_ids)
    reporting.save_json(split_report, RESULTS_PATH / "split_summary.json")
    reporting.save_split_tables(split_report, RESULTS_PATH / "tables")

    print("\n3. prepare model inputs")

    def load_and_extract_features(indices, desc=""):
        hsv_images = []
        valid_indices = []

        for idx in tqdm(indices, desc=desc):
            img_id = image_ids[idx]
            img_path = IMAGES_DIR / f"{img_id}.jpg"

            hsv_img = data.load_image_hsv(str(img_path), target_size=(256, 256))
            hsv_images.append(hsv_img)
            valid_indices.append(idx)

        X = features.compute_hsv_histograms_batch(hsv_images)

        return X, valid_indices

    train_valid_idx = split_indices["train"]
    selection_valid_idx = split_indices["selection"]
    development_valid_idx = split_indices["development"]
    calibration_valid_idx = split_indices["calibration"]
    train_mean = train_std = None
    if model_kind != "prevalence":
        X_train_raw, train_valid_idx = load_and_extract_features(
            train_valid_idx, "Training set"
        )
        X_selection_raw, selection_valid_idx = load_and_extract_features(
            selection_valid_idx, "Selection set"
        )
        X_calibration_raw, calibration_valid_idx = load_and_extract_features(
            calibration_valid_idx, "Calibration set"
        )
        X_development_raw, development_valid_idx = load_and_extract_features(
            development_valid_idx, "Development holdout set"
        )
        if model_kind == "gmm":
            # GMM scaling uses training statistics; logistic owns its train scaler.
            X_train, train_mean, train_std = features.standardize_features(X_train_raw)
            X_calibration = features.apply_standardization(
                X_calibration_raw, train_mean, train_std
            )
            X_selection = features.apply_standardization(
                X_selection_raw, train_mean, train_std
            )
            X_development = features.apply_standardization(
                X_development_raw, train_mean, train_std
            )
        else:
            X_calibration = X_calibration_raw
            X_train, X_selection, X_development = (
                X_train_raw,
                X_selection_raw,
                X_development_raw,
            )
    else:
        X_train = np.empty((len(train_valid_idx), 0))
        X_selection = np.empty((len(selection_valid_idx), 0))
        X_development = np.empty((len(development_valid_idx), 0))

    y_calibration = labels[calibration_valid_idx]
    y_train = labels[train_valid_idx]
    y_selection = labels[selection_valid_idx]
    y_development = labels[development_valid_idx]

    print(f"""
feature extraction done
feature dimension: {X_train.shape[1]}
train: {X_train.shape[0]} samples
selection: {X_selection.shape[0]} samples
development: {X_development.shape[0]} samples""")

    print(f"\n4. fit {model_kind} baseline")
    class_priors = bayes.compute_class_priors(y_train)
    selection_info = None
    if model_kind == "prevalence":
        fitted = classical.fit_prevalence(y_train)
        probs_calibration = classical.predict_prevalence(fitted, len(y_calibration))
        probs_train = classical.predict_prevalence(fitted, len(y_train))
        probs_selection = classical.predict_prevalence(fitted, len(y_selection))
    elif model_kind == "logistic":
        fitted, selection_info = classical.fit_logistic(
            X_train,
            y_train,
            X_selection,
            y_selection,
            seed=getattr(args, "seed", 42),
        )
        probs_calibration = classical.predict_logistic(fitted, X_calibration)
        probs_train = classical.predict_logistic(fitted, X_train)
        probs_selection = classical.predict_logistic(fitted, X_selection)
    elif model_kind == "gmm":
        fitted, selection_info = gmm.train_class_gmms_auto(
            X_train,
            y_train,
            max_components=getattr(args, "gmm_max_components", 10),
            cv_type=getattr(args, "gmm_covariance_type", "full"),
            random_state=getattr(args, "seed", 42),
            reg_covar=getattr(args, "gmm_reg_covar", 1e-4),
            max_iter=getattr(args, "gmm_max_iter", 300),
        )
        gmm.print_gmm_summary(selection_info)

        def posterior(x):
            log_likelihoods = gmm.compute_class_likelihoods(fitted, x)
            probabilities = bayes.compute_posterior_probabilities(
                log_likelihoods, class_priors
            )
            return bayes.get_melanoma_probability(probabilities)

        probs_calibration = posterior(X_calibration)
        probs_train = posterior(X_train)
        probs_selection = posterior(X_selection)
    else:
        raise ValueError("invalid classical model")

    for probabilities in (probs_train, probs_selection, probs_calibration):
        if not np.isfinite(probabilities).all() or np.any(
            (probabilities < 0) | (probabilities > 1)
        ):
            raise ValueError("invalid classical probabilities")

    policy = calibration.fit_policy(
        y_calibration,
        probs_calibration,
        image_ids=image_ids[calibration_valid_idx],
        split_hash=split_report["split_hash"],
        cost_fn=10.0,
        cost_fp=1.0,
    )
    reporting.save_json(policy, MODELS_PATH / "decision_policy.json")
    if model_kind == "prevalence":
        probs_development = classical.predict_prevalence(fitted, len(y_development))
    elif model_kind == "logistic":
        probs_development = classical.predict_logistic(fitted, X_development)
    else:
        probs_development = posterior(X_development)
    raw_scores = dict(
        train=probs_train,
        selection=probs_selection,
        calibration=probs_calibration,
        development=probs_development,
    )
    applied = {
        role: calibration.apply_policy(policy, scores)
        for role, scores in raw_scores.items()
    }
    probs_train, probs_selection, probs_calibration, probs_development = (
        applied[role]["calibrated_probability"]
        for role in ("train", "selection", "calibration", "development")
    )
    calibration_report = {
        role: calibration.policy_report(policy, y, raw_scores[role])
        for role, y in (("calibration", y_calibration), ("development", y_development))
    }
    reporting.save_json(calibration_report, RESULTS_PATH / "calibration_report.json")
    print("\n5. compute fitted probabilities")

    print(f"""
fitted probabilities computed
training class priors: P(benign)={class_priors[0]:.4f}, P(melanoma)={class_priors[1]:.4f}

6. threshold with the cost matrix""")

    # missing melanoma is worse than a false alarm
    COST_FN = 10.0
    COST_FP = 1.0

    bayes.print_cost_analysis(COST_FN, COST_FP)
    cost_threshold = policy["decision"]["threshold"]
    map_threshold = 0.5

    preds_train = (probs_train >= cost_threshold).astype(int)
    preds_selection = (probs_selection >= cost_threshold).astype(int)
    preds_development = (probs_development >= cost_threshold).astype(int)

    preds_train_map = (probs_train >= map_threshold).astype(int)
    preds_selection_map = (probs_selection >= map_threshold).astype(int)
    preds_development_map = (probs_development >= map_threshold).astype(int)

    print("\n7. evaluate")

    metrics_train = evaluation.compute_classification_metrics(
        y_train, preds_train, probs_train
    )
    metrics_selection = evaluation.compute_classification_metrics(
        y_selection, preds_selection, probs_selection
    )
    metrics_development = evaluation.compute_classification_metrics(
        y_development, preds_development, probs_development
    )
    metrics_train = evaluation.add_average_cost(
        metrics_train, len(y_train), COST_FN, COST_FP
    )
    metrics_selection = evaluation.add_average_cost(
        metrics_selection, len(y_selection), COST_FN, COST_FP
    )
    metrics_development = evaluation.add_average_cost(
        metrics_development, len(y_development), COST_FN, COST_FP
    )

    metrics_train_map = evaluation.compute_classification_metrics(
        y_train, preds_train_map, probs_train
    )
    metrics_selection_map = evaluation.compute_classification_metrics(
        y_selection, preds_selection_map, probs_selection
    )
    metrics_development_map = evaluation.compute_classification_metrics(
        y_development, preds_development_map, probs_development
    )
    metrics_train_map = evaluation.add_average_cost(
        metrics_train_map, len(y_train), COST_FN, COST_FP
    )
    metrics_selection_map = evaluation.add_average_cost(
        metrics_selection_map, len(y_selection), COST_FN, COST_FP
    )
    metrics_development_map = evaluation.add_average_cost(
        metrics_development_map, len(y_development), COST_FN, COST_FP
    )

    evaluation.print_evaluation_summary(
        metrics_train, f"Training (Cost Threshold {cost_threshold:.4f})"
    )
    evaluation.print_evaluation_summary(
        metrics_selection, f"Selection (Cost Threshold {cost_threshold:.4f})"
    )
    evaluation.print_evaluation_summary(
        metrics_development,
        f"Development holdout (Cost Threshold {cost_threshold:.4f})",
    )

    def print_threshold_comparison(split_name, cost_metrics, map_metrics):
        msg = f"""
{split_name} threshold compare
cost threshold {cost_threshold:.4f}: recall={cost_metrics["recall"]:.3f}, specificity={cost_metrics["specificity"]:.3f}, FP={cost_metrics["fp"]}, FN={cost_metrics["fn"]}
MAP threshold  {map_threshold:.4f}: recall={map_metrics["recall"]:.3f}, specificity={map_metrics["specificity"]:.3f}, FP={map_metrics["fp"]}, FN={map_metrics["fn"]}
"""
        print(msg)

    print_threshold_comparison("Training", metrics_train, metrics_train_map)
    print_threshold_comparison("Selection", metrics_selection, metrics_selection_map)
    print_threshold_comparison(
        "Development holdout", metrics_development, metrics_development_map
    )

    print("\n8. calibration")

    ece_train = evaluation.compute_calibration_error(y_train, probs_train)
    ece_selection = evaluation.compute_calibration_error(y_selection, probs_selection)
    ece_development = evaluation.compute_calibration_error(
        y_development, probs_development
    )

    print(f"""
Expected Calibration Error (ECE)
train: {ece_train:.4f}
selection: {ece_selection:.4f}
development: {ece_development:.4f}""")

    metrics_by_split = {
        "train": {"cost_threshold": metrics_train, "map_threshold": metrics_train_map},
        "selection": {
            "cost_threshold": metrics_selection,
            "map_threshold": metrics_selection_map,
        },
        "development": {
            "cost_threshold": metrics_development,
            "map_threshold": metrics_development_map,
        },
    }
    metrics_by_split["calibration"] = {}
    for name, cutoff in (
        ("cost_threshold", cost_threshold),
        ("map_threshold", map_threshold),
    ):
        scores = evaluation.compute_classification_metrics(
            y_calibration, (probs_calibration >= cutoff).astype(int), probs_calibration
        )
        metrics_by_split["calibration"][name] = evaluation.add_average_cost(
            scores, len(y_calibration), COST_FN, COST_FP
        )
    ece_by_split = {
        "calibration": evaluation.compute_calibration_error(
            y_calibration, probs_calibration
        ),
        "train": ece_train,
        "selection": ece_selection,
        "development": ece_development,
    }
    reporting.save_metrics_tables(
        metrics_by_split, ece_by_split, RESULTS_PATH / "tables"
    )

    metrics_summary = {
        "pipeline": f"classical_{model_kind}",
        "task": "binary melanoma-vs-benign classification",
        "positive_class": "melanoma",
        "feature_representation": {
            "name": (
                "training prevalence"
                if model_kind == "prevalence"
                else "HSV per-channel histograms"
            ),
            "h_bins": 0 if model_kind == "prevalence" else 64,
            "s_bins": 0 if model_kind == "prevalence" else 32,
            "v_bins": 0 if model_kind == "prevalence" else 32,
            "dimension": int(X_train.shape[1]),
            "image_size": None if model_kind == "prevalence" else [256, 256],
            "standardization": (
                "none" if model_kind == "prevalence" else "train statistics only"
            ),
        },
        "split": split_report,
        "cost_matrix": {
            "false_negative": COST_FN,
            "false_positive": COST_FP,
            "true_positive": 0.0,
            "true_negative": 0.0,
            "cost_threshold": cost_threshold,
            "map_threshold": map_threshold,
        },
        "class_priors": {
            "benign": float(class_priors[0]),
            "melanoma": float(class_priors[1]),
        },
        "metrics": metrics_by_split,
        "expected_calibration_error": ece_by_split,
        "score_semantics": "calibration-fitted probability; raw scores retained separately",
        "decision_policy": "models/decision_policy.json",
        "calibration_report": calibration_report,
        "notes": [
            "All model comparisons use the same frozen development manifest.",
            "Costs are illustrative. Fitted calibration does not establish deployment calibration.",
        ],
    }
    if model_kind == "gmm":
        metrics_summary["bic_selection"] = reporting.selection_summary(selection_info)
        metrics_summary["gmm_fit"] = {
            ("benign" if label == 0 else "melanoma"): {
                "requested_max_components": info["requested_max_components"],
                "effective_max_components": info["effective_max_components"],
                "converged": info["converged"],
                "n_iter": info["n_iter"],
                "reg_covar": info["reg_covar"],
                "covariance_type": info["covariance_type"],
                "max_iter": info["max_iter"],
                "candidate_warnings": info["bic_info"]["initialization_warnings"],
                "fit_warnings": info["fit_warnings"],
            }
            for label, info in selection_info.items()
        }
    elif model_kind == "logistic":
        metrics_summary["model_selection"] = selection_info
    else:
        metrics_summary["training_prevalence"] = fitted["prevalence"]
        metrics_summary["majority_class"] = fitted["majority_class"]
    reporting.save_json(metrics_summary, RESULTS_PATH / "metrics_summary.json")

    print("\n9. make plots")

    fpr_development, tpr_development, auc_development = evaluation.compute_roc_curve(
        y_development, probs_development
    )
    precision_development, recall_development, pr_auc_development = (
        evaluation.compute_pr_curve(y_development, probs_development)
    )

    plots.plot_roc_curve(
        fpr_development,
        tpr_development,
        auc_development,
        save_path=str(RESULTS_PATH / "figures" / "roc_curve.png"),
    )
    plots.plot_pr_curve(
        precision_development,
        recall_development,
        pr_auc_development,
        baseline=float(np.mean(y_development)),
        save_path=str(RESULTS_PATH / "figures" / "pr_curve.png"),
    )

    cm_development = evaluation.compute_confusion_matrix(
        y_development, preds_development
    )
    plots.plot_confusion_matrix(
        cm_development, save_path=str(RESULTS_PATH / "figures" / "confusion_matrix.png")
    )
    plots.plot_threshold_comparison(
        {
            "cost threshold": metrics_development,
            "MAP threshold": metrics_development_map,
        },
        save_path=str(RESULTS_PATH / "figures" / "threshold_comparison.png"),
    )

    mean_probs, freqs, bin_sizes = evaluation.compute_calibration_curve(
        y_development, probs_development
    )
    plots.plot_calibration_curve(
        mean_probs,
        freqs,
        bin_sizes,
        save_path=str(RESULTS_PATH / "figures" / "calibration.png"),
    )

    if model_kind == "gmm":
        for class_label in (0, 1):
            info = selection_info[class_label]
            class_name = "benign" if class_label == 0 else "melanoma"
            plots.plot_bic_scores(
                info["bic_info"]["component_range"],
                info["bic_info"]["bic_scores"],
                info["n_components"],
                save_path=str(RESULTS_PATH / "figures" / f"bic_{class_name}.png"),
            )

    print("""
plots generated and saved

10. save predictions""")

    def save_predictions(
        image_ids_subset,
        predictions,
        map_predictions,
        probabilities,
        labels,
        split_name,
    ):
        df_pred = pd.DataFrame(
            {
                "image_id": image_ids_subset,
                "target": labels,
                "prediction": predictions,
                "prediction_map": map_predictions,
                "prob_melanoma": probabilities,
                "raw_score": raw_scores[split_name],
                "corrected_score": applied[split_name]["corrected_score"],
                "review_recommended": applied[split_name]["review_recommended"],
                "score": np.where(predictions == 1, probabilities, 1 - probabilities),
                "model_name": f"classical_{model_kind}",
            }
        )

        csv_path = RESULTS_PATH / "tables" / f"predictions_{split_name}.csv"
        df_pred.to_csv(csv_path, index=False)
        print(f"  Saved {len(df_pred)} predictions to {csv_path.name}")

    save_predictions(
        image_ids[train_valid_idx],
        preds_train,
        preds_train_map,
        probs_train,
        y_train,
        "train",
    )
    save_predictions(
        image_ids[selection_valid_idx],
        preds_selection,
        preds_selection_map,
        probs_selection,
        y_selection,
        "selection",
    )
    save_predictions(
        image_ids[development_valid_idx],
        preds_development,
        preds_development_map,
        probs_development,
        y_development,
        "development",
    )

    save_predictions(
        image_ids[calibration_valid_idx],
        applied["calibration"]["prediction"],
        (probs_calibration >= 0.5).astype(int),
        probs_calibration,
        y_calibration,
        "calibration",
    )

    print("""
11. save model""")

    model_info = {
        "decision_policy": policy,
        "model_kind": model_kind,
        "split_hash": split_report["split_hash"],
        "cohort_hash": split_report["cohort_hash"],
        "protocol_version": split_report["protocol_version"],
        "models": fitted if model_kind == "gmm" else None,
        "fitted": fitted,
        "selection_info": selection_info,
        "train_mean": train_mean,
        "train_std": train_std,
        "class_priors": class_priors,
        "threshold": cost_threshold,
        "map_threshold": map_threshold,
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
    }

    model_filename = (
        "bayesian_gmm_model.pkl" if model_kind == "gmm" else f"{model_kind}_model.pkl"
    )
    with open(MODELS_PATH / model_filename, "wb") as f:
        pickle.dump(model_info, f)

    print(f"  Saved model to {MODELS_PATH / model_filename}")

    print(f"""
done
results: {RESULTS_PATH}
model: {MODELS_PATH}""")


if __name__ == "__main__":
    main()
