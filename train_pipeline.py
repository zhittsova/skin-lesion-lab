"""
Small script for the classical pipeline.

1. Load HAM10000 images and convert to HSV colour space.
2. Compute per-channel histograms (H, S, V) and concatenate into one feature vector.
3. Load the frozen development allocation shared by all models.
4. Fit a Gaussian mixture model per class (melanoma, benign) using EM.
5. Select the number of components K by minimizing BIC on the training set.
6. Compute class-conditional likelihoods and apply Bayes rule to get posteriors.
7. Derive decision threshold from asymmetric cost matrix; compare with MAP threshold.
8. Evaluate with ROC curve, confusion matrix, and reliability diagram.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from src import bayes, data, evaluation, features, gmm, plots, reporting, splitting
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    project_path = Path(__file__).parent
    dataset_path = project_path / "data" / "raw"
    parser = argparse.ArgumentParser(
        description="Train the HSV histogram GMM baseline."
    )
    parser.add_argument(
        "--metadata-path", type=Path, default=dataset_path / "metadata.csv"
    )
    parser.add_argument(
        "--source", choices=["ham10000", "isic2018_task3"], required=True
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=dataset_path)
    parser.add_argument("--results-dir", type=Path, default=project_path / "results")
    parser.add_argument("--models-dir", type=Path, default=project_path / "models")
    return parser.parse_args()


def main():
    args = parse_args()
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

    print("\n3. load images and make HSV histograms")

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

    X_train_raw, train_valid_idx = load_and_extract_features(
        split_indices["train"], "Training set"
    )
    X_selection_raw, selection_valid_idx = load_and_extract_features(
        split_indices["selection"], "Selection set"
    )
    X_development_raw, development_valid_idx = load_and_extract_features(
        split_indices["development"], "Development holdout set"
    )

    # using train stats here, so selection/development dont leak into the scaling
    X_train, train_mean, train_std = features.standardize_features(X_train_raw)
    X_selection = features.apply_standardization(X_selection_raw, train_mean, train_std)
    X_development = features.apply_standardization(
        X_development_raw, train_mean, train_std
    )

    y_train = labels[train_valid_idx]
    y_selection = labels[selection_valid_idx]
    y_development = labels[development_valid_idx]

    print(f"""
feature extraction done
feature dimension: {X_train.shape[1]}
train: {X_train.shape[0]} samples
selection: {X_selection.shape[0]} samples
development: {X_development.shape[0]} samples""")

    print("\n4. fit GMMs and pick K with BIC")

    models, selection_info = gmm.train_class_gmms_auto(
        X_train, y_train, max_components=10, cv_type="full", random_state=42
    )

    gmm.print_gmm_summary(selection_info)

    print("\n5. compute posterior probabilities")

    log_likelihoods_train = gmm.compute_class_likelihoods(models, X_train)
    class_priors = bayes.compute_class_priors(y_train)
    posteriors_train = bayes.compute_posterior_probabilities(
        log_likelihoods_train, class_priors
    )
    probs_train = bayes.get_melanoma_probability(posteriors_train)

    log_likelihoods_selection = gmm.compute_class_likelihoods(models, X_selection)
    posteriors_selection = bayes.compute_posterior_probabilities(
        log_likelihoods_selection, class_priors
    )
    probs_selection = bayes.get_melanoma_probability(posteriors_selection)

    log_likelihoods_development = gmm.compute_class_likelihoods(models, X_development)
    posteriors_development = bayes.compute_posterior_probabilities(
        log_likelihoods_development, class_priors
    )
    probs_development = bayes.get_melanoma_probability(posteriors_development)

    print(f"""
posterior probabilities computed
class priors: P(benign)={class_priors[0]:.4f}, P(melanoma)={class_priors[1]:.4f}

6. threshold with the cost matrix""")

    # missing melanoma is worse than a false alarm
    COST_FN = 10.0
    COST_FP = 1.0

    bayes.print_cost_analysis(COST_FN, COST_FP)
    cost_threshold = COST_FP / (COST_FP + COST_FN)
    map_threshold = 0.5

    preds_train = bayes.threshold_with_costs(probs_train, COST_FN, COST_FP)
    preds_selection = bayes.threshold_with_costs(probs_selection, COST_FN, COST_FP)
    preds_development = bayes.threshold_with_costs(probs_development, COST_FN, COST_FP)

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
    ece_by_split = {
        "train": ece_train,
        "selection": ece_selection,
        "development": ece_development,
    }
    reporting.save_metrics_tables(
        metrics_by_split, ece_by_split, RESULTS_PATH / "tables"
    )

    metrics_summary = {
        "pipeline": "Classical Bayesian HSV + GMM skin lesion triage",
        "task": "binary melanoma-vs-benign classification",
        "positive_class": "melanoma",
        "feature_representation": {
            "name": "HSV per-channel histograms",
            "h_bins": 64,
            "s_bins": 32,
            "v_bins": 32,
            "dimension": int(X_train.shape[1]),
            "image_size": [256, 256],
            "standardization": "train statistics only",
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
        "bic_selection": reporting.selection_summary(selection_info),
        "metrics": metrics_by_split,
        "expected_calibration_error": ece_by_split,
        "notes": [
            "GMM component counts are selected with BIC on the training set.",
            "Selection and development features are standardized with training statistics only.",
            "The shared manifest reserves distinct training, selection, calibration and development groups.",
        ],
    }
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

    for class_label in [0, 1]:
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
                "score": np.where(predictions == 1, probabilities, 1 - probabilities),
                "model_name": "Bayesian_GMM_HSV",
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

    print("""
11. save model""")

    model_info = {
        "split_hash": split_report["split_hash"],
        "cohort_hash": split_report["cohort_hash"],
        "protocol_version": split_report["protocol_version"],
        "models": models,
        "selection_info": selection_info,
        "train_mean": train_mean,
        "train_std": train_std,
        "class_priors": class_priors,
        "threshold": cost_threshold,
        "map_threshold": map_threshold,
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
    }

    with open(MODELS_PATH / "bayesian_gmm_model.pkl", "wb") as f:
        pickle.dump(model_info, f)

    print(f"  Saved model to {MODELS_PATH / 'bayesian_gmm_model.pkl'}")

    print(f"""
done
results: {RESULTS_PATH}
model: {MODELS_PATH}""")


if __name__ == "__main__":
    main()
