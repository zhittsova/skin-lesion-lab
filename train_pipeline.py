"""
Small script for the classical pipeline.

1. Load HAM10000 images and convert to HSV colour space.
2. Compute per-channel histograms (H, S, V) and concatenate into one feature vector.
3. Split data into train, validation, and test sets with class stratication.
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
    split_indices = splitting.split_dataset(
        lesion_ids, labels, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42
    )

    splitting.print_split_summary(labels, split_indices, lesion_ids)
    split_report = splitting.get_split_report(labels, split_indices, lesion_ids)
    reporting.save_json(split_report, RESULTS_PATH / "split_summary.json")
    reporting.save_split_tables(split_report, RESULTS_PATH / "tables")

    print("\n3. load images and make HSV histograms")

    def load_and_extract_features(indices, desc=""):
        hsv_images = []
        valid_indices = []

        for idx in tqdm(indices, desc=desc):
            img_id = image_ids[idx]
            img_path = IMAGES_DIR / f"{img_id}.jpg"

            try:
                hsv_img = data.load_image_hsv(str(img_path), target_size=(256, 256))
                hsv_images.append(hsv_img)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error loading {img_id}: {e}")

        X = features.compute_hsv_histograms_batch(hsv_images)

        return X, valid_indices

    X_train_raw, train_valid_idx = load_and_extract_features(
        split_indices["train"], "Training set"
    )
    X_val_raw, val_valid_idx = load_and_extract_features(
        split_indices["val"], "Validation set"
    )
    X_test_raw, test_valid_idx = load_and_extract_features(
        split_indices["test"], "Test set"
    )

    # using train stats here, so val/test dont leak into the scaling
    X_train, train_mean, train_std = features.standardize_features(X_train_raw)
    X_val = features.apply_standardization(X_val_raw, train_mean, train_std)
    X_test = features.apply_standardization(X_test_raw, train_mean, train_std)

    y_train = labels[train_valid_idx]
    y_val = labels[val_valid_idx]
    y_test = labels[test_valid_idx]

    print(f"""
feature extraction done
feature dimension: {X_train.shape[1]}
train: {X_train.shape[0]} samples
val: {X_val.shape[0]} samples
test: {X_test.shape[0]} samples""")

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

    log_likelihoods_val = gmm.compute_class_likelihoods(models, X_val)
    posteriors_val = bayes.compute_posterior_probabilities(
        log_likelihoods_val, class_priors
    )
    probs_val = bayes.get_melanoma_probability(posteriors_val)

    log_likelihoods_test = gmm.compute_class_likelihoods(models, X_test)
    posteriors_test = bayes.compute_posterior_probabilities(
        log_likelihoods_test, class_priors
    )
    probs_test = bayes.get_melanoma_probability(posteriors_test)

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
    preds_val = bayes.threshold_with_costs(probs_val, COST_FN, COST_FP)
    preds_test = bayes.threshold_with_costs(probs_test, COST_FN, COST_FP)

    preds_train_map = (probs_train >= map_threshold).astype(int)
    preds_val_map = (probs_val >= map_threshold).astype(int)
    preds_test_map = (probs_test >= map_threshold).astype(int)

    print("\n7. evaluate")

    metrics_train = evaluation.compute_classification_metrics(
        y_train, preds_train, probs_train
    )
    metrics_val = evaluation.compute_classification_metrics(y_val, preds_val, probs_val)
    metrics_test = evaluation.compute_classification_metrics(
        y_test, preds_test, probs_test
    )
    metrics_train = evaluation.add_average_cost(
        metrics_train, len(y_train), COST_FN, COST_FP
    )
    metrics_val = evaluation.add_average_cost(metrics_val, len(y_val), COST_FN, COST_FP)
    metrics_test = evaluation.add_average_cost(
        metrics_test, len(y_test), COST_FN, COST_FP
    )

    metrics_train_map = evaluation.compute_classification_metrics(
        y_train, preds_train_map, probs_train
    )
    metrics_val_map = evaluation.compute_classification_metrics(
        y_val, preds_val_map, probs_val
    )
    metrics_test_map = evaluation.compute_classification_metrics(
        y_test, preds_test_map, probs_test
    )
    metrics_train_map = evaluation.add_average_cost(
        metrics_train_map, len(y_train), COST_FN, COST_FP
    )
    metrics_val_map = evaluation.add_average_cost(
        metrics_val_map, len(y_val), COST_FN, COST_FP
    )
    metrics_test_map = evaluation.add_average_cost(
        metrics_test_map, len(y_test), COST_FN, COST_FP
    )

    evaluation.print_evaluation_summary(
        metrics_train, f"Training (Cost Threshold {cost_threshold:.4f})"
    )
    evaluation.print_evaluation_summary(
        metrics_val, f"Validation (Cost Threshold {cost_threshold:.4f})"
    )
    evaluation.print_evaluation_summary(
        metrics_test, f"Test (Cost Threshold {cost_threshold:.4f})"
    )

    def print_threshold_comparison(split_name, cost_metrics, map_metrics):
        msg = f"""
{split_name} threshold compare
cost threshold {cost_threshold:.4f}: recall={cost_metrics["recall"]:.3f}, specificity={cost_metrics["specificity"]:.3f}, FP={cost_metrics["fp"]}, FN={cost_metrics["fn"]}
MAP threshold  {map_threshold:.4f}: recall={map_metrics["recall"]:.3f}, specificity={map_metrics["specificity"]:.3f}, FP={map_metrics["fp"]}, FN={map_metrics["fn"]}
"""
        print(msg)

    print_threshold_comparison("Training", metrics_train, metrics_train_map)
    print_threshold_comparison("Validation", metrics_val, metrics_val_map)
    print_threshold_comparison("Test", metrics_test, metrics_test_map)

    print("\n8. calibration")

    ece_train = evaluation.compute_calibration_error(y_train, probs_train)
    ece_val = evaluation.compute_calibration_error(y_val, probs_val)
    ece_test = evaluation.compute_calibration_error(y_test, probs_test)

    print(f"""
Expected Calibration Error (ECE)
train: {ece_train:.4f}
val: {ece_val:.4f}
test: {ece_test:.4f}""")

    metrics_by_split = {
        "train": {"cost_threshold": metrics_train, "map_threshold": metrics_train_map},
        "val": {"cost_threshold": metrics_val, "map_threshold": metrics_val_map},
        "test": {"cost_threshold": metrics_test, "map_threshold": metrics_test_map},
    }
    ece_by_split = {"train": ece_train, "val": ece_val, "test": ece_test}
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
            "Validation and test features are standardized with training statistics only.",
            "Lesion IDs are kept disjoint across train, validation, and test splits.",
        ],
    }
    reporting.save_json(metrics_summary, RESULTS_PATH / "metrics_summary.json")

    print("\n9. make plots")

    fpr_test, tpr_test, auc_test = evaluation.compute_roc_curve(y_test, probs_test)
    precision_test, recall_test, pr_auc_test = evaluation.compute_pr_curve(
        y_test, probs_test
    )

    plots.plot_roc_curve(
        fpr_test,
        tpr_test,
        auc_test,
        save_path=str(RESULTS_PATH / "figures" / "roc_curve.png"),
    )
    plots.plot_pr_curve(
        precision_test,
        recall_test,
        pr_auc_test,
        baseline=float(np.mean(y_test)),
        save_path=str(RESULTS_PATH / "figures" / "pr_curve.png"),
    )

    cm_test = evaluation.compute_confusion_matrix(y_test, preds_test)
    plots.plot_confusion_matrix(
        cm_test, save_path=str(RESULTS_PATH / "figures" / "confusion_matrix.png")
    )
    plots.plot_threshold_comparison(
        {"cost threshold": metrics_test, "MAP threshold": metrics_test_map},
        save_path=str(RESULTS_PATH / "figures" / "threshold_comparison.png"),
    )

    mean_probs, freqs, bin_sizes = evaluation.compute_calibration_curve(
        y_test, probs_test
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
        image_ids[val_valid_idx], preds_val, preds_val_map, probs_val, y_val, "val"
    )
    save_predictions(
        image_ids[test_valid_idx],
        preds_test,
        preds_test_map,
        probs_test,
        y_test,
        "test",
    )

    print("""
11. save model""")

    model_info = {
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
