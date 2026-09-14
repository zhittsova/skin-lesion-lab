"""Bayes rule and thresholds."""

import numpy as np


def compute_class_priors(y: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(y, return_counts=True)
    priors = np.zeros(2)
    for label, count in zip(unique, counts):
        priors[label] = count / len(y)
    return priors


def compute_posterior_probabilities(
    log_likelihoods: np.ndarray, class_priors: np.ndarray
) -> np.ndarray:
    """Bayes rule, done in log space."""
    log_posteriors = log_likelihoods + np.log(class_priors + 1e-10)

    log_posteriors = log_posteriors - log_posteriors.max(axis=1, keepdims=True)

    posteriors = np.exp(log_posteriors)

    posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)

    return posteriors


def get_melanoma_probability(posteriors: np.ndarray) -> np.ndarray:
    return posteriors[:, 1]


def threshold_with_costs(
    probabilities: np.ndarray, cost_fn: float = 1.0, cost_fp: float = 1.0
) -> np.ndarray:
    threshold = cost_fp / (cost_fp + cost_fn)

    predictions = (probabilities >= threshold).astype(int)

    return predictions


def get_threshold_info(cost_fn: float, cost_fp: float) -> dict[str, float | str]:
    threshold = cost_fp / (cost_fp + cost_fn)
    cost_ratio = cost_fn / cost_fp

    return {
        "threshold": threshold,
        "cost_ratio_FN_to_FP": cost_ratio,
        "interpretation": f"Cost of missing melanoma is {cost_ratio:.1f}x worse than false alarm",
    }


def calibrate_probabilities(
    probabilities: np.ndarray, temperature: float = 1.0
) -> np.ndarray:
    p = np.clip(probabilities, 1e-10, 1 - 1e-10)

    log_odds = np.log(p / (1 - p)) / temperature
    calibrated = 1 / (1 + np.exp(-log_odds))

    return calibrated


def compute_predicted_scores(
    posteriors: np.ndarray, calibration_temperature: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = get_melanoma_probability(posteriors)

    if calibration_temperature is not None:
        probabilities = calibrate_probabilities(probabilities, calibration_temperature)

    predictions = (probabilities >= 0.5).astype(int)
    scores = np.where(predictions == 1, probabilities, 1 - probabilities)

    return predictions, scores


def print_cost_analysis(cost_fn: float, cost_fp: float) -> None:
    threshold = cost_fp / (cost_fp + cost_fn)
    cost_ratio = cost_fn / cost_fp

    print(
        f"\ncosts: FN={cost_fn:.1f}, FP={cost_fp:.1f}, ratio={cost_ratio:.1f}x; "
        f"threshold={threshold:.4f}"
    )
