"""Uncertainty summaries for Monte Carlo dropout predictions."""

import numpy as np


def binary_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Entropy of Bernoulli probabilities in nats."""
    p = np.clip(probabilities, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def summarize_mc_dropout_probabilities(
    stochastic_probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    """Summarize T stochastic melanoma-probability predictions per image.

    Expected shape is (n_passes, n_samples). Each row is one dropout-active
    inference pass. The output can be joined with image ids and labels for a
    report table.
    """
    probs = np.asarray(stochastic_probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("Expected probabilities with shape (n_passes, n_samples)")
    if probs.shape[0] < 2:
        raise ValueError("At least two stochastic passes are needed for uncertainty")
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("Probabilities must be in [0, 1]")

    mean_prob = probs.mean(axis=0)
    variance = probs.var(axis=0, ddof=1)
    predictive_entropy = binary_entropy(mean_prob)
    expected_entropy = binary_entropy(probs).mean(axis=0)

    return {
        "mean_prob_melanoma": mean_prob,
        "variance": variance,
        "predictive_entropy": predictive_entropy,
        "mutual_information": predictive_entropy - expected_entropy,
    }


def uncertainty_flags(
    mean_probabilities: np.ndarray,
    variances: np.ndarray,
    threshold: float,
    uncertainty_quantile: float = 0.9,
    margin: float = 0.05,
) -> np.ndarray:
    """Flag borderline or high-variance cases for clinician review."""
    if not 0 <= uncertainty_quantile <= 1:
        raise ValueError("uncertainty_quantile must be in [0, 1]")

    variance_cutoff = np.quantile(variances, uncertainty_quantile)
    near_threshold = np.abs(mean_probabilities - threshold) <= margin
    high_variance = variances >= variance_cutoff
    return near_threshold | high_variance
