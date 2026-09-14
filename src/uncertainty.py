"""MC dropout summaries in nats and referral with previously fitted cutoffs."""

import numpy as np


def _probabilities(values):
    p = np.asarray(values, dtype=float)
    if not p.size or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be nonempty, finite and in [0, 1]")
    return p


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Bernoulli entropy in nats, with exact zero at both endpoints."""
    p = _probabilities(probabilities)
    log_p = np.zeros_like(p)
    log_q = np.zeros_like(p)
    np.log(p, out=log_p, where=p > 0)
    np.log1p(-p, out=log_q, where=p < 1)
    return -p * log_p - (1 - p) * log_q


def summarize_mc_dropout_probabilities(
    stochastic_probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    probs = _probabilities(stochastic_probabilities)
    if probs.ndim != 2 or probs.shape[0] < 2:
        raise ValueError("expected at least two passes with shape (passes, samples)")
    mean_prob = probs.mean(axis=0)
    predictive_entropy = binary_entropy(mean_prob)
    expected_entropy = binary_entropy(probs).mean(axis=0)
    return {
        "mean_prob_melanoma": mean_prob,
        "variance": probs.var(axis=0, ddof=1),
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": np.maximum(0, predictive_entropy - expected_entropy),
    }


def validate_variances(variances, count):
    values = np.asarray(variances, dtype=float)
    # Unbiased Bernoulli sample variance can reach .5 with two draws.
    if (
        values.shape != (count,)
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 0.5))
    ):
        raise ValueError("variances must be aligned, finite and in [0, .5]")
    return values


def uncertainty_flags(
    mean_probabilities, variances, threshold, *, variance_cutoff, margin=0.05
):
    """Apply a saved cutoff; the inference batch never fits a quantile."""
    means = _probabilities(mean_probabilities)
    if means.ndim != 1:
        raise ValueError("probabilities must be a vector")
    if not np.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1.0, np.inf):
        raise ValueError("invalid decision threshold")
    if not np.isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError("margin must be in [0, 1]")
    flags = np.abs(means - threshold) <= margin
    if variance_cutoff is not None:
        if not np.isfinite(variance_cutoff) or not 0 <= variance_cutoff <= 0.5:
            raise ValueError("invalid saved variance cutoff")
        flags = flags | (validate_variances(variances, len(means)) >= variance_cutoff)
    elif variances is not None:
        raise ValueError("variances require a fitted cutoff")
    return flags
