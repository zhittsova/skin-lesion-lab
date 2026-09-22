"""Calibration-only probability fitting and locked decision/referral policies."""

from __future__ import annotations

import hashlib
import json
import warnings
from fractions import Fraction

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src import splitting, uncertainty


def expit(values):
    values = np.asarray(values, dtype=float)
    tail = np.exp(-np.abs(values))
    return np.where(values >= 0, 1 / (1 + tail), tail / (1 + tail))


def probabilities(values, *, ndim=1):
    array = np.asarray(values, dtype=float)
    if (
        array.ndim != ndim
        or not array.size
        or not np.isfinite(array).all()
        or np.any((array < 0) | (array > 1))
    ):
        raise ValueError("probabilities must be nonempty, finite and in [0, 1]")
    return array


def targets(values, count):
    array = np.asarray(values)
    if count < 1 or array.shape != (count,) or not np.isin(array, [0, 1]).all():
        raise ValueError("targets must be aligned binary labels")
    return array.astype(int)


def cost_threshold(cost_fn, cost_fp):
    costs = np.asarray([cost_fn, cost_fp], dtype=float)
    if not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError("costs must be finite and positive")
    # Rescale before addition to avoid overflow for finite large costs.
    costs = costs / costs.max()
    threshold = float(costs[1] / costs.sum())
    with np.errstate(over="ignore", divide="ignore"):
        ratio = float(cost_fn) / np.float64(cost_fp)
    if not 0 < threshold < 1 or not np.isfinite(ratio) or ratio == 0:
        raise ValueError("cost ratio cannot be represented safely")
    return threshold


def correct_weighted_scores(scores, weight=1.0):
    """Undo idealized positive BCE weighting; this does not fit calibration."""
    scores = probabilities(scores, ndim=np.asarray(scores).ndim)
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("weight must be finite and positive")
    # q / (q + w(1-q)); avoid overflow and cancellation near endpoints.
    if weight >= 1:
        return (scores / weight) / ((scores / weight) + (1 - scores))
    return scores / (scores + weight * (1 - scores))


def _logit(scores):
    scores = np.clip(scores, 1e-10, 1 - 1e-10)
    return np.log(scores) - np.log1p(-scores)


def select_cost_threshold(labels, scores, cost_fn=10.0, cost_fp=1.0):
    scores = probabilities(scores)
    labels = targets(labels, len(scores))
    cost_threshold(cost_fn, cost_fp)
    rows = []
    exact_costs = []
    fn_cost, fp_cost = Fraction(float(cost_fn)), Fraction(float(cost_fp))
    # >= is the saved comparison; nextafter permits all-negative at score 1.
    for threshold in np.unique(np.r_[0.0, scores, np.nextafter(scores.max(), np.inf)]):
        predicted = scores >= threshold
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum(~predicted & (labels == 1)))
        tp = int(np.sum(predicted & (labels == 1)))
        tn = int(np.sum(~predicted & (labels == 0)))
        total_cost = fp_cost * fp + fn_cost * fn
        exact_costs.append(total_cost)
        rows.append(
            {
                "threshold": float(threshold),
                "average_cost": float(total_cost / len(labels)),
                "recall": tp / (tp + fn) if tp + fn else None,
                "specificity": tn / (tn + fp) if tn + fp else None,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )
    best_index = min(
        range(len(rows)),
        key=lambda i: (
            exact_costs[i],
            -(rows[i]["recall"] or 0),
            -(rows[i]["specificity"] or 0),
            rows[i]["threshold"],
        ),
    )
    return rows[best_index]["threshold"], rows


def _corrected_mean(scores, weight):
    array = np.asarray(scores)
    if array.ndim == 2:
        uncertainty.summarize_mc_dropout_probabilities(array)
        return correct_weighted_scores(array, weight).mean(axis=0)
    return correct_weighted_scores(probabilities(array), weight)


def fit_data_digest(labels, corrected, image_ids):
    payload = {
        "labels": np.asarray(labels, dtype=int).tolist(),
        "corrected_scores": np.asarray(corrected, dtype=float).tolist(),
        "image_ids": list(image_ids),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def fit_policy(
    labels,
    scores,
    *,
    image_ids,
    split_hash,
    weight=1.0,
    variances=None,
    cost_fn=10.0,
    cost_fp=1.0,
    schema_version=1,
    split_manifest=None,
):
    if type(schema_version) is not int or schema_version not in (1, 2):
        raise ValueError("unsupported decision policy version")
    if schema_version == 2:
        return _fit_log_policy(
            labels,
            scores,
            image_ids=image_ids,
            split_hash=split_hash,
            split_manifest=split_manifest,
            weight=weight,
            variances=variances,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
        )
    corrected = _corrected_mean(scores, weight)
    labels = targets(labels, len(corrected))
    ids = list(image_ids)
    if len(np.unique(labels)) != 2:
        raise ValueError("calibration requires both classes")
    if (
        len(ids) != len(labels)
        or any(not isinstance(i, str) or not i for i in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("calibration requires unique aligned image IDs")
    if not isinstance(split_hash, str) or not split_hash:
        raise ValueError("calibration requires split identity")
    formula = cost_threshold(cost_fn, cost_fp)
    # Fixed regularization and clipping; neither is selected on the holdout.
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(_logit(corrected).reshape(-1, 1), labels)
    slope, intercept = float(model.coef_[0, 0]), float(model.intercept_[0])
    fitted = expit(slope * _logit(corrected) + intercept)
    threshold, _ = select_cost_threshold(labels, fitted, cost_fn, cost_fp)
    quantiles = [0.0, 0.5, 0.8, 0.9, 1.0]
    cutoffs = None
    if variances is not None:
        values = uncertainty.validate_variances(variances, len(corrected))
        cutoffs = [float(np.quantile(values, q)) for q in quantiles]
    return {
        "schema_version": 1,
        "fit": {
            "role": "calibration",
            "image_ids": ids,
            "split_hash": split_hash,
            "data_sha256": fit_data_digest(labels, corrected, ids),
        },
        "score_transform": {
            "positive_weight": float(weight),
            "aggregation": "correct_each_pass_then_mean",
        },
        "calibrator": {
            "method": "regularized_sigmoid",
            "C": 1.0,
            "logit_clip": 1e-10,
            "solver": "lbfgs",
            "max_iter": 1000,
            "tol": 1e-10,
            "penalty": "l2_slope_only",
            "class_weight": None,
            "slope": slope,
            "intercept": intercept,
        },
        "decision": {
            "threshold": threshold,
            "formula_threshold": formula,
            "comparison": ">=",
            "cost_fn": float(cost_fn),
            "cost_fp": float(cost_fp),
            "tie_break": "recall_desc_specificity_desc_threshold_asc",
        },
        "referral": {
            "margin": 0.05,
            "variance_source": "raw_mc_sample_variance",
            "ddof": 1,
            "quantile_method": "linear",
            "mc_passes": int(np.asarray(scores).shape[0])
            if np.asarray(scores).ndim == 2
            else None,
            "quantile": 0.9,
            "variance_cutoff": None if cutoffs is None else cutoffs[3],
            "grid_quantiles": quantiles if cutoffs is not None else [],
            "grid_cutoffs": cutoffs or [],
            "comparison": ">=",
        },
    }


def apply_policy(policy, scores, *, variances=None):
    version = policy.get("schema_version")
    if type(version) is not int or version not in (1, 2):
        raise ValueError("unsupported decision policy version")
    if version == 2:
        return _apply_log_policy(policy, scores, variances=variances)
    passes = policy["referral"]["mc_passes"]
    if passes is not None and (
        np.asarray(scores).ndim != 2 or np.asarray(scores).shape[0] != passes
    ):
        raise ValueError("MC pass count differs from fitted referral policy")
    corrected = _corrected_mean(scores, policy["score_transform"]["positive_weight"])
    fit = policy["calibrator"]
    coefficients = np.array([fit["slope"], fit["intercept"]], dtype=float)
    if not np.isfinite(coefficients).all():
        raise ValueError("invalid fitted calibrator")
    fitted = expit(fit["slope"] * _logit(corrected) + fit["intercept"])
    threshold = policy["decision"]["threshold"]
    referral = policy["referral"]
    if referral["variance_cutoff"] is None:
        if variances is not None:
            raise ValueError("variance referral was not fitted")
        flags = uncertainty.uncertainty_flags(
            fitted, None, threshold, variance_cutoff=None, margin=referral["margin"]
        )
    else:
        if variances is None:
            raise ValueError("saved variance referral requires variances")
        flags = uncertainty.uncertainty_flags(
            fitted,
            variances,
            threshold,
            variance_cutoff=referral["variance_cutoff"],
            margin=referral["margin"],
        )
    return {
        "corrected_score": corrected,
        "calibrated_probability": fitted,
        "prediction": (fitted >= threshold).astype(int),
        "review_recommended": flags,
    }


def probability_report(labels, scores, n_bins=10, *, binning_version=2):
    scores = probabilities(scores)
    labels = targets(labels, len(scores))
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.minimum(np.searchsorted(edges, scores, side="right") - 1, n_bins - 1)
    if type(binning_version) is not int or binning_version not in (1, 2):
        raise ValueError("unsupported binning version")
    if binning_version == 1:
        index = np.minimum((scores * n_bins).astype(int), n_bins - 1)
    bins = []
    for i in range(n_bins):
        mask = index == i
        count = int(mask.sum())
        bins.append(
            {
                "lower": i / n_bins if binning_version == 1 else float(edges[i]),
                "upper": (i + 1) / n_bins
                if binning_version == 1
                else float(edges[i + 1]),
                "count": count,
                "mean_probability": float(scores[mask].mean()) if count else None,
                "positive_fraction": float(labels[mask].mean()) if count else None,
            }
        )
    # Explicit finite numerical convention for endpoint log scores.
    clipped = np.clip(scores, 1e-15, 1 - 1e-15)
    return {
        "brier_score": float(np.mean((scores - labels) ** 2)),
        "log_loss": float(
            -np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
        ),
        "log_loss_clip": 1e-15,
        "reliability_bins": bins,
    }


def _selective(labels, predictions, flags, decision):
    keep = ~flags
    count = int(keep.sum())
    fp = int(np.sum(keep & (labels == 0) & (predictions == 1)))
    fn = int(np.sum(keep & (labels == 1) & (predictions == 0)))
    return {
        "total_count": len(labels),
        "referred_count": int(flags.sum()),
        "retained_count": count,
        "coverage": count / len(labels),
        "retained_error": float(np.mean(predictions[keep] != labels[keep]))
        if count
        else None,
        "retained_average_cost": decision["cost_fp"] * (fp / count)
        + decision["cost_fn"] * (fn / count)
        if count
        else None,
        "retained_fp": fp,
        "retained_fn": fn,
        "retained_positive_count": int(np.sum(labels[keep] == 1)),
        "retained_negative_count": int(np.sum(labels[keep] == 0)),
    }


def policy_report(policy, labels, scores, *, variances=None, metrics_version=2):
    applied = apply_policy(policy, scores, variances=variances)
    labels = targets(labels, len(applied["prediction"]))
    raw = np.asarray(scores)
    if policy["schema_version"] == 2:
        raw = expit(finite_scores(raw))
    elif raw.ndim == 2:
        raw = raw.mean(axis=0)
    result = {
        name: probability_report(labels, values, binning_version=metrics_version)
        for name, values in {
            "raw_score": raw,
            "corrected_score": applied["corrected_score"],
            "fitted_probability": applied["calibrated_probability"],
        }.items()
    }
    result["selective"] = _selective(
        labels, applied["prediction"], applied["review_recommended"], policy["decision"]
    )
    result["risk_coverage"] = []
    for q, cutoff in zip(
        policy["referral"]["grid_quantiles"], policy["referral"]["grid_cutoffs"]
    ):
        flags = uncertainty.uncertainty_flags(
            applied["calibrated_probability"],
            variances,
            policy["decision"]["threshold"],
            variance_cutoff=cutoff,
            margin=policy["referral"]["margin"],
        )
        result["risk_coverage"].append(
            {
                "calibration_quantile": q,
                "variance_cutoff": cutoff,
                **_selective(labels, applied["prediction"], flags, policy["decision"]),
            }
        )
    return result


def finite_scores(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("log scores must be a nonempty finite vector")
    return array


def _fit_log_policy(
    labels,
    scores,
    *,
    image_ids,
    split_hash,
    split_manifest,
    weight,
    variances,
    cost_fn,
    cost_fp,
):
    scores = finite_scores(scores)
    labels = targets(labels, len(scores))
    ids = list(image_ids)
    if weight != 1 or variances is not None:
        raise ValueError("log score policy requires unweighted deterministic scores")
    if (
        not isinstance(split_manifest, dict)
        or split_hash != split_manifest.get("split_hash")
        or splitting.canonical_hash(
            {k: v for k, v in split_manifest.items() if k != "split_hash"}
        )
        != split_hash
        or ids != split_manifest["partitions"]["calibration"]
        or len(ids) != len(labels)
        or len(set(ids)) != len(ids)
        or any(not isinstance(i, str) or not i for i in ids)
    ):
        raise ValueError(
            "log score fit requires the exact calibration manifest identity"
        )
    parts = split_manifest["partitions"]
    partition_ids = [i for members in parts.values() for i in members]
    row_ids = [r["image_id"] for r in split_manifest["rows"]]
    if (
        split_manifest.get("purpose") != "development"
        or set(parts) != set(splitting.ROLES)
        or len(partition_ids) != len(set(partition_ids))
        or len(row_ids) != len(set(row_ids))
        or set(partition_ids) != set(row_ids)
    ):
        raise ValueError(
            "calibration manifest partitions must be disjoint and complete"
        )
    by_id = {row["image_id"]: row["target"] for row in split_manifest["rows"]}
    if (
        not np.array_equal(labels, [by_id[i] for i in ids])
        or len(np.unique(labels)) != 2
    ):
        raise ValueError(
            "log score fit requires both calibration classes and aligned labels"
        )
    formula_threshold = cost_threshold(cost_fn, cost_fp)
    scale = max(1.0, float(np.abs(scores).max()))
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit((scores / scale).reshape(-1, 1), labels)
    slope, intercept = float(model.coef_[0, 0]), float(model.intercept_[0])
    fitted = expit(slope * (scores / scale) + intercept)
    threshold, _ = select_cost_threshold(labels, fitted, cost_fn, cost_fp)
    return {
        "schema_version": 2,
        "protocol": "gmm-log-odds-sigmoid-v2",
        "fit": {
            "role": "calibration",
            "image_ids": ids,
            "split_hash": split_hash,
            "data_sha256": fit_data_digest(labels, scores, ids),
        },
        "score_transform": {
            "input_kind": "finite_log_odds",
            "positive_weight": 1.0,
            "aggregation": "none",
            "scale": scale,
            "scale_rule": "calibration_max_abs_at_least_one",
        },
        "calibrator": {
            "method": "regularized_sigmoid",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "tol": 1e-10,
            "penalty": "l2_slope_only",
            "class_weight": None,
            "slope": slope,
            "intercept": intercept,
        },
        "ranking": "sign_slope_times_input",
        "decision": {
            "threshold": threshold,
            "formula_threshold": formula_threshold,
            "comparison": ">=",
            "cost_fn": float(cost_fn),
            "cost_fp": float(cost_fp),
            "tie_break": "recall_desc_specificity_desc_threshold_asc",
        },
        "referral": {
            "margin": 0.05,
            "variance_cutoff": None,
            "mc_passes": None,
            "grid_quantiles": [],
            "grid_cutoffs": [],
            "comparison": ">=",
        },
    }


def _apply_log_policy(policy, scores, *, variances):
    scores = finite_scores(scores)
    transform = policy["score_transform"]
    scale = transform["scale"]
    if (
        policy.get("protocol") != "gmm-log-odds-sigmoid-v2"
        or transform["input_kind"] != "finite_log_odds"
        or transform["aggregation"] != "none"
        or transform["positive_weight"] != 1
        or transform["scale_rule"] != "calibration_max_abs_at_least_one"
        or not np.isfinite(scale)
        or scale < 1
        or policy.get("ranking") != "sign_slope_times_input"
        or variances is not None
        or policy["referral"]["mc_passes"] is not None
        or policy["referral"]["variance_cutoff"] is not None
    ):
        raise ValueError("unsupported log score policy contract")
    fit = policy["calibrator"]
    fixed = {
        "method": "regularized_sigmoid",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "tol": 1e-10,
        "penalty": "l2_slope_only",
        "class_weight": None,
    }
    if any(fit.get(k) != v for k, v in fixed.items()):
        raise ValueError("unsupported log score calibrator contract")
    slope, intercept = fit["slope"], fit["intercept"]
    if not np.isfinite([slope, intercept]).all():
        raise ValueError("invalid fitted calibrator")
    with np.errstate(over="ignore", invalid="ignore"):
        affine = slope * (scores / scale) + intercept
    if not np.isfinite(affine).all():
        raise ValueError("calibrated log score cannot be represented")
    fitted = expit(affine)
    threshold = policy["decision"]["threshold"]
    if (
        policy["decision"]["comparison"] != ">="
        or not np.isfinite(threshold)
        or not 0 <= threshold <= np.nextafter(1.0, np.inf)
    ):
        raise ValueError("invalid decision policy threshold")
    return {
        "corrected_score": expit(scores),
        "calibrated_probability": fitted,
        "ranking_score": np.sign(slope) * scores,
        "prediction": (fitted >= threshold).astype(int),
        "review_recommended": uncertainty.uncertainty_flags(
            fitted,
            None,
            threshold,
            variance_cutoff=None,
            margin=policy["referral"]["margin"],
        ),
    }
