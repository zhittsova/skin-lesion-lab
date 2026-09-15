"""Paired development estimates conditional on prespecified model seeds."""

from collections.abc import Mapping

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

SEEDS = (17, 42, 73)


def _arrays(frame):
    if frame.empty:
        raise ValueError("empty prediction cohort")
    y = frame.target.to_numpy()
    p = frame.prob_melanoma.to_numpy(dtype=float)
    d = frame.prediction.to_numpy()
    if (
        not np.isin(y, [0, 1]).all()
        or set(y) != {0, 1}
        or not np.isin(d, [0, 1]).all()
        or not np.isfinite(p).all()
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError("endpoints require both classes and finite binary predictions")
    return y, p, d


def endpoints(frame):
    """Image-level endpoints; undefined precision remains null, never zero."""
    y, p, d = _arrays(frame)
    tp, tn = np.sum((y == 1) & (d == 1)), np.sum((y == 0) & (d == 0))
    fp, fn = np.sum((y == 0) & (d == 1)), np.sum((y == 1) & (d == 0))
    precision, recall, _ = precision_recall_curve(y, p)
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "pr_auc": float(auc(recall, precision)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        "average_cost": float((10 * fn + fp) / len(y)),
    }


def group_draws(frame, *, draws=2000, seed=2026):
    """Sample whole components within class, preserving image multiplicities."""
    _arrays(frame)
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    for column in ("image_id", "group_id"):
        if not frame[column].map(lambda s: isinstance(s, str) and bool(s)).all():
            raise ValueError("image and group IDs must be nonempty strings")
    if frame.image_id.duplicated().any():
        raise ValueError("duplicate image IDs")
    labels = frame.groupby("group_id", sort=True).target.agg(["min", "max"])
    if (labels["min"] != labels["max"]).any():
        raise ValueError("mixed-label components are unsupported")
    strata = [labels.index[labels["min"] == y].to_numpy() for y in (0, 1)]
    members = {g: np.flatnonzero(frame.group_id.to_numpy() == g) for g in labels.index}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        groups = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        yield np.concatenate([members[g] for g in groups])


def _mean(values):
    return None if any(v is None for v in values) else float(np.mean(values))


def _summary(seed_values, draws):
    invalid = sum(v is None for v in draws)
    point = _mean(seed_values)
    return {
        "estimate": point,
        "interval": None
        if invalid or point is None
        else np.quantile(draws, [0.025, 0.975]).tolist(),
        "undefined_draws": invalid,
        "seed_values": list(seed_values),
        "seed_range": None if point is None else [min(seed_values), max(seed_values)],
        "seed_sd": None if point is None else float(np.std(seed_values, ddof=1)),
    }


def paired_report(models: Mapping, *, reference: str, draws=2000, seed=2026):
    """Average seed metrics on shared class-stratified group draws.

    Inputs map each named, already selected model stratum to all three seeds.
    Callers loading files must validate run provenance before passing frames.
    """
    if not models or reference not in models:
        raise ValueError("a reference model and nonempty comparison are required")
    aligned = {}
    identity = None
    for model in sorted(models):
        if set(models[model]) != set(SEEDS):
            raise ValueError("every model requires exactly seeds 17, 42, 73")
        aligned[model] = {}
        for s in SEEDS:
            f = models[model][s].sort_values("image_id").reset_index(drop=True)
            _arrays(f)
            if f.image_id.duplicated().any():
                raise ValueError("duplicate image IDs")
            ids = f[["image_id", "group_id", "target"]]
            if identity is None:
                identity = ids
            elif not identity.equals(ids):
                raise ValueError(
                    "models and seeds require identical images, groups and labels"
                )
            aligned[model][s] = f
    cohort = aligned[reference][SEEDS[0]]
    points = {
        m: {s: endpoints(f) for s, f in runs.items()} for m, runs in aligned.items()
    }
    metrics = tuple(points[reference][SEEDS[0]])
    samples = {m: {k: [] for k in metrics} for m in aligned}
    contrasts = {m: {k: [] for k in metrics} for m in aligned if m != reference}
    for indices in group_draws(cohort, draws=draws, seed=seed):
        values = {
            m: {s: endpoints(f.iloc[indices]) for s, f in runs.items()}
            for m, runs in aligned.items()
        }
        for m in aligned:
            for k in metrics:
                samples[m][k].append(_mean([values[m][s][k] for s in SEEDS]))
                if m != reference:
                    differences = [
                        None
                        if values[m][s][k] is None or values[reference][s][k] is None
                        else values[m][s][k] - values[reference][s][k]
                        for s in SEEDS
                    ]
                    contrasts[m][k].append(_mean(differences))
    return {
        "purpose": "development",
        "log_loss_clip": 1e-15,
        "seeds": list(SEEDS),
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "unit": "manifest_connected_component",
            "method": "within_class_paired_percentile",
            "confidence": 0.95,
            "undefined_policy": "null_interval_if_any_draw_undefined",
        },
        "counts": {
            "images": len(cohort),
            "groups": cohort.group_id.nunique(),
            "classes": {str(y): int(sum(cohort.target == y)) for y in (0, 1)},
        },
        "reference": reference,
        "models": {
            m: {
                k: _summary([points[m][s][k] for s in SEEDS], samples[m][k])
                for k in metrics
            }
            for m in aligned
        },
        "differences": {
            m: {
                k: _summary(
                    [
                        None
                        if points[m][s][k] is None or points[reference][s][k] is None
                        else points[m][s][k] - points[reference][s][k]
                        for s in SEEDS
                    ],
                    contrasts[m][k],
                )
                for k in metrics
            }
            for m in contrasts
        },
    }


def planned_runs():
    """The nine classical runs and eighteen two-candidate deep searches."""
    jobs = []
    for seed in SEEDS:
        for model in ("prevalence", "logistic", "gmm"):
            jobs.append(
                {
                    "run_id": f"{model}-{seed}",
                    "stratum": model,
                    "seed": seed,
                    "pipeline": "classical",
                    "candidate_count": 1,
                    "config": {
                        "model": model,
                        "seed": seed,
                        "gmm_max_components": 10,
                        "gmm_covariance_type": "full",
                        "gmm_reg_covar": 1e-4,
                        "gmm_max_iter": 300,
                    },
                }
            )
        for architecture, mode in [
            ("small_cnn", "scratch"),
            ("efficientnet_b0", "head"),
            ("efficientnet_b0", "full"),
        ]:
            for loss in ("unweighted", "pos_weight"):
                stratum = f"{'cnn' if architecture == 'small_cnn' else 'efficientnet'}_{mode}_{loss}"
                jobs.append(
                    {
                        "run_id": f"{stratum.replace('_', '-')}-{seed}",
                        "stratum": stratum,
                        "seed": seed,
                        "pipeline": "deep",
                        "candidate_count": 2,
                        "config": {
                            "architecture": architecture,
                            "seed": seed,
                            "pretrained": architecture == "efficientnet_b0",
                            "fine_tune_backbone": mode == "full",
                            "loss_strategy": loss,
                            "learning_rates": [0.001, 0.0003],
                            "learning_rate": 0.001,
                            "weight_decay": 0.0001,
                            "dropout": 0.3,
                            "image_size": 128,
                            "batch_size": 64,
                            "epochs": 20,
                            "mc_samples": 30,
                            "num_workers": 0,
                            "cost_fn": 10.0,
                            "cost_fp": 1.0,
                        },
                    }
                )
    return jobs


def training_command(job, *, python, metadata, images, manifest, runs, source, device):
    """Return argv without shell interpolation or implicit parameter defaults."""
    command = [
        str(python),
        "train_deep_pipeline.py" if job["pipeline"] == "deep" else "train_pipeline.py",
        "--metadata-path",
        str(metadata),
        "--images-dir",
        str(images),
        "--split-manifest",
        str(manifest),
        "--source",
        source,
        "--runs-dir",
        str(runs),
        "--run-id",
        job["run_id"],
    ]
    for key, value in job["config"].items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif isinstance(value, list):
            command.extend([flag, *map(str, value)])
        else:
            command.extend([flag, str(value)])
    if job["pipeline"] == "deep":
        command.extend(["--device", device])
    return command
