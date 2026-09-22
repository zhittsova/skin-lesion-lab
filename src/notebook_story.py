"""Validated notebook inputs and a small, visibly synthetic demonstration."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark import SEEDS, endpoints, paired_report

ACCEPTED_SHA256 = "0f0e898a1af36e05898692a95c9a3f2dfbb59b5c4bb8ad08c144a6e55fd529c7"
ACCEPTED_PLAN_SHA256 = (
    "14911b9a37c1437a7f4d7c22989679db71830c5e1e576b4596af3cd793905498"
)


def accepted_report(path):
    """Load the independently audited S09 development report, byte for byte."""
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != ACCEPTED_SHA256:
        raise ValueError("S09 report SHA-256 differs from the accepted report")
    report = json.loads(content)
    if (
        report["purpose"] != "development"
        or report["plan_sha256"] != ACCEPTED_PLAN_SHA256
        or report["seeds"] != list(SEEDS)
        or report["counts"]
        != {"images": 1395, "groups": 1037, "classes": {"0": 1219, "1": 176}}
        or report["registry"]["complete"] is not True
    ):
        raise ValueError("accepted S09 report contract differs")
    summary_rows(report, "roc_auc")
    return report


def summary_rows(report, metric):
    """Keep source run IDs alongside every displayed model estimate."""
    jobs = report["registry"]["jobs"]
    if len(jobs) != len(report["models"]) * len(SEEDS) or {
        job["stratum"] for job in jobs
    } != set(report["models"]):
        raise ValueError("registry jobs do not match the displayed model seeds")
    if any(
        not isinstance(job["run_id"], str) or not job["run_id"].strip() for job in jobs
    ):
        raise ValueError("registry contains an empty run ID")
    rows = []
    for model, metrics in report["models"].items():
        matches = [job for job in jobs if job["stratum"] == model]
        if (
            len(matches) != len(SEEDS)
            or {job["seed"] for job in matches} != set(SEEDS)
            or any(
                job["status"] != "completed" or not job.get("run_record_sha256")
                for job in matches
            )
        ):
            raise ValueError(f"missing or incomplete run IDs for {model}")
        value = metrics[metric]
        rows.append(
            {
                "model": model,
                "estimate": value["estimate"],
                "interval": value["interval"],
                "seed_sd": value["seed_sd"],
                "run_ids": [
                    job["run_id"]
                    for job in sorted(matches, key=lambda item: item["seed"])
                ],
            }
        )
    if len({job["run_id"] for job in jobs}) != len(jobs):
        raise ValueError("duplicate run IDs")
    return rows


def synthetic_demo():
    """Illustrate memorized group identity, not a measured lesion experiment."""
    groups = ["a", "a", "b", "b", "c", "d"]
    y = [1, 1, 0, 0, 1, 0]
    frame = pd.DataFrame(
        {
            "image_id": [f"generated-{i}" for i in range(6)],
            "group_id": groups,
            "target": y,
            "prob_melanoma": [0.9, 0.9, 0.1, 0.1, 0.5, 0.5],
            "prediction": [1, 1, 0, 0, 1, 1],
        }
    )
    # An image-level holdout repeats a and b. The grouped holdout sees only c and d.
    naive = endpoints(frame.iloc[:4])
    grouped = endpoints(frame.iloc[4:])
    return {
        "kind": "illustrative synthetic data",
        "run_id": "synthetic-example-not-a-run",
        "counts": {"records": len(frame), "groups": frame.group_id.nunique()},
        "ungrouped_roc_auc": naive["roc_auc"],
        "grouped_roc_auc": grouped["roc_auc"],
    }


def synthetic_report():
    """Generate an illustrative, run-labeled CI fixture with the real evaluator."""
    models = {}
    for name, probabilities in {
        "logistic": [0.2, 0.2, 0.7, 0.7, 0.6, 0.3],
        "cnn_scratch_unweighted": [0.8, 0.8, 0.4, 0.4, 0.7, 0.2],
    }.items():
        models[name] = {}
        for seed in SEEDS:
            p = np.array(probabilities) + (seed - 42) * 0.0001
            models[name][seed] = pd.DataFrame(
                {
                    "image_id": [f"generated-{i}" for i in range(6)],
                    "group_id": ["a", "a", "b", "b", "c", "d"],
                    "target": [1, 1, 0, 0, 1, 0],
                    "prob_melanoma": p,
                    "prediction": (p >= 0.5).astype(int),
                }
            )
    report = paired_report(models, reference="logistic", draws=32, seed=2026)
    report["registry"] = {
        "jobs": [
            {
                "stratum": name,
                "seed": seed,
                "run_id": f"synthetic-{name}-{seed}-not-a-run",
                "status": "completed",
                "run_record_sha256": "illustrative-only",
            }
            for name in models
            for seed in SEEDS
        ]
    }
    return report
