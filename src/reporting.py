"""Helpers for durable report artifacts."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_builtin(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-serializable Python objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_builtin(item) for item in value]
    return value


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_builtin(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


def build_metrics_rows(
    metrics_by_split: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    ece_by_split: Mapping[str, float],
) -> list[dict[str, float | int | str]]:
    rows = []
    for split_name, operating_points in metrics_by_split.items():
        for operating_point, metrics in operating_points.items():
            row: dict[str, float | int | str] = {
                "split": split_name,
                "operating_point": operating_point,
                "ece": ece_by_split[split_name],
            }
            row.update(metrics)
            rows.append(row)
    return rows


def save_metrics_tables(
    metrics_by_split: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    ece_by_split: Mapping[str, float],
    tables_path: Path,
) -> pd.DataFrame:
    tables_path.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(build_metrics_rows(metrics_by_split, ece_by_split))
    metrics_df.to_csv(tables_path / "metrics_summary.csv", index=False)

    threshold_columns = [
        "split",
        "operating_point",
        "recall",
        "specificity",
        "precision",
        "f1",
        "fp",
        "fn",
        "tp",
        "tn",
        "average_cost",
    ]
    metrics_df[threshold_columns].to_csv(
        tables_path / "threshold_comparison.csv", index=False
    )
    return metrics_df


def selection_summary(
    selection_info: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    summary = {}
    for class_label, info in selection_info.items():
        class_name = str(info["class_name"]).lower()
        bic_info = info["bic_info"]
        summary[class_name] = {
            "class_label": int(class_label),
            "selected_components": int(info["n_components"]),
            "component_range": bic_info["component_range"],
            "bic_scores": bic_info["bic_scores"],
        }
    return summary


def save_split_tables(
    split_report: Mapping[str, Any], tables_path: Path
) -> pd.DataFrame:
    rows = []
    for split_name, split_stats in split_report["splits"].items():
        row = {"split": split_name}
        row.update(split_stats)
        rows.append(row)

    split_df = pd.DataFrame(rows)
    split_df.to_csv(tables_path / "split_summary.csv", index=False)
    return split_df
