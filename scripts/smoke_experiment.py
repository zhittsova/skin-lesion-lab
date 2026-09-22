"""Run the public pipelines on generated images and check their saved evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def independent_metrics(targets, probabilities, decisions, cost_fn, cost_fp):
    """Compute binary scores without calling the application's metric helpers."""
    y = np.asarray(targets, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    d = np.asarray(decisions, dtype=int)
    if len(y) == 0 or not (len(y) == len(p) == len(d)):
        raise ValueError("metric inputs must have equal nonzero length")
    if not np.isin(y, [0, 1]).all() or not np.isin(d, [0, 1]).all():
        raise ValueError("metric labels must be binary")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("metric probabilities must be finite and bounded")
    tp = int(np.sum((y == 1) & (d == 1)))
    tn = int(np.sum((y == 0) & (d == 0)))
    fp = int(np.sum((y == 0) & (d == 1)))
    fn = int(np.sum((y == 1) & (d == 0)))
    positive = p[y == 1]
    negative = p[y == 0]
    auc = (
        float(
            np.mean(
                (positive[:, None] > negative[None, :])
                + 0.5 * (positive[:, None] == negative[None, :])
            )
        )
        if len(positive) and len(negative)
        else None
    )
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / len(y),
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "roc_auc": auc,
        "brier_score": float(np.mean((p - y) ** 2)),
        "average_cost": (cost_fn * fn + cost_fp * fp) / len(y),
    }


def run_cli(root: Path, name: str, args: list[str], *, success=True):
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(cache / "matplotlib"),
            "XDG_CACHE_HOME": str(cache),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    result = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
    )
    logs = root / "command_logs"
    logs.mkdir(exist_ok=True)
    (logs / f"{name}.stdout.txt").write_text(result.stdout)
    (logs / f"{name}.stderr.txt").write_text(result.stderr)
    if (result.returncode == 0) != success:
        raise AssertionError(
            f"{name}: exit {result.returncode}; see {logs / f'{name}.stderr.txt'}"
        )
    return result


def make_cohort(root: Path) -> tuple[Path, Path]:
    images = root / "images"
    images.mkdir()
    rows = []
    rng = np.random.default_rng(123)
    for label in (0, 1):
        for index in range(20):
            image_id = {
                (0, 0): "001",
                (0, 1): "NA",
                (1, 0): "000",
                (1, 1): "NULL",
            }.get((label, index), f"S{label}_{index:03d}")
            pixels = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(images / f"{image_id}.jpg")
            rows.append(
                {
                    "image_id": image_id,
                    "lesion_id": f"L{label}_{index // 2:03d}",
                    "dx": "mel" if label else "nv",
                }
            )
    metadata = root / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata, index=False)
    return metadata, images


def check_run(
    root: Path,
    run_id: str,
    manifest: dict,
    *,
    deep: bool,
    requested_device: str | None = None,
) -> dict:
    run_dir = root / "runs" / run_id
    record = json.loads((run_dir / "run.json").read_text())
    if record["status"] != "completed" or record["run_id"] != run_id:
        raise AssertionError(f"{run_id}: incomplete or wrong run record")
    if record["split_hash"] != manifest["split_hash"]:
        raise AssertionError(f"{run_id}: wrong split hash")
    observed_device = None
    if deep:
        if requested_device not in {"cpu", "cuda"}:
            raise ValueError("deep smoke verification requires cpu or cuda")
        configured_device = record.get("config", {}).get("device")
        observed_device = record.get("environment", {}).get("device")
        if configured_device != requested_device or observed_device != requested_device:
            raise AssertionError(
                f"{run_id}: requested or observed backend differs: "
                f"requested={requested_device}, configured={configured_device}, "
                f"observed={observed_device}"
            )
    summary_name = "small_cnn_metrics_summary.json" if deep else "metrics_summary.json"
    summary = json.loads((run_dir / "results" / summary_name).read_text())
    if summary.get("metrics_version") != 2:
        raise AssertionError(f"{run_id}: new run requires metrics version 2")
    from src.run_contract import _read_csv

    predictions = _read_csv(run_dir / "predictions.csv")
    by_id = {row["image_id"]: row["target"] for row in manifest["rows"]}
    checked = 0
    for role, points in summary["metrics"].items():
        frame = predictions.loc[predictions.role == role]
        expected_ids = manifest["partitions"][role]
        if frame.image_id.tolist() != expected_ids:
            raise AssertionError(f"{run_id}: {role} IDs differ from frozen manifest")
        if frame.target.tolist() != [by_id[image_id] for image_id in expected_ids]:
            raise AssertionError(f"{run_id}: {role} labels differ from frozen manifest")
        p = frame.prob_melanoma.to_numpy(dtype=float)
        for point, saved in points.items():
            if point == "cost_threshold" and not deep:
                decisions = frame.prediction.to_numpy(dtype=int)
            else:
                threshold_key = {
                    "map_threshold": "map_threshold",
                    "cost_formula_threshold": "cost_formula_threshold",
                    "calibration_cost_threshold": "selected_threshold_from_calibration",
                }[point]
                decisions = (p >= summary["cost_matrix"][threshold_key]).astype(int)
            observed = independent_metrics(
                frame.target,
                p,
                decisions,
                summary["cost_matrix"]["false_negative"],
                summary["cost_matrix"]["false_positive"],
            )
            for metric, value in observed.items():
                if (value is None or saved[metric] is None) and value != saved[metric]:
                    raise AssertionError(
                        f"{run_id}: {role}/{point}/{metric}: {value} != {saved[metric]}"
                    )
                if value is not None and not math.isclose(
                    value, saved[metric], rel_tol=1e-7, abs_tol=1e-8
                ):
                    raise AssertionError(
                        f"{run_id}: {role}/{point}/{metric}: {value} != {saved[metric]}"
                    )
                checked += 1
    if deep:
        from src import deep as deep_model

        checkpoint = torch.load(
            run_dir / "models" / "small_cnn_mc_dropout.pt",
            map_location="cpu",
            weights_only=True,
        )
        model = deep_model.build_model("small_cnn", dropout=0.3)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        torch.set_num_threads(1)
        image_id = manifest["partitions"]["development"][0]
        image = Image.open(root / "images" / f"{image_id}.jpg").convert("RGB")
        tensor = deep_model.build_transforms(32, train=False)(image).unsqueeze(0)
        with torch.no_grad():
            first = model(tensor).cpu().numpy()
            second = model(tensor).cpu().numpy()
        np.testing.assert_allclose(first, second, rtol=1e-7, atol=1e-8)
    result = {"run_id": run_id, "metric_values_checked": checked, "roles": 4}
    if deep:
        result["requested_device"] = requested_device
        result["observed_device"] = observed_device
    return result


def execute(root: Path, device="cpu") -> dict:
    if not isinstance(device, str) or device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    metadata, images = make_cohort(root)
    manifest_path = root / "manifest.json"
    common = [
        "--source",
        "ham10000",
        "--metadata-path",
        str(metadata),
        "--images-dir",
        str(images),
        "--split-manifest",
        str(manifest_path),
    ]
    run_cli(root, "freeze", ["freeze_splits.py", *common])
    manifest = json.loads(manifest_path.read_text())
    run_root = root / "runs"
    run_cli(
        root,
        "classical",
        [
            "train_pipeline.py",
            *common,
            "--runs-dir",
            str(run_root),
            "--run-id",
            "classical-smoke",
            "--model",
            "prevalence",
        ],
    )
    run_cli(
        root,
        "deep",
        [
            "train_deep_pipeline.py",
            *common,
            "--runs-dir",
            str(run_root),
            "--run-id",
            "deep-smoke",
            "--epochs",
            "1",
            "--mc-samples",
            "2",
            "--image-size",
            "32",
            "--batch-size",
            "8",
            "--seed",
            "73",
            "--device",
            device,
        ],
    )
    results = [
        check_run(root, "classical-smoke", manifest, deep=False),
        check_run(
            root,
            "deep-smoke",
            manifest,
            deep=True,
            requested_device=device,
        ),
    ]
    for run_id in ("classical-smoke", "deep-smoke"):
        run_cli(
            root,
            f"summarize-{run_id}",
            [
                "summarize_results.py",
                "--run-dir",
                str(run_root / run_id),
                "--output",
                str(root / f"{run_id}-recomputed.json"),
            ],
        )
    stale = run_root / "classical-smoke" / "predictions.csv"
    original = stale.read_bytes()
    try:
        stale.write_bytes(original + b"\n")
        failed = run_cli(
            root,
            "stale-artifact",
            ["summarize_results.py", "--run-dir", str(run_root / "classical-smoke")],
            success=False,
        )
        if "artifact hash mismatch" not in failed.stderr:
            raise AssertionError("stale artifact lacked a diagnostic")
    finally:
        stale.write_bytes(original)
    malformed = root / "malformed.csv"
    bad = pd.read_csv(metadata, keep_default_na=False, dtype=str)
    bad.loc[0, "dx"] = "unknown-diagnosis"
    bad.to_csv(malformed, index=False)
    run_cli(
        root,
        "malformed-cohort",
        [
            "train_pipeline.py",
            *common[:2],
            "--metadata-path",
            str(malformed),
            *common[4:],
            "--runs-dir",
            str(run_root),
            "--run-id",
            "malformed-smoke",
            "--model",
            "prevalence",
        ],
        success=False,
    )
    malformed_record = json.loads(
        (run_root / "malformed-smoke" / "run.json").read_text()
    )
    if (
        malformed_record["status"] != "failed"
        or malformed_record["failure"]["reason"] != "invalid_cohort"
    ):
        raise AssertionError("malformed cohort left no diagnostic run record")
    invalid_ids = root / "invalid-ids.csv"
    invalid = pd.read_csv(metadata, keep_default_na=False, dtype=str)
    invalid.loc[0, "image_id"] = ""
    invalid.to_csv(invalid_ids, index=False)
    run_cli(
        root,
        "missing-image-id",
        [
            "freeze_splits.py",
            "--source",
            "ham10000",
            "--metadata-path",
            str(invalid_ids),
            "--images-dir",
            str(images),
            "--split-manifest",
            str(root / "invalid-manifest.json"),
        ],
        success=False,
    )
    interrupt_code = (
        "import train_pipeline; from unittest.mock import patch; "
        "patch.object(train_pipeline, '_run', side_effect=KeyboardInterrupt("
        "'synthetic interruption')).start(); train_pipeline.main()"
    )
    run_cli(
        root,
        "interrupted-run",
        [
            "-c",
            interrupt_code,
            *common,
            "--runs-dir",
            str(run_root),
            "--run-id",
            "interrupted-smoke",
            "--model",
            "prevalence",
        ],
        success=False,
    )
    interrupted = json.loads((run_root / "interrupted-smoke" / "run.json").read_text())
    if (
        interrupted["status"] != "failed"
        or interrupted["failure"]["type"] != "KeyboardInterrupt"
        or interrupted["failure"]["reason"] != "interrupted"
    ):
        raise AssertionError("interrupted CLI left no failed run record")
    run_cli(
        root,
        "recovery",
        [
            "train_pipeline.py",
            *common,
            "--runs-dir",
            str(run_root),
            "--run-id",
            "recovered-smoke",
            "--resume-from",
            "interrupted-smoke",
            "--model",
            "prevalence",
        ],
    )
    recovered = json.loads((run_root / "recovered-smoke" / "run.json").read_text())
    if (
        recovered["resume_from"] != "interrupted-smoke"
        or recovered["status"] != "completed"
    ):
        raise AssertionError("failed-run recovery did not create a completed new run")
    return {
        "split_hash": manifest["split_hash"],
        "deep_backend": results[1]["observed_device"],
        "runs": results,
        "failure_cases": 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, help="Keep generated evidence here.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=False)
        result = execute(args.work_dir.resolve(), device=args.device)
    else:
        with tempfile.TemporaryDirectory(prefix="skin-lesion-smoke-") as temporary:
            result = execute(Path(temporary), device=args.device)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
