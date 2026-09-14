"""Immutable local run records and portable prediction artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluation, splitting, uncertainty

SCHEMA_VERSION = 1
PREDICTION_COLUMNS = (
    "schema_version",
    "run_id",
    "model_key",
    "role",
    "split_hash",
    "image_id",
    "lesion_id",
    "group_id",
    "target",
    "prob_melanoma",
    "prediction",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _git(*args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=True,
    )
    return result.stdout


def _source_record() -> tuple[dict, bytes, list[str]]:
    root = Path(__file__).resolve().parents[1]
    candidates = (
        _git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    )
    untracked = [
        name
        for name in candidates
        if name.endswith(".py")
        and (name.startswith("src/") or name.startswith("scripts/"))
    ]
    diff = _git("diff", "--binary", "HEAD")
    record = {
        "commit": _git("rev-parse", "HEAD").decode().strip(),
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_files": {
            name: sha256(root / name)
            for name in untracked
            if name and (root / name).is_file()
        },
    }
    return record, diff, list(record["untracked_files"])


def _config_value(key: str, value):
    if key in {"results_dir", "models_dir", "runs_dir"}:
        return None
    if isinstance(value, Path):
        return (
            {"input": key}
            if key in {"metadata_path", "images_dir", "split_manifest"}
            else value.name
        )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported config value for {key}")


def _check_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id
    ):
        raise ValueError("invalid run ID")
    return run_id


def _failure_reason(error: BaseException) -> str:
    message = str(error).lower()
    if isinstance(error, FileNotFoundError):
        return "missing_file"
    for prefix, code in (
        ("input changed", "input_changed"),
        ("source changed", "source_changed"),
        ("run manifest changed", "manifest_changed"),
        ("invalid split manifest", "invalid_manifest"),
        ("prediction", "invalid_predictions"),
        ("artifact", "invalid_artifact"),
        ("no finite checkpoint", "checkpoint_unavailable"),
    ):
        if message.startswith(prefix):
            return code
    return "unclassified"


def _manifest_rows(manifest):
    by_id = {row["image_id"]: row for row in manifest["rows"]}
    groups = {
        image_id: group["group_id"]
        for group in manifest["groups"]
        for image_id in group["image_ids"]
    }
    return by_id, groups


def _normalize_predictions(frame, *, role, manifest, run_id, model_key):
    required = {"image_id", "target", "prediction"}
    if not required.issubset(frame.columns):
        raise ValueError("missing prediction columns")
    prob_col = "prob_melanoma" if "prob_melanoma" in frame else "mean_prob_melanoma"
    if prob_col not in frame:
        raise ValueError("missing probability column")
    expected = manifest["partitions"][role]
    if frame.image_id.tolist() != expected:
        raise ValueError("prediction IDs differ from manifest")
    by_id, groups = _manifest_rows(manifest)
    targets = [by_id[image_id]["target"] for image_id in expected]
    if frame.target.tolist() != targets:
        raise ValueError("prediction targets differ from manifest")
    probabilities = pd.to_numeric(frame[prob_col], errors="raise").to_numpy(dtype=float)
    predictions = pd.to_numeric(frame["prediction"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("invalid probabilities")
    if not np.isfinite(predictions).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("invalid predictions")
    return pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "model_key": model_key,
            "role": role,
            "split_hash": manifest["split_hash"],
            "image_id": expected,
            "lesion_id": [by_id[i]["lesion_id"] for i in expected],
            "group_id": [groups[i] for i in expected],
            "target": targets,
            "prob_melanoma": probabilities,
            "prediction": predictions.astype(int),
        }
    )


def _check_deep_arrays(path: Path, record: dict, predictions: pd.DataFrame) -> None:
    if not record["pipeline"].startswith("deep_"):
        return
    architecture = record["pipeline"].removeprefix("deep_")
    arrays = path / "arrays"
    calibration = predictions.loc[predictions.role == "calibration"]
    development = predictions.loc[predictions.role == "development"]
    if calibration.empty or development.empty:
        raise ValueError("missing deep prediction role")
    cal_mc = np.load(
        arrays / f"{architecture}_calibration_mc_probabilities.npy", allow_pickle=False
    )
    dev_mc = np.load(
        arrays / f"{architecture}_development_mc_probabilities.npy", allow_pickle=False
    )
    dev_prob = np.load(
        arrays / f"{architecture}_development_probability.npy", allow_pickle=False
    )
    dev_uncertainty = np.load(
        arrays / f"{architecture}_development_uncertainty.npy", allow_pickle=False
    )
    dev_target = np.load(
        arrays / f"{architecture}_development_y_true.npy", allow_pickle=False
    )
    passes = int(record["config"]["mc_samples"])
    if (
        cal_mc.shape != (passes, len(calibration))
        or dev_mc.shape != (passes, len(development))
        or any(
            array.shape != (len(development),)
            for array in (dev_prob, dev_uncertainty, dev_target)
        )
    ):
        raise ValueError("deep array sample count mismatch")
    if (
        not np.isfinite(cal_mc).all()
        or not np.isfinite(dev_mc).all()
        or np.any((cal_mc < 0) | (cal_mc > 1))
        or np.any((dev_mc < 0) | (dev_mc > 1))
    ):
        raise ValueError("invalid MC probability array")
    if (
        not np.allclose(
            cal_mc.mean(axis=0), calibration.prob_melanoma, rtol=1e-6, atol=1e-7
        )
        or not np.allclose(dev_mc.mean(axis=0), dev_prob, rtol=1e-6, atol=1e-7)
        or not np.allclose(dev_prob, development.prob_melanoma, rtol=1e-6, atol=1e-7)
        or not np.allclose(
            dev_mc.std(axis=0, ddof=1), dev_uncertainty, rtol=1e-6, atol=1e-7
        )
        or not np.array_equal(dev_target, development.target.to_numpy())
    ):
        raise ValueError("deep array content disagrees with predictions")
    for role, mc in (("calibration", cal_mc), ("development", dev_mc)):
        file = path / "results" / "tables" / f"{architecture}_predictions_{role}.csv"
        producer = pd.read_csv(file)
        required = {"predictive_std", "predictive_entropy", "mutual_information"}
        if not required.issubset(producer.columns):
            raise ValueError("missing deep uncertainty columns")
        expected = uncertainty.summarize_mc_dropout_probabilities(mc)
        for column, values in (
            ("predictive_std", np.sqrt(expected["variance"])),
            ("predictive_entropy", expected["predictive_entropy"]),
            ("mutual_information", expected["mutual_information"]),
        ):
            observed = pd.to_numeric(producer[column], errors="raise").to_numpy(
                dtype=float
            )
            if not np.isfinite(observed).all() or not np.allclose(
                observed, values, rtol=1e-6, atol=1e-7
            ):
                raise ValueError("deep uncertainty disagrees with MC samples")


class RunRecord:
    def __init__(self, path: Path, record: dict, inputs: dict[str, Path]):
        self.path = path
        self.record = record
        self.input_paths = {name: Path(source) for name, source in inputs.items()}
        self.started_monotonic = time.monotonic()

    @classmethod
    def start(
        cls,
        root: Path,
        *,
        run_id: str | None,
        pipeline: str,
        config: dict,
        inputs: dict[str, Path],
        resume_from: str | None = None,
    ):
        root = Path(root)
        run_id = _check_id(run_id or uuid.uuid4().hex)
        if resume_from is not None:
            previous = root / _check_id(resume_from)
            prior = _json(previous / "run.json")
            if prior.get("status") not in {"failed", "running"}:
                raise ValueError("resume source must be failed or incomplete")
        path = root / run_id
        path.mkdir(parents=True, exist_ok=False)
        stage = "hash_inputs"
        try:
            input_hashes = {
                name: sha256(Path(source)) for name, source in inputs.items()
            }
            stage = "snapshot_manifest"
            input_dir = path / "inputs"
            input_dir.mkdir()
            shutil.copyfile(inputs["split_manifest"], input_dir / "split-manifest.json")
            if (
                sha256(input_dir / "split-manifest.json")
                != input_hashes["split_manifest"]
            ):
                raise ValueError("split manifest changed during run startup")
            stage = "snapshot_source"
            source, diff, untracked = _source_record()
            (input_dir / "source.diff").write_bytes(diff)
            project_root = Path(__file__).resolve().parents[1]
            for name in untracked:
                destination = input_dir / "untracked" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(project_root / name, destination)
                if sha256(destination) != source["untracked_files"][name]:
                    raise ValueError("untracked source changed during run startup")
            stage = "capture_environment"
            software = {}
            for package in ("numpy", "pandas", "scikit-learn", "torch", "torchvision"):
                try:
                    software[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    software[package] = None
            clean_config = {
                key: clean
                for key, value in config.items()
                if (clean := _config_value(key, value)) is not None
            }
            record = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "pipeline": pipeline,
                "status": "running",
                "resume_policy": "new-run-only",
                "resume_from": resume_from,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": clean_config,
                "config_sha256": splitting.canonical_hash(clean_config),
                "inputs": input_hashes,
                "source": source,
                "environment": {
                    "python": platform.python_version(),
                    "platform": sys.platform,
                    "machine": platform.machine(),
                    "software": software,
                },
                "artifacts": {},
            }
            stage = "write_record"
            _write_json(path / "run.json", record)
            return cls(path, record, inputs)
        except BaseException as error:
            # The unique directory remains as evidence even if provenance capture fails.
            _write_json(
                path / "run.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "status": "failed",
                    "failure": {
                        "type": type(error).__name__,
                        "stage": stage,
                        "reason": _failure_reason(error),
                    },
                },
            )
            raise

    def fail(self, error: BaseException, *, stage: str = "run") -> None:
        if self.record["status"] == "completed":
            raise ValueError("cannot fail a completed run")
        self.record["status"] = "failed"
        self.record["failure"] = {
            "type": type(error).__name__,
            "stage": stage,
            "reason": _failure_reason(error),
        }
        self.record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self.record["runtime_seconds"] = time.monotonic() - self.started_monotonic
        _write_json(self.path / "run.json", self.record)

    def finish(self, manifest: dict, prediction_files: dict[str, Path]) -> None:
        if self.record["status"] != "running":
            raise ValueError("only a running run can complete")
        for name, source in self.input_paths.items():
            if sha256(source) != self.record["inputs"][name]:
                raise ValueError(f"input changed during run: {name}")
        current_source, _, _ = _source_record()
        if current_source != self.record["source"]:
            raise ValueError("source changed during run")
        if manifest["split_hash"] != splitting.canonical_hash(
            {key: value for key, value in manifest.items() if key != "split_hash"}
        ):
            raise ValueError("invalid split manifest hash")
        source_manifest = _json(self.path / "inputs" / "split-manifest.json")
        if (
            source_manifest != manifest
            or sha256(self.path / "inputs" / "split-manifest.json")
            != self.record["inputs"]["split_manifest"]
        ):
            raise ValueError("run manifest changed")
        rows = []
        for role, file in prediction_files.items():
            if role not in splitting.ROLES or not Path(file).is_relative_to(self.path):
                raise ValueError("invalid prediction artifact")
            frame = pd.read_csv(file)
            normalized = _normalize_predictions(
                frame,
                role=role,
                manifest=manifest,
                run_id=self.record["run_id"],
                model_key=self.record["pipeline"],
            )
            extras = frame.drop(
                columns=[name for name in PREDICTION_COLUMNS if name in frame]
            )
            pd.concat([normalized, extras], axis=1).to_csv(file, index=False)
            rows.append(normalized)
        if not rows:
            raise ValueError("no prediction artifacts")
        predictions = pd.concat(rows, ignore_index=True)
        predictions.to_csv(self.path / "predictions.csv", index=False)
        _check_deep_arrays(self.path, self.record, predictions)
        self.record["split_hash"] = manifest["split_hash"]
        self.record["cohort_hash"] = manifest["cohort_hash"]
        self.record["split_seed"] = manifest["seed"]
        self.record["grouping"] = manifest["grouping"]
        self.record["prediction_roles"] = sorted(prediction_files)
        self.record["prediction_files"] = {
            role: str(Path(file).relative_to(self.path))
            for role, file in prediction_files.items()
        }
        self.record["array_schemas"] = {
            str(file.relative_to(self.path)): {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            for file in sorted(self.path.rglob("*.npy"))
            for array in [np.load(file, allow_pickle=False)]
        }
        self.record["artifacts"] = {
            str(file.relative_to(self.path)): sha256(file)
            for file in sorted(self.path.rglob("*"))
            if file.is_file() and file != self.path / "run.json"
        }
        completed = dict(self.record)
        completed["status"] = "completed"
        completed["finished_utc"] = datetime.now(timezone.utc).isoformat()
        completed["runtime_seconds"] = time.monotonic() - self.started_monotonic
        _write_json(self.path / "run.json", completed)
        self.record = completed


def validate_run(run_dir: Path, *, input_paths: dict[str, Path] | None = None):
    path = Path(run_dir)
    record = _json(path / "run.json")
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("status") != "completed"
    ):
        raise ValueError("run is not completed or has an unsupported schema")
    if _check_id(record.get("run_id")) != path.name:
        raise ValueError("run ID/path mismatch")
    if splitting.canonical_hash(record["config"]) != record["config_sha256"]:
        raise ValueError("config hash mismatch")
    expected_files = set(record["artifacts"])
    actual_files = {
        str(file.relative_to(path))
        for file in path.rglob("*")
        if file.is_file() and file != path / "run.json"
    }
    if expected_files != actual_files:
        raise ValueError("missing or unregistered artifact")
    if set(record["array_schemas"]) != {
        name for name in expected_files if name.endswith(".npy")
    }:
        raise ValueError("array artifact set mismatch")
    for name, digest in record["artifacts"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid artifact path")
        file = path / name
        if sha256(file) != digest:
            raise ValueError(f"artifact hash mismatch: {name}")
        if file.suffix == ".npy":
            array = np.load(file, allow_pickle=False)
            if (
                not np.issubdtype(array.dtype, np.number)
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"invalid array: {name}")
            if record["array_schemas"].get(name) != {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }:
                raise ValueError(f"array schema mismatch: {name}")
        elif file.suffix == ".json":
            _json(file)
    if input_paths:
        for name, file in input_paths.items():
            if sha256(Path(file)) != record["inputs"][name]:
                raise ValueError(f"input hash mismatch: {name}")
    manifest = _json(path / "inputs" / "split-manifest.json")
    if (
        hashlib.sha256((path / "inputs" / "source.diff").read_bytes()).hexdigest()
        != record["source"]["dirty_diff_sha256"]
    ):
        raise ValueError("source diff hash mismatch")
    for name, digest in record["source"]["untracked_files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid untracked source path")
        if sha256(path / "inputs" / "untracked" / name) != digest:
            raise ValueError("untracked source hash mismatch")
    if (
        sha256(path / "inputs" / "split-manifest.json")
        != record["inputs"]["split_manifest"]
        or manifest["split_hash"] != record["split_hash"]
        or manifest["cohort_hash"] != record["cohort_hash"]
        or manifest["grouping"] != record["grouping"]
    ):
        raise ValueError("split manifest hash mismatch")
    if (
        splitting.canonical_hash(
            {key: value for key, value in manifest.items() if key != "split_hash"}
        )
        != manifest["split_hash"]
    ):
        raise ValueError("invalid split manifest content hash")
    frame = pd.read_csv(path / "predictions.csv")
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        raise ValueError("prediction schema mismatch")
    roles = record["prediction_roles"]
    if sorted(frame.role.unique()) != roles:
        raise ValueError("prediction roles mismatch")
    for role in roles:
        subset = frame.loc[frame.role == role].reset_index(drop=True)
        normalized = _normalize_predictions(
            subset,
            role=role,
            manifest=manifest,
            run_id=record["run_id"],
            model_key=record["pipeline"],
        )
        if not normalized.equals(subset):
            raise ValueError("prediction identity or grouping mismatch")
        producer_name = record["prediction_files"][role]
        if producer_name not in expected_files:
            raise ValueError("missing producer prediction artifact")
        producer = pd.read_csv(path / producer_name)
        if not set(PREDICTION_COLUMNS).issubset(producer.columns):
            raise ValueError("producer prediction schema mismatch")
        if not producer[list(PREDICTION_COLUMNS)].equals(subset):
            raise ValueError("producer predictions disagree with shared records")
    _check_deep_arrays(path, record, frame)
    return record, frame


def recompute_metrics(run_dir: Path):
    """Recompute binary scores from checked prediction records only."""
    _, predictions = validate_run(run_dir)
    metrics = {}
    for role, frame in predictions.groupby("role", sort=True):
        metrics[role] = evaluation.compute_classification_metrics(
            frame.target.to_numpy(),
            frame.prediction.to_numpy(),
            frame.prob_melanoma.to_numpy(),
        )
    return metrics


def recompute_report(run_dir: Path):
    """Reproduce reported operating-point scores without cohort files."""
    record, predictions = validate_run(run_dir)
    if record["pipeline"] == "classical_gmm":
        summary_path = Path(run_dir) / "results" / "metrics_summary.json"
        points = {
            "cost_threshold": lambda p, c: c.prediction.to_numpy(),
            "map_threshold": lambda p, c: (p >= 0.5).astype(int),
        }
    elif record["pipeline"].startswith("deep_"):
        architecture = record["pipeline"].removeprefix("deep_")
        summary_path = (
            Path(run_dir) / "results" / f"{architecture}_metrics_summary.json"
        )
        points = None
    else:
        return {"metrics": recompute_metrics(run_dir)}
    summary = _json(summary_path)
    cost_fn = float(summary["cost_matrix"]["false_negative"])
    cost_fp = float(summary["cost_matrix"]["false_positive"])
    if points is None:
        points = {
            "cost_formula_threshold": lambda p, c: (
                p >= float(summary["cost_matrix"]["cost_formula_threshold"])
            ).astype(int),
            "calibration_cost_threshold": lambda p, c: (
                p
                >= float(summary["cost_matrix"]["selected_threshold_from_calibration"])
            ).astype(int),
            "map_threshold": lambda p, c: (p >= 0.5).astype(int),
        }
    metrics = {}
    ece = {}
    for role, frame in predictions.groupby("role", sort=True):
        y_true = frame.target.to_numpy()
        probabilities = frame.prob_melanoma.to_numpy()
        metrics[role] = {}
        for name, predict in points.items():
            labels = predict(probabilities, frame)
            scores = evaluation.compute_classification_metrics(
                y_true, labels, probabilities
            )
            scores = evaluation.add_average_cost(scores, len(frame), cost_fn, cost_fp)
            if record["pipeline"].startswith("deep_"):
                threshold_name = {
                    "cost_formula_threshold": "cost_formula_threshold",
                    "calibration_cost_threshold": "selected_threshold_from_calibration",
                    "map_threshold": "map_threshold",
                }[name]
                scores["threshold"] = float(summary["cost_matrix"][threshold_name])
            metrics[role][name] = scores
        ece[role] = evaluation.compute_calibration_error(y_true, probabilities)
    result = {"metrics": metrics, "expected_calibration_error": ece}
    if record["pipeline"].startswith("deep_"):
        file = (
            Path(run_dir)
            / "results"
            / "tables"
            / f"{architecture}_predictions_development.csv"
        )
        frame = pd.read_csv(file)
        if not {"predictive_std", "mutual_information"}.issubset(frame.columns):
            raise ValueError("missing uncertainty prediction columns")
        result["mc_dropout_uncertainty"] = {
            "development_mean_predictive_std": float(frame.predictive_std.mean()),
            "development_median_predictive_std": float(frame.predictive_std.median()),
            "development_p90_predictive_std": float(frame.predictive_std.quantile(0.9)),
            "development_mean_mutual_information": float(
                frame.mutual_information.mean()
            ),
        }
    return result
