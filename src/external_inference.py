"""Load trusted frozen models for evaluation without training or calibration fit."""

import importlib.metadata
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import (
    calibration,
    classical,
    data,
    external,
    features,
    run_contract,
    splitting,
)


def validate_audit(path, digest, manifest_sha256):
    if run_contract.sha256(Path(path)) != digest:
        raise ValueError("overlap audit hash mismatch")
    audit = run_contract._json(Path(path))
    if (
        audit.get("clear") is not True
        or audit.get("manifest_sha256") != manifest_sha256
        or audit.get("exact") != []
        or audit.get("near") != []
    ):
        raise ValueError("unresolved or stale overlap audit")
    return audit


def predict_raw(run_dir, record, image_ids, images_dir, device):
    """Score images without accepting external labels as model inputs.

    The caller must verify the release digest before loading its local pickle.
    """
    run_dir, images_dir = Path(run_dir), Path(images_dir)
    if record["pipeline"] == "classical_logistic":
        with (run_dir / "models/logistic_model.pkl").open("rb") as handle:
            fitted = pickle.load(handle)["fitted"]
        x = features.compute_hsv_histograms_batch(
            [
                data.load_image_hsv(
                    str(images_dir / f"{i}.jpg"), target_size=(256, 256)
                )
                for i in image_ids
            ]
        )
        return classical.predict_logistic(fitted, x)
    if record["pipeline"] not in {"deep_efficientnet_b0", "deep_small_cnn"}:
        raise ValueError("unsupported frozen model")
    import torch
    from torch.utils.data import DataLoader

    from src import deep

    config = record["config"]
    device = deep.get_default_device(device)
    deep.set_seed(config["seed"])
    model = deep.build_model(
        config["architecture"],
        dropout=config["dropout"],
        pretrained=False,
        freeze_backbone=config["architecture"] == "small_cnn",
    )
    checkpoint = torch.load(
        run_dir / "models" / f"{config['architecture']}_mc_dropout.pt",
        map_location="cpu",
        weights_only=True,
    )
    for key in ("architecture", "image_size", "dropout", "seed"):
        if checkpoint[key] != config[key]:
            raise ValueError(f"checkpoint mismatch: {key}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    dataset = deep.SkinLesionImageDataset(
        image_ids,
        np.zeros(len(image_ids), dtype=int),
        images_dir,
        transform=deep.build_transforms(config["image_size"], train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(config["seed"] + 3),
    )
    before = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    with torch.inference_mode():
        result = deep.predict_with_mc_dropout(
            model, loader, device, n_passes=config["mc_samples"]
        )
    if list(result["image_id"]) != list(image_ids):
        raise ValueError("inference image alignment mismatch")
    if any(
        not torch.equal(value.detach().cpu(), before[key])
        for key, value in model.state_dict().items()
    ):
        raise ValueError("model state changed during inference")
    return result["all_probabilities"]


def evaluate_run(
    *,
    root,
    release_path,
    release_sha256,
    manifest_path,
    manifest_sha256,
    audit_path,
    audit_sha256,
    images_dir,
    run_id,
    output,
    device="cpu",
):
    """Verify frozen inputs, write one new external run, and retain failures."""
    root, output, images_dir = Path(root), Path(output), Path(images_dir)
    release = external.verify_files(release_path, release_sha256, root)
    for path, digest in [(manifest_path, manifest_sha256), (audit_path, audit_sha256)]:
        if run_contract.sha256(Path(path)) != digest:
            raise ValueError("external input hash mismatch")
    manifest = run_contract._json(Path(manifest_path))
    content = {k: v for k, v in manifest.items() if k != "cohort_sha256"}
    if manifest.get("purpose") != "external" or manifest[
        "cohort_sha256"
    ] != splitting.canonical_hash(content):
        raise ValueError("invalid external manifest")
    validate_audit(audit_path, audit_sha256, manifest_sha256)
    selected = release["runs"][run_id]
    run_dir = root / selected["path"]
    record = run_contract._json(run_dir / "run.json")
    if (
        record["config_sha256"] != splitting.canonical_hash(record["config"])
        or record["status"] != "completed"
    ):
        raise ValueError("invalid frozen run configuration")
    policy = run_contract._json(run_dir / "models/decision_policy.json")
    rows = [r for r in manifest["rows"] if r["reason"] == "retained"]
    if not rows or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicated external membership")
    frame = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    if set(frame.target) != {0, 1}:
        raise ValueError("external cohort requires both classes")
    for row in manifest["rows"]:
        image_id = splitting._identifier(row["image_id"], "image ID")
        if (
            row["image_sha256"]
            and run_contract.sha256(images_dir / f"{image_id}.jpg")
            != row["image_sha256"]
        ):
            raise ValueError("external image hash mismatch")
    if set(frame.image_id) & set(policy["fit"]["image_ids"]):
        raise ValueError("external labels overlap policy fitting data")
    output.mkdir(parents=True, exist_ok=False)
    status = {
        "schema_version": 1,
        "status": "running",
        "run_id": run_id,
        "release_sha256": release_sha256,
        "manifest_sha256": manifest_sha256,
        "audit_sha256": audit_sha256,
        "cohort_sha256": manifest["cohort_sha256"],
        "device": "cpu" if record["pipeline"] == "classical_logistic" else device,
        "source": run_contract._source_record()[0],
        "software": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "torch",
                "torchvision",
                "pillow",
                "opencv-python-headless",
            )
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "training_software": record["environment"]["software"],
        "external_labels_used_for_fitting": False,
    }
    run_contract._write_json(output / "run.json", status)
    start = time.monotonic()
    try:
        raw = predict_raw(run_dir, record, frame.image_id.tolist(), images_dir, device)
        predictions = external.prediction_frame(frame, raw, policy, run_id=run_id)
        predictions["cohort_sha256"] = manifest["cohort_sha256"]
        predictions.to_csv(output / "predictions.csv", index=False)
        np.save(output / "raw_scores.npy", raw)
        report = calibration.policy_report(
            policy,
            frame.target.to_numpy(dtype=int),
            raw,
            variances=raw.var(axis=0, ddof=1) if raw.ndim == 2 else None,
        )
        run_contract._write_json(output / "policy-report.json", report)
        external.verify_files(release_path, release_sha256, root)
        status.update(
            status="completed",
            runtime_seconds=time.monotonic() - start,
            artifacts={
                p.name: run_contract.sha256(p)
                for p in output.iterdir()
                if p.name != "run.json"
            },
        )
    except BaseException as error:
        status.update(
            status="failed",
            runtime_seconds=time.monotonic() - start,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        run_contract._write_json(output / "run.json", status)
    return status


def validate_predictions(
    run_path,
    release,
    release_sha256,
    manifest,
    manifest_sha256,
    root,
    *,
    expected_run_id,
    audit_sha256,
):
    """Reconstruct saved external predictions from raw scores and frozen policy."""
    path, root = Path(run_path), Path(root)
    status = run_contract._json(path / "run.json")
    if status.get("run_id") != expected_run_id:
        raise ValueError("external run identity mismatch")
    if status.get("audit_sha256") != audit_sha256:
        raise ValueError("external run audit mismatch")
    if (
        status.get("status") != "completed"
        or status.get("release_sha256") != release_sha256
        or status.get("manifest_sha256") != manifest_sha256
        or status.get("cohort_sha256") != manifest["cohort_sha256"]
    ):
        raise ValueError("incomplete or stale external run")
    artifacts = status.get("artifacts", {})
    if set(artifacts) != {"predictions.csv", "raw_scores.npy", "policy-report.json"}:
        raise ValueError("invalid external artifact set")
    for name, digest in artifacts.items():
        if run_contract.sha256(path / name) != digest:
            raise ValueError("external artifact hash mismatch")
    run_id = status["run_id"]
    saved = release["runs"][run_id]
    policy = run_contract._json(root / saved["path"] / "models/decision_policy.json")
    cohort = (
        pd.DataFrame([r for r in manifest["rows"] if r["reason"] == "retained"])
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    raw = np.load(path / "raw_scores.npy", allow_pickle=False)
    expected = external.prediction_frame(cohort, raw, policy, run_id=run_id)
    expected["cohort_sha256"] = manifest["cohort_sha256"]
    observed = pd.read_csv(path / "predictions.csv", float_precision="round_trip")
    try:
        pd.testing.assert_frame_equal(
            observed, expected, check_dtype=False, check_exact=True
        )
    except AssertionError as error:
        raise ValueError(
            "external predictions differ from frozen policy or membership"
        ) from error
    report = calibration.policy_report(
        policy,
        cohort.target.to_numpy(dtype=int),
        raw,
        variances=raw.var(axis=0, ddof=1) if raw.ndim == 2 else None,
    )
    if report != run_contract._json(path / "policy-report.json"):
        raise ValueError("external policy report differs from frozen policy")
    return observed
