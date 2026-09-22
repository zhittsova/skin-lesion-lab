"""Load trusted frozen models for evaluation without training or calibration fit."""

import importlib.metadata
import pickle
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src import (
    bayes,
    calibration,
    classical,
    data,
    external,
    external_release,
    features,
    gmm,
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


def validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("invalid external manifest")
    content = {k: v for k, v in manifest.items() if k != "cohort_sha256"}
    if (
        manifest.get("schema_version") != 1
        or manifest.get("purpose") != "external"
        or manifest.get("cohort_sha256") != splitting.canonical_hash(content)
    ):
        raise ValueError("invalid external manifest")
    rows = manifest.get("rows")
    counts = manifest.get("counts")
    if not isinstance(rows, list):
        raise ValueError("invalid external manifest rows")
    if not isinstance(counts, dict):
        raise ValueError("invalid external manifest counts: expected an object")
    required = {"input", "retained"}
    allowed = {"input", *external.EXTERNAL_ROW_REASONS}
    if not required.issubset(counts) or not set(counts).issubset(allowed):
        raise ValueError("invalid external manifest counts: missing or unknown key")
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError(
            "invalid external manifest counts: values must be nonnegative integers"
        )

    ids = []
    reasons = Counter()
    for row in rows:
        if not isinstance(row, dict) or row.get("reason") not in allowed - {"input"}:
            raise ValueError("invalid external manifest counts: unknown row reason")
        reasons[row["reason"]] += 1
        ids.append(splitting._identifier(row["image_id"], "image ID"))
        if not isinstance(row.get("group_id"), str) or not row["group_id"].strip():
            raise ValueError("missing external group identity")
        if row["reason"] == "retained" and not row.get("image_sha256"):
            raise ValueError("retained image lacks a hash")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("empty or duplicated external membership")
    nonzero_exclusions = {
        reason for reason in external.EXTERNAL_EXCLUSION_REASONS if reasons[reason] > 0
    }
    if not nonzero_exclusions.issubset(counts):
        raise ValueError("invalid external manifest counts: missing nonzero exclusion")
    expected = {"input": len(rows), **reasons}
    if any(counts[key] != expected.get(key, 0) for key in counts):
        raise ValueError("invalid external manifest counts: values differ from rows")
    excluded = sum(
        counts.get(reason, 0) for reason in external.EXTERNAL_EXCLUSION_REASONS
    )
    if counts["input"] - counts["retained"] != excluded:
        raise ValueError(
            "invalid external manifest counts: attrition does not reconcile"
        )
    return manifest


def cache_dataset(dataset):
    """Reuse deterministic evaluation tensors across MC passes."""
    return [dataset[i] for i in range(len(dataset))]


def validate_raw(record, raw, count):
    expected = (
        (record["config"]["mc_samples"], count)
        if record["pipeline"].startswith("deep_")
        else (count,)
    )
    if np.asarray(raw).shape != expected:
        raise ValueError("saved scores differ from the fitted inference dimensions")


def predict_raw(run_dir, record, image_ids, images_dir, device):
    """Score images without accepting external labels as model inputs.

    The caller must verify the release digest before loading its local pickle.
    """
    run_dir, images_dir = Path(run_dir), Path(images_dir)
    if record["pipeline"].startswith("classical_"):
        with (run_dir / external_release.estimator_name(record)).open("rb") as handle:
            saved = pickle.load(handle)
        fitted = saved["fitted"]
        family = record["pipeline"].removeprefix("classical_")
        if saved.get("model_kind") != family:
            raise ValueError("estimator model kind mismatch")
        if family == "prevalence":
            return classical.predict_prevalence(fitted, len(image_ids))
        if (
            family == "logistic"
            and fitted["model"].random_state != record["config"]["seed"]
        ):
            raise ValueError("estimator fitted seed mismatch")
        x = features.compute_hsv_histograms_batch(
            [
                data.load_image_hsv(
                    str(images_dir / f"{i}.jpg"), target_size=(256, 256)
                )
                for i in image_ids
            ]
        )
        if family == "logistic":
            return classical.predict_logistic(fitted, x)
        if family != "gmm":
            raise ValueError("unsupported classical estimator")
        x = features.apply_standardization(x, saved["train_mean"], saved["train_std"])
        likelihood = gmm.compute_class_likelihoods(fitted, x)
        if record["config"].get("decision_policy_version", 1) == 2:
            return bayes.compute_posterior_log_odds(likelihood, saved["class_priors"])
        return bayes.compute_posterior_probabilities(likelihood, saved["class_priors"])[
            :, 1
        ]
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
    for key in (
        "architecture",
        "image_size",
        "dropout",
        "seed",
        "fine_tune_backbone",
        "pretrained",
        "loss_strategy",
    ):
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
    if len(dataset) * 3 * config["image_size"] ** 2 * 4 > 512 * 1024**2:
        raise ValueError("external evaluation tensor cache exceeds 512 MiB")
    loader = DataLoader(
        cache_dataset(dataset),
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
    manifest = validate_manifest(run_contract._json(Path(manifest_path)))
    validate_audit(audit_path, audit_sha256, manifest_sha256)
    selected = release["runs"][run_id]
    if selected["device"] != device:
        raise ValueError("device differs from released execution configuration")
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
    for entry in release["runs"].values():
        fitted_manifest = run_contract._json(
            root / entry["path"] / "inputs/split-manifest.json"
        )
        if set(frame.image_id) & {
            item for ids in fitted_manifest["partitions"].values() for item in ids
        }:
            raise ValueError("external labels overlap fitted partitions")
    output.mkdir(parents=True, exist_ok=False)
    status = {
        "schema_version": 2,
        "metrics_version": 2,
        "release_schema_version": 2,
        "fitted_identity": selected,
        "execution_environment": external_release.environment(),
        "status": "running",
        "run_id": run_id,
        "release_sha256": release_sha256,
        "manifest_sha256": manifest_sha256,
        "audit_sha256": audit_sha256,
        "cohort_sha256": manifest["cohort_sha256"],
        "device": "cpu" if record["pipeline"] == "classical_logistic" else device,
        "source": {name: release["files"][name] for name in release["source_files"]},
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
        validate_raw(record, raw, len(frame))
        predictions = external.prediction_frame(frame, raw, policy, run_id=run_id)
        predictions["cohort_sha256"] = manifest["cohort_sha256"]
        predictions.to_csv(output / "predictions.csv", index=False)
        np.save(output / "raw_scores.npy", raw)
        report = calibration.policy_report(
            policy,
            frame.target.to_numpy(dtype=int),
            raw,
            variances=raw.var(axis=0, ddof=1) if raw.ndim == 2 else None,
            metrics_version=2,
        )
        run_contract._write_json(output / "policy-report.json", report)
        external.verify_files(release_path, release_sha256, root)
        for path, digest in [
            (manifest_path, manifest_sha256),
            (audit_path, audit_sha256),
        ]:
            external_release.check_hash(Path(path), digest)
        if validate_manifest(run_contract._json(Path(manifest_path))) != manifest:
            raise ValueError("external manifest changed during inference")
        for row in manifest["rows"]:
            if row["image_sha256"]:
                external_release.check_hash(
                    images_dir / f"{row['image_id']}.jpg", row["image_sha256"]
                )
        pd.testing.assert_frame_equal(
            run_contract._read_csv(output / "predictions.csv"),
            predictions,
            check_dtype=False,
            check_exact=True,
        )
        if (
            not np.array_equal(
                np.load(output / "raw_scores.npy", allow_pickle=False), raw
            )
            or run_contract._json(output / "policy-report.json") != report
        ):
            raise ValueError("saved external scores or policy report changed")
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
    legacy=False,
):
    """Reconstruct saved external predictions from raw scores and frozen policy."""
    path, root = Path(run_path), Path(root)
    validate_manifest(manifest)
    status = run_contract._json(path / "run.json")
    if status.get("run_id") != expected_run_id:
        raise ValueError("external run identity mismatch")
    version = 1 if legacy else 2
    if (
        release.get("schema_version") != version
        or status.get("schema_version") != version
    ):
        raise ValueError("external release/result contract mismatch")
    record, _ = external_release.run_facts(
        root,
        expected_run_id,
        release["runs"][expected_run_id],
        release["files"],
        legacy=legacy,
    )
    if not legacy and (
        status.get("fitted_identity") != release["runs"][expected_run_id]
        or status.get("execution_environment")
        != run_contract._json(root / release["environment_path"])
        or status.get("metrics_version") != 2
        or status.get("release_schema_version") != 2
        or status.get("device") != release["runs"][expected_run_id]["device"]
        or status.get("external_labels_used_for_fitting") is not False
        or status.get("source")
        != {name: release["files"][name] for name in release["source_files"]}
    ):
        raise ValueError("external execution identity mismatch")
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
    validate_raw(record, raw, len(cohort))
    expected = external.prediction_frame(cohort, raw, policy, run_id=run_id)
    expected["cohort_sha256"] = manifest["cohort_sha256"]
    observed = run_contract._read_csv(path / "predictions.csv")
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
        metrics_version=version,
    )
    if report != run_contract._json(path / "policy-report.json"):
        raise ValueError("external policy report differs from frozen policy")
    return observed


def write_report(
    *,
    root,
    release_path,
    release_sha256,
    manifest_path,
    manifest_sha256,
    audit_path,
    audit_sha256,
    runs_dir,
    output_path=None,
    legacy_source_root=None,
    legacy_protocol_path=None,
    draws=None,
):
    """Accept the complete matrix once, with a separate labeled legacy replay."""
    root, runs_dir = Path(root), Path(runs_dir)
    legacy = legacy_source_root is not None
    if legacy and (output_path is None or legacy_protocol_path is None):
        raise ValueError(
            "legacy inspection requires historical source/protocol and new output"
        )
    output_path = (
        Path(output_path) if output_path else runs_dir / "external-report.json"
    )

    def verify():
        if legacy:
            release = external_release.inspect_legacy_release(
                release_path,
                release_sha256,
                root,
                historical_source_root=legacy_source_root,
                protocol_path=legacy_protocol_path,
            )
        else:
            release = external.verify_files(release_path, release_sha256, root)
        external_release.validate_matrix(release["runs"])
        external_release.check_hash(Path(manifest_path), manifest_sha256)
        validate_audit(audit_path, audit_sha256, manifest_sha256)
        manifest = validate_manifest(run_contract._json(Path(manifest_path)))
        return release, manifest

    release, manifest = verify()
    if legacy:
        draws = 2000 if draws is None else draws
    else:
        settings = release["reporting"]
        if draws is not None and draws != settings["draws"]:
            raise ValueError("report settings differ from the reviewed release")
        draws = settings["draws"]
    if {p.name for p in runs_dir.iterdir() if p.is_dir()} != set(release["runs"]):
        raise ValueError("missing or extra external run directories")
    models, registry, observed_hashes = {}, {}, {}
    for name, run in release["runs"].items():
        path = runs_dir / name
        before = {p.name: run_contract.sha256(p) for p in path.iterdir() if p.is_file()}
        frame = validate_predictions(
            path,
            release,
            release_sha256,
            manifest,
            manifest_sha256,
            root,
            expected_run_id=name,
            audit_sha256=audit_sha256,
            legacy=legacy,
        )
        models.setdefault(run["family"], {})[run["seed"]] = frame
        registry[name] = {
            "run_sha256": before["run.json"],
            "prediction_sha256": before["predictions.csv"],
        }
        observed_hashes[path] = before
    report = external.paired_report(
        models, draws=draws, metrics_version=1 if legacy else 2
    )
    retained = [row for row in manifest["rows"] if row["reason"] == "retained"]
    expected_summary = {
        "images": len(retained),
        "groups": len({row["group_id"] for row in retained}),
        "classes": {
            str(target): sum(row["target"] == target for row in retained)
            for target in (0, 1)
        },
    }
    if report.get("counts") != expected_summary:
        raise ValueError("external report counts differ from retained manifest rows")
    report.update(
        schema_version=2,
        contract="legacy-v1-inspection" if legacy else "strict-v2",
        strict_release_accepted=not legacy,
        release_sha256=release_sha256,
        manifest_sha256=manifest_sha256,
        audit_sha256=audit_sha256,
        cohort_sha256=manifest["cohort_sha256"],
        registry=registry,
        exclusions=manifest["counts"],
    )
    verify()
    for path, hashes in observed_hashes.items():
        for name, digest in hashes.items():
            external_release.check_hash(path / name, digest)
    with output_path.open("x") as handle:
        import json

        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report
