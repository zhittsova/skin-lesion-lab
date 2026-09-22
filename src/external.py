"""Evaluation-only contracts for independently sourced lesion cohorts."""

import hashlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src import (
    benchmark,
    calibration,
    cohort,
    external_release,
    run_contract,
    splitting,
)


def fingerprint(path, image_id):
    """Hash original bytes and RGB pixels, with rotation/flip pHash candidates."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    digest = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    reduced = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)

    def phash(small):
        values = cv2.dct(small.astype(np.float32))[:8, :8].ravel()
        bits = values > np.median(values[1:])
        return int.from_bytes(np.packbits(bits).tobytes(), "big")

    hashes = [
        phash(np.rot90(view, k).copy())
        for view in (reduced, reduced[:, ::-1])
        for k in range(4)
    ]
    return {
        "image_id": image_id,
        "image_sha256": run_contract.sha256(Path(path)),
        "rgb_sha256": digest,
        "phash": hashes[0],
        "phashes": hashes,
    }


def endpoints(frame, *, metrics_version=2):
    """Keep defined one-class scores while marking undefined rates as null."""
    if metrics_version == 2 or frame.target.nunique() == 2:
        return benchmark.endpoints(frame, metrics_version=metrics_version)
    if type(metrics_version) is not int or metrics_version != 1:
        raise ValueError("unsupported external metrics version")
    y = calibration.targets(frame.target.to_numpy(), len(frame))
    p = calibration.probabilities(frame.prob_melanoma.to_numpy())
    d = frame.prediction.to_numpy()
    if not len(frame) or not np.isin(d, [0, 1]).all():
        raise ValueError("invalid predictions")
    tp, tn = np.sum((y == 1) & (d == 1)), np.sum((y == 0) & (d == 0))
    fp, fn = np.sum((y == 0) & (d == 1)), np.sum((y == 1) & (d == 0))
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return {
        "roc_auc": None,
        "average_precision": None,
        "pr_auc": None,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        "average_cost": float((10 * fn + fp) / len(y)),
    }


def verify_files(release_path, expected_sha256, root):
    """Strict v2 acceptance; v1 releases have a separate inspection-only API."""
    return external_release.verify_release(release_path, expected_sha256, root)


def connected_groups(frame):
    """Link all source rows before filtering; permit mixed-outcome patients."""
    required = {"image_id", "lesion_id", "patient_id", "rgb_sha256"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("missing identity columns or empty cohort")
    rows = frame.sort_values("image_id").to_dict("records")
    ids = [r["image_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate image IDs")
    for column in ("image_id", "lesion_id", "patient_id"):
        if any(not isinstance(r[column], str) or not r[column].strip() for r in rows):
            raise ValueError(f"missing {column}")
    if (frame.groupby("lesion_id").patient_id.nunique() > 1).any():
        raise ValueError("a lesion belongs to conflicting patient IDs")
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in ("lesion_id", "patient_id", "rgb_sha256", "duplicate_cluster_id"):
            value = row.get(key, "")
            if not value:
                continue
            token = (key, value)
            if token in seen:
                parents[find(i)] = find(seen[token])
            else:
                seen[token] = i
    members = {}
    for i, row in enumerate(rows):
        members.setdefault(find(i), []).append(row["image_id"])
    return {image: min(group) for group in members.values() for image in group}


def hiba_manifest(metadata_path, images_dir):
    """Normalize the pinned HIBA DOI export without fitting or repartitioning."""
    source = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    required = {
        "isic_id",
        "lesion_id",
        "patient_id",
        "diagnosis",
        "diagnosis_1",
        "diagnosis_2",
        "diagnosis_3",
        "image_type",
        "dermoscopic_type",
        "copyright_license",
        "attribution",
    }
    if source.empty or not required.issubset(source.columns):
        raise ValueError("empty HIBA metadata or missing columns")
    labels = {
        "melanoma": (
            "mel",
            1,
            "Malignant",
            "Malignant melanocytic proliferations (Melanoma)",
            "Melanoma, NOS",
        ),
        "nevus": ("nv", 0, "Benign", "Benign melanocytic proliferations", "Nevus"),
        "seborrheic keratosis": (
            "bkl",
            0,
            "Benign",
            "Benign epidermal proliferations",
            "Seborrheic keratosis",
        ),
        "solar lentigo": (
            "bkl",
            0,
            "Benign",
            "Benign epidermal proliferations",
            "Solar lentigo",
        ),
        "lichenoid keratosis": (
            "bkl",
            0,
            "Benign",
            "Benign epidermal proliferations",
            "Lichen planus like keratosis",
        ),
        "dermatofibroma": (
            "df",
            0,
            "Benign",
            "Benign soft tissue proliferations - Fibro-histiocytic",
            "Dermatofibroma",
        ),
        "vascular lesion": (
            "vasc",
            0,
            "Benign",
            "Benign soft tissue proliferations - Vascular",
            "",
        ),
        "basal cell carcinoma": (
            "bcc",
            None,
            "Malignant",
            "Malignant adnexal epithelial proliferations - Follicular",
            "Basal cell carcinoma",
        ),
        "squamous cell carcinoma": (
            "scc",
            None,
            "Malignant",
            "Malignant epidermal proliferations",
            "Squamous cell carcinoma, NOS",
        ),
        "actinic keratosis": (
            "akiec",
            None,
            "Indeterminate",
            "Indeterminate epidermal proliferations",
            "Solar or actinic keratosis",
        ),
    }
    rows = []
    images_dir = Path(images_dir).resolve()
    lesion_labels = {}
    for item in source.sort_values("isic_id").to_dict("records"):
        image_id = cohort._validate_id(item["isic_id"], "image_id")
        lesion = cohort._validate_id(item["lesion_id"], "lesion_id")
        patient = cohort._validate_id(item["patient_id"], "patient_id")
        if (
            item["copyright_license"] != "CC-BY"
            or item["attribution"] != "Hospital Italiano de Buenos Aires"
        ):
            raise ValueError("unreviewed source license or attribution")
        diagnosis = item["diagnosis"]
        definition = labels.get(diagnosis)
        if definition:
            code, target, *hierarchy = definition
            if hierarchy != [item[f"diagnosis_{i}"] for i in (1, 2, 3)]:
                raise ValueError(f"conflicting diagnosis: {image_id}")
            reason = (
                "retained"
                if target is not None
                else "excluded_indeterminate"
                if code == "akiec"
                else "excluded_malignancy"
            )
        else:
            code, target, reason = "unknown", None, "unknown_diagnosis"
        if lesion in lesion_labels and lesion_labels[lesion] != diagnosis:
            raise ValueError("conflicting lesion diagnosis")
        lesion_labels[lesion] = diagnosis
        if item["image_type"] not in {
            "dermoscopic",
            "clinical: overview",
            "clinical: close-up",
        }:
            raise ValueError("unknown modality")
        if (
            item["image_type"] == "dermoscopic"
            and item["dermoscopic_type"] != "contact polarized"
        ):
            raise ValueError("unreviewed dermoscopy acquisition type")
        if item["image_type"] != "dermoscopic":
            reason = "excluded_modality"
        path = images_dir / f"{image_id}.jpg"
        if not path.resolve().is_relative_to(images_dir):
            raise ValueError("unsafe image path")
        fingerprints = {
            "image_id": image_id,
            "image_sha256": "",
            "rgb_sha256": "",
            "phash": None,
            "phashes": [],
        }
        if not path.is_file():
            if reason == "retained":
                reason = "missing_image"
        else:
            try:
                fingerprints = fingerprint(path, image_id)
            except OSError, ValueError:
                if reason == "retained":
                    reason = "corrupt_image"
        rows.append(
            {
                **fingerprints,
                "lesion_id": lesion,
                "patient_id": patient,
                "diagnosis": diagnosis,
                "dx": code,
                "target": target,
                "reason": reason,
                "modality": item["image_type"],
                "duplicate_cluster_id": item.get("duplicate_cluster_id", ""),
            }
        )
    fingerprints = pd.DataFrame(rows)
    present = fingerprints[fingerprints.rgb_sha256 != ""]
    if (present.groupby("rgb_sha256").diagnosis.nunique() > 1).any():
        raise ValueError("duplicate pixels have conflicting diagnoses")
    groups = connected_groups(fingerprints)
    seen = set()
    for row in rows:
        row["group_id"] = groups[row["image_id"]]
        if row["reason"] == "retained":
            if row["rgb_sha256"] in seen:
                row["reason"] = "duplicate_pixels"
            else:
                seen.add(row["rgb_sha256"])
    manifest = {
        "schema_version": 1,
        "purpose": "external",
        "source": "hiba-10.34970-587329",
        "label_policy_version": splitting.LABEL_POLICY_VERSION,
        "metadata_sha256": run_contract.sha256(Path(metadata_path)),
        "rows": rows,
        "counts": {"input": len(rows), **Counter(r["reason"] for r in rows)},
    }
    manifest["cohort_sha256"] = splitting.canonical_hash(manifest)
    return manifest


def overlap_audit(rows, reference, *, max_distance=4):
    """Compare global archive IDs, bytes, decoded pixels and pHash candidates.

    Lesion/patient IDs are cohort-local and deliberately not matched across
    collections. A clear result covers only these image-overlap detectors.
    """
    if not rows or not reference:
        raise ValueError("overlap audit requires both complete inventories")
    indexes = {
        key: {r[key] for r in reference if r.get(key)}
        for key in ("image_id", "image_sha256", "rgb_sha256")
    }
    exact, near = [], []
    for row in sorted(rows, key=lambda r: r["image_id"]):
        reasons = [key for key, values in indexes.items() if row.get(key) in values]
        if reasons:
            exact.append({"image_id": row["image_id"], "matches": reasons})
            continue
        for other in reference:
            distance = min(
                (int(h) ^ int(other["phash"])).bit_count() for h in row["phashes"]
            )
            if distance <= max_distance:
                near.append(
                    {
                        "image_id": row["image_id"],
                        "reference_id": other["image_id"],
                        "distance": distance,
                    }
                )
    return {
        "exact": exact,
        "near": near,
        "clear": not exact and not near,
        "external_images": len(rows),
        "reference_images": len(reference),
        "max_distance": max_distance,
    }


def component_draws(frame, *, draws=2000, seed=2026):
    """Unstratified patient-component draws, including all member images."""
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if frame.empty or frame.image_id.duplicated().any():
        raise ValueError("empty cohort or duplicate image IDs")
    for key in ("image_id", "group_id"):
        if not frame[key].map(lambda x: isinstance(x, str) and bool(x)).all():
            raise ValueError("invalid image or group ID")
    groups = sorted(frame.group_id.unique())
    members = {g: np.flatnonzero(frame.group_id.to_numpy() == g) for g in groups}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        yield np.concatenate(
            [members[g] for g in rng.choice(groups, len(groups), replace=True)]
        )


def paired_report(
    models, *, reference="logistic", draws=2000, seed=2026, metrics_version=2
):
    """Paired image metrics with mixed-label patient components kept intact."""
    if not models or reference not in models:
        raise ValueError("missing reference model")
    aligned, identity = {}, None
    for model, runs in sorted(models.items()):
        if set(runs) != set(benchmark.SEEDS):
            raise ValueError("exactly seeds 17, 42, 73 are required")
        aligned[model] = {}
        for s in benchmark.SEEDS:
            frame = runs[s].sort_values("image_id").reset_index(drop=True)
            benchmark._arrays(frame)
            if frame.image_id.duplicated().any():
                raise ValueError("duplicate prediction IDs")
            current = frame[["image_id", "group_id", "target"]]
            if identity is not None and not identity.equals(current):
                raise ValueError("prediction alignment mismatch")
            identity = current
            aligned[model][s] = frame
    cohort = aligned[reference][benchmark.SEEDS[0]]
    points = {
        m: {s: endpoints(f, metrics_version=metrics_version) for s, f in runs.items()}
        for m, runs in aligned.items()
    }
    metrics = tuple(points[reference][benchmark.SEEDS[0]])
    samples = {m: {k: [] for k in metrics} for m in aligned}
    differences = {m: {k: [] for k in metrics} for m in aligned if m != reference}
    for indices in component_draws(cohort, draws=draws, seed=seed):
        values = {
            m: {
                s: endpoints(f.iloc[indices], metrics_version=metrics_version)
                for s, f in runs.items()
            }
            for m, runs in aligned.items()
        }
        for m in aligned:
            for k in metrics:
                samples[m][k].append(
                    benchmark._mean([values[m][s][k] for s in benchmark.SEEDS])
                )
                if m != reference:
                    differences[m][k].append(
                        benchmark._mean(
                            [
                                None
                                if values[m][s][k] is None
                                or values[reference][s][k] is None
                                else values[m][s][k] - values[reference][s][k]
                                for s in benchmark.SEEDS
                            ]
                        )
                    )
    return {
        "metrics_version": metrics_version,
        "purpose": "external",
        "seeds": list(benchmark.SEEDS),
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "unit": "patient_lesion_duplicate_component",
            "method": "unstratified_paired_percentile",
            "confidence": 0.95,
            "undefined_policy": "null_interval_if_any_draw_undefined",
        },
        "counts": {
            "images": len(cohort),
            "groups": int(cohort.group_id.nunique()),
            "classes": {str(y): int(sum(cohort.target == y)) for y in (0, 1)},
        },
        "reference": reference,
        "models": {
            m: {
                k: benchmark._summary(
                    [points[m][s][k] for s in benchmark.SEEDS], samples[m][k]
                )
                for k in metrics
            }
            for m in aligned
        },
        "differences": {
            m: {
                k: benchmark._summary(
                    [
                        None
                        if points[m][s][k] is None or points[reference][s][k] is None
                        else points[m][s][k] - points[reference][s][k]
                        for s in benchmark.SEEDS
                    ],
                    differences[m][k],
                )
                for k in metrics
            }
            for m in differences
        },
    }


def prediction_frame(cohort, raw_scores, policy, *, run_id):
    """Apply a saved policy; labels are carried through only for reporting."""
    scores = np.asarray(raw_scores)
    if scores.ndim not in (1, 2) or scores.shape[-1] != len(cohort):
        raise ValueError("prediction row count mismatch")
    if (
        cohort.empty
        or cohort.image_id.duplicated().any()
        or not cohort.target.isin([0, 1]).all()
    ):
        raise ValueError("invalid prediction identity or targets")
    variances = scores.var(axis=0, ddof=1) if scores.ndim == 2 else None
    applied = calibration.apply_policy(policy, scores, variances=variances)
    result = cohort[["image_id", "group_id", "target"]].copy()
    if policy["schema_version"] == 2:
        result["calibration_score"] = scores
        result["ranking_score"] = applied["ranking_score"]
    result["schema_version"] = policy["schema_version"]
    result["role"] = "external"
    result["run_id"] = run_id
    result["prob_melanoma"] = applied["calibrated_probability"]
    result["prediction"] = applied["prediction"]
    result["review_recommended"] = applied["review_recommended"]
    return result
