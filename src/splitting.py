"""Frozen development partitions linked by lesion, patient and known duplicates."""

import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np

ROLES = ("train", "selection", "calibration", "development")
FRACTIONS = (0.6, 0.15, 0.10, 0.15)
PROTOCOL_VERSION = "development-v1"
LABEL_POLICY_VERSION = "melanoma-selected-benign-v1"


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def metadata_hash(frame):
    """Hash all source cells independently of row and column order."""
    columns = sorted(frame.columns)
    rows = sorted(frame[columns].astype(str).values.tolist())
    return canonical_hash({"columns": columns, "rows": rows})


def _identifier(value, name, optional=False):
    if optional and value == "":
        return ""
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", value
    ):
        raise ValueError(f"invalid or missing {name}")
    return value


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid SHA-256")
    return value


def _records(frame):
    required = {"isic_id", "lesion_id", "target", "dx", "image_sha256"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("empty cohort or missing manifest columns")
    rows = []
    for row in frame.to_dict("records"):
        if isinstance(row["target"], bool) or row["target"] not in (0, 1):
            raise ValueError("expected binary labels")
        rows.append(
            {
                "image_id": _identifier(row["isic_id"], "image ID"),
                "lesion_id": _identifier(row["lesion_id"], "lesion ID"),
                "patient_id": _identifier(
                    row.get("patient_id", ""), "patient ID", True
                ),
                "duplicate_cluster_id": _identifier(
                    row.get("duplicate_cluster_id", ""), "duplicate cluster ID", True
                ),
                "image_sha256": _sha(row["image_sha256"]),
                "diagnosis": _identifier(row["dx"], "diagnosis"),
                "target": int(row["target"]),
            }
        )
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate image IDs")
    patients = [bool(r["patient_id"]) for r in rows]
    if any(patients) and not all(patients):
        raise ValueError("partial patient IDs are unsupported; resolve missing linkage")
    return sorted(rows, key=lambda r: r["image_id"])


def _components(rows):
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in ("lesion_id", "patient_id", "duplicate_cluster_id", "image_sha256"):
            value = row[key]
            if not value:
                continue
            token = (key, value)
            if token in seen:
                parents[root(i)] = root(seen[token])
            else:
                seen[token] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(root(i), []).append(row)
    result = []
    for members in groups.values():
        if len({row["target"] for row in members}) != 1:
            raise ValueError(
                "mixed labels in linked group; mixed-outcome patients are unsupported"
            )
        result.append(members)
    return sorted(result, key=lambda members: members[0]["image_id"])


def _allocate(groups, fractions, seed, roles):
    ratios = np.asarray(fractions, dtype=float)
    if (
        ratios.shape != (len(roles),)
        or not np.isfinite(ratios).all()
        or np.any(ratios <= 0)
        or not np.isclose(ratios.sum(), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("fractions must be finite, positive and sum to one")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    parts = {role: [] for role in roles}
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        selected = [group for group in groups if group[0]["target"] == label]
        n = len(selected)
        if n < len(roles):
            raise ValueError(f"insufficient independent groups for class {label}")
        # Closest integer quotas with at least one group of each class per role.
        counts = np.ones(len(roles), dtype=int)
        for _ in range(n - len(roles)):
            counts[int(np.argmax(n * ratios - counts))] += 1
        offset = 0
        order = rng.permutation(n)
        for role, count in zip(roles, counts):
            for index in order[offset : offset + count]:
                parts[role].extend(row["image_id"] for row in selected[index])
            offset += count
    return {role: sorted(ids) for role, ids in parts.items()}


def create_manifest(frame, *, fractions=FRACTIONS, seed=42):
    """Create a development manifest. Confirmation cohorts are not accepted here.

    Call with a freshly prepared cohort so image digests reflect current bytes.
    Ratios target independent groups within each class, not image counts.
    """
    rows = _records(frame)
    source = frame.attrs.get("source")
    if source not in {"ham10000", "isic2018_task3"}:
        raise ValueError("unsupported development source")
    if frame.attrs.get("label_policy_version") != LABEL_POLICY_VERSION:
        raise ValueError("unsupported label policy")
    metadata = _sha(frame.attrs.get("metadata_content_hash"))
    groups = _components(rows)
    parts = _allocate(groups, fractions, seed, ROLES)
    payload = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "development",
        "source": source,
        "label_policy_version": LABEL_POLICY_VERSION,
        "seed": seed,
        "fractions": dict(zip(ROLES, map(float, fractions))),
        "metadata_content_hash": metadata,
        "cohort_hash": canonical_hash(
            {
                "source": source,
                "policy": LABEL_POLICY_VERSION,
                "metadata": metadata,
                "rows": rows,
            }
        ),
        "grouping": "patient+lesion+known-duplicates"
        if rows[0]["patient_id"]
        else "lesion+known-duplicates",
        "groups": [
            {
                "group_id": members[0]["image_id"],
                "image_ids": [r["image_id"] for r in members],
            }
            for members in groups
        ],
        "rows": rows,
        "partitions": parts,
    }
    payload["split_hash"] = canonical_hash(payload)
    return payload


def manifest_indices(frame, manifest):
    """Validate content, linkage, purpose and deterministic membership before use."""
    try:
        expected = create_manifest(
            frame,
            fractions=[manifest["fractions"][role] for role in ROLES],
            seed=manifest["seed"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid manifest") from exc
    if manifest != expected:
        raise ValueError("manifest mismatch: cohort, protocol or membership changed")
    by_id = {image_id: i for i, image_id in enumerate(frame.isic_id)}
    return {
        role: np.array([by_id[image_id] for image_id in ids], dtype=int)
        for role, ids in manifest["partitions"].items()
    }


def save_manifest(manifest, path):
    """Never overwrite an existing allocation, including a damaged file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def load_manifest(frame, path):
    manifest = json.loads(Path(path).read_text())
    manifest_indices(frame, manifest)
    return manifest


def load_development_split(frame, path):
    """Shared entry point; training seeds and architectures cannot reassign rows."""
    manifest = load_manifest(frame, path)
    # The comparison protocol has one allocation; alternate ratios require a new version.
    if manifest["seed"] != 42 or manifest["fractions"] != dict(zip(ROLES, FRACTIONS)):
        raise ValueError("manifest differs from frozen development-v1 allocation")
    indices = manifest_indices(frame, manifest)
    report = get_split_report(
        frame.target.to_numpy(), indices, frame.lesion_id.to_numpy()
    )
    report.update(
        {
            key: manifest[key]
            for key in (
                "split_hash",
                "cohort_hash",
                "metadata_content_hash",
                "protocol_version",
                "purpose",
                "grouping",
            )
        }
    )
    membership = {
        image: role for role, ids in manifest["partitions"].items() for image in ids
    }
    report["independent_groups"] = {role: 0 for role in ROLES}
    for group in manifest["groups"]:
        report["independent_groups"][membership[group["image_ids"][0]]] += 1
    report["known_duplicate_overlap"] = (
        0  # Full reconstruction above verifies connected components.
    )
    return indices, report


def stratified_lesion_split(
    lesion_ids, labels, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42
):
    """Compatibility helper for legacy callers; never used for protocol runs."""
    if len(lesion_ids) != len(labels):
        raise ValueError("length mismatch")
    rows = []
    for i, (group, label) in enumerate(zip(lesion_ids, labels)):
        if label not in (0, 1):
            raise ValueError("expected binary labels")
        rows.append(
            {
                "image_id": f"I{i:09d}",
                "lesion_id": _identifier(group, "lesion ID"),
                "patient_id": "",
                "duplicate_cluster_id": "",
                "image_sha256": str(i),
                "target": label,
            }
        )
    parts = _allocate(
        _components(rows),
        (train_size, val_size, test_size),
        random_state,
        ("train", "val", "test"),
    )
    return tuple(np.array([int(image[1:]) for image in ids]) for ids in parts.values())


def split_dataset(
    lesion_ids, labels, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42
):
    return dict(
        zip(
            ("train", "val", "test"),
            stratified_lesion_split(
                lesion_ids, labels, train_size, val_size, test_size, random_state
            ),
        )
    )


def get_split_statistics(labels, split_indices):
    stats = {}
    for role, indices in split_indices.items():
        y = labels[indices]
        stats[role] = {
            "benign": int(np.sum(y == 0)),
            "melanoma": int(np.sum(y == 1)),
            "total": len(y),
            "melanoma_ratio": float(np.mean(y)) if len(y) else 0.0,
        }
    return stats


def get_split_report(labels, split_indices, lesion_ids):
    if len(labels) != len(lesion_ids):
        raise ValueError("length mismatch")
    arrays = [np.asarray(indices) for indices in split_indices.values()]
    if not arrays or any(a.ndim != 1 or a.dtype.kind not in "iu" for a in arrays):
        raise ValueError("invalid split indices")
    combined = np.concatenate(arrays)
    if sorted(combined.tolist()) != list(range(len(labels))):
        raise ValueError("split coverage or row overlap failure")
    stats = get_split_statistics(labels, split_indices)
    groups = {role: set(lesion_ids[indices]) for role, indices in split_indices.items()}
    for role in stats:
        stats[role]["unique_lesions"] = len(groups[role])
    overlaps = {
        f"{a}_{b}": len(groups[a] & groups[b])
        for a, b in itertools.combinations(groups, 2)
    }
    if any(overlaps.values()):
        raise ValueError("lesion overlap between splits")
    return {"splits": stats, "lesion_overlap_counts": overlaps, "leakage_free": True}


def print_split_summary(labels, split_indices, lesion_ids):
    for role, stats in get_split_report(labels, split_indices, lesion_ids)[
        "splits"
    ].items():
        print(
            f"{role}: {stats['total']} images, {stats['melanoma']} melanoma, {stats['unique_lesions']} lesions"
        )
