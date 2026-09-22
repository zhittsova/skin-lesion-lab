"""Versioned release coverage and fitted identities for external evaluation."""

import importlib.metadata
import os
import pickle
import platform
import re
from pathlib import Path

import numpy as np

from src import calibration, run_contract, splitting

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("logistic", "efficientnet-full-unweighted")
SEEDS = (17, 42, 73)
ENVIRONMENT_FILES = ("pyproject.toml", "uv.lock", ".python-version")
PROTOCOL_FILES = ("docs/evaluation-protocol.md",)
PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "torch",
    "torchvision",
    "pillow",
    "opencv-python-headless",
    "matplotlib",
    "seaborn",
    "tqdm",
)
ENVIRONMENT_KEYS = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTORCH_ENABLE_MPS_FALLBACK",
)


def source_files(root=ROOT):
    # Deliberately a superset: adding a transitive src module changes the inventory.
    return sorted(
        [str(p.relative_to(root)) for p in (root / "src").rglob("*.py")]
        + ["scripts/evaluate_external.py"]
    )


IMPORTED_SOURCE = {name: run_contract.sha256(ROOT / name) for name in source_files()}


def environment():
    """Execution inputs checked independently of the fitted training environment."""
    import cv2
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "software": {name: importlib.metadata.version(name) for name in PACKAGES},
        "variables": {name: os.environ.get(name) for name in ENVIRONMENT_KEYS},
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "opencv_threads": cv2.getNumThreads(),
        "opencv_optimized": cv2.useOptimized(),
    }


def checked_path(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("invalid release path")
    relative = Path(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != name
        or name == "."
        or not (root / relative).resolve().is_relative_to(root.resolve())
    ):
        raise ValueError("unsafe or noncanonical release path")
    return root / relative


def check_hash(path, digest):
    if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError("malformed SHA-256 digest")
    if not path.is_file() or run_contract.sha256(path) != digest:
        raise ValueError(f"missing file or artifact hash mismatch: {path}")


def estimator_name(record):
    pipeline = record["pipeline"]
    if pipeline.startswith("deep_"):
        return f"models/{record['config']['architecture']}_mc_dropout.pt"
    family = pipeline.removeprefix("classical_")
    return (
        "models/bayesian_gmm_model.pkl"
        if family == "gmm"
        else f"models/{family}_model.pkl"
    )


def run_facts(root, name, entry, files, *, legacy=False):
    """Verify required references before any estimator deserialization."""
    run_dir = checked_path(root, entry["path"])
    record_path = f"{entry['path']}/run.json"
    if record_path not in files:
        raise ValueError("release omits fitted run record")
    check_hash(run_dir / "run.json", entry["run_sha256"])
    record = run_contract._json(run_dir / "run.json")
    if name != record.get("run_id") or run_dir.name != name:
        raise ValueError("fitted run ID/path mismatch")
    artifacts = record.get("artifacts", {})
    required = {"models/decision_policy.json", "inputs/split-manifest.json"}
    if record["pipeline"].startswith("deep_"):
        required.add("results/deep_training_metadata.json")
    required.add(estimator_name(record))
    if not required.issubset(artifacts):
        raise ValueError("fitted record omits estimator/policy/preprocessing/split")
    for relative, digest in artifacts.items():
        path = checked_path(run_dir, relative)
        if files.get(f"{entry['path']}/{relative}") != digest:
            raise ValueError(
                f"release omits or disagrees with fitted artifact: {relative}"
            )
        check_hash(path, digest)
    training = (
        run_contract._json(run_dir / "results/deep_training_metadata.json")
        if record["pipeline"].startswith("deep_")
        else None
    )
    facts = run_contract.fitted_identity(record, training)
    keys = ("family", "seed", "config_sha256") if legacy else tuple(facts)
    if not legacy and set(entry) != set(facts) | {
        "path",
        "run_sha256",
        "estimator",
        "policy",
        "preprocessing",
        "device",
    }:
        raise ValueError("invalid release run entry schema")
    if any(entry.get(key) != facts[key] for key in keys):
        raise ValueError("release mapping differs from fitted identity")
    if not record.get("environment", {}).get("software"):
        raise ValueError("missing fitted environment")
    policy = run_contract._json(run_dir / "models/decision_policy.json")
    manifest = run_contract._json(run_dir / "inputs/split-manifest.json")
    if (
        splitting.canonical_hash(
            {k: v for k, v in manifest.items() if k != "split_hash"}
        )
        != manifest["split_hash"]
        or policy.get("schema_version") != facts["policy_version"]
        or policy["fit"]["role"] != "calibration"
        or policy["fit"]["image_ids"] != manifest["partitions"]["calibration"]
        or policy["fit"]["split_hash"] != record["split_hash"]
        or manifest["split_hash"] != record["split_hash"]
        or policy["score_transform"]["positive_weight"] != facts["positive_weight"]
    ):
        raise ValueError("fitted policy/partition/loss identity mismatch")
    if training:
        estimator = estimator_name(record)
        if (
            training["checkpoint"]["path"] != estimator
            or training["checkpoint"]["sha256"] != artifacts[estimator]
            or policy["model"]["architecture"] != facts["architecture"]
            or policy["model"]["checkpoint_sha256"] != artifacts[estimator]
            or policy["referral"]["mc_passes"] != record["config"]["mc_samples"]
        ):
            raise ValueError("fitted checkpoint/policy reference mismatch")
    if not legacy:
        decision = policy["decision"]
        if (
            decision["comparison"] != ">="
            or not np.isfinite(decision["threshold"])
            or not 0 <= decision["threshold"] <= np.nextafter(1.0, np.inf)
            or decision["formula_threshold"]
            != calibration.cost_threshold(decision["cost_fn"], decision["cost_fp"])
        ):
            raise ValueError("invalid fitted decision policy")
        probe = np.array([0.25, 0.75])
        if training:
            config = record["config"]
            if any(
                type(config.get(key)) is not int or config[key] < minimum
                for key, minimum in (
                    ("image_size", 1),
                    ("batch_size", 1),
                    ("mc_samples", 2),
                )
            ):
                raise ValueError("invalid fitted inference configuration")
            probe = np.array([probe, probe])
        calibration.apply_policy(
            policy, probe, variances=probe.var(axis=0, ddof=1) if training else None
        )
        estimator = f"{entry['path']}/{estimator_name(record)}"
        preprocessing = (
            {
                "kind": "transforms",
                "artifact": f"{entry['path']}/results/deep_training_metadata.json",
            }
            if training
            else {"kind": "embedded", "artifact": estimator}
        )
        if (
            entry.get("estimator") != estimator
            or entry.get("policy") != f"{entry['path']}/models/decision_policy.json"
            or entry.get("preprocessing") != preprocessing
            or entry.get("device") not in {"cpu", "mps", "cuda"}
            or (not training and entry["device"] != "cpu")
        ):
            raise ValueError("invalid estimator/policy/preprocessing/device reference")
    return record, facts


def check_estimator(root, entry, record, facts):
    """Inspect trusted payload metadata only after all release mappings pass."""
    from src import deep

    run_dir = root / entry["path"]
    path = run_dir / estimator_name(record)
    config = record["config"]
    policy = run_contract._json(run_dir / "models/decision_policy.json")
    if facts["architecture"] is not None:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        for key in (
            "architecture",
            "seed",
            "image_size",
            "dropout",
            "pretrained",
            "fine_tune_backbone",
            "loss_strategy",
        ):
            if checkpoint.get(key) != config.get(key):
                raise ValueError(f"checkpoint identity mismatch: {key}")
        weight = checkpoint.get("pos_weight")
        expected = (
            None if facts["loss_strategy"] == "unweighted" else facts["positive_weight"]
        )
        if weight != expected or not checkpoint.get("model_state_dict"):
            raise ValueError("checkpoint loss weight or model state mismatch")
        training = run_contract._json(run_dir / "results/deep_training_metadata.json")
        if training["transforms"]["eval"] != repr(
            deep.build_transforms(config["image_size"], train=False)
        ):
            raise ValueError("fitted preprocessing differs from execution transforms")
    else:
        with path.open("rb") as handle:
            saved = pickle.load(handle)
        if (
            saved.get("model_kind") != facts["family"]
            or saved.get("decision_policy") != policy
        ):
            raise ValueError("estimator/policy identity mismatch")
        fitted = saved["fitted"]
        if facts["family"] == "logistic":
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            if (
                not isinstance(fitted.get("model"), LogisticRegression)
                or not isinstance(fitted.get("scaler"), StandardScaler)
                or fitted["model"].random_state != facts["seed"]
                or fitted["model"].n_features_in_ != 128
                or fitted["scaler"].n_features_in_ != 128
            ):
                raise ValueError("fitted estimator seed or preprocessing mismatch")
        elif facts["family"] == "gmm":
            from sklearn.mixture import GaussianMixture

            if (
                set(fitted) != {0, 1}
                or any(
                    not isinstance(model, GaussianMixture)
                    or model.random_state != facts["seed"] + label
                    or model.n_features_in_ != 128
                    for label, model in fitted.items()
                )
                or any(
                    np.asarray(saved[key]).shape != (128,)
                    for key in ("train_mean", "train_std")
                )
            ):
                raise ValueError("fitted GMM seed or preprocessing mismatch")
    for key in ("split_hash", "cohort_hash"):
        payload = checkpoint if facts["architecture"] else saved
        if key in record and payload.get(key) != record[key]:
            raise ValueError(f"estimator fitted {key} mismatch")


def validate_matrix(runs):
    pairs = [(run["family"], run["seed"]) for run in runs.values()]
    if len(pairs) != len(set(pairs)) or set(pairs) != {
        (family, seed) for family in FAMILIES for seed in SEEDS
    }:
        raise ValueError(
            "report requires the complete unique two-family/three-seed matrix"
        )


def validate_release(release, root):
    root = Path(root)
    if type(release.get("schema_version")) is not int or release["schema_version"] != 2:
        raise ValueError(
            "strict execution requires release schema 2; legacy is inspection only"
        )
    files = release.get("files", {})
    if not isinstance(files, dict) or not files:
        raise ValueError("empty release file inventory")
    resolved = [checked_path(root, name).resolve() for name in files]
    if len(set(resolved)) != len(resolved):
        raise ValueError("repeated release file reference")
    for name, digest in files.items():
        check_hash(checked_path(root, name), digest)
    source = release.get("source_files")
    if source != source_files(ROOT):
        raise ValueError("incomplete or repeated execution source inventory")
    required = set(source) | set(ENVIRONMENT_FILES) | set(PROTOCOL_FILES)
    protocol = release.get("protocol", {})
    env_path = release.get("environment_path")
    required.update([protocol.get("path"), env_path])
    if not required.issubset(files):
        raise ValueError("release omits source/protocol/environment input")
    if protocol.get("sha256") != files[protocol["path"]]:
        raise ValueError("protocol digest mismatch")
    for name in source + list(ENVIRONMENT_FILES) + list(PROTOCOL_FILES):
        check_hash(ROOT / name, files[name])
        if name in IMPORTED_SOURCE and files[name] != IMPORTED_SOURCE[name]:
            raise ValueError("execution source changed since import")
    if run_contract._json(root / env_path) != environment():
        raise ValueError("execution environment mismatch")
    if release.get("comparison") != {"families": list(FAMILIES), "seeds": list(SEEDS)}:
        raise ValueError("unsupported external comparison protocol")
    settings = release.get("reporting", {})
    if (
        set(settings) != {"draws", "seed", "reference", "metrics_version"}
        or type(settings.get("draws")) is not int
        or settings["draws"] < 1
        or settings.get("seed") != 2026
        or settings.get("reference") != "logistic"
        or settings.get("metrics_version") != 2
    ):
        raise ValueError("invalid frozen report settings")
    runs = release.get("runs", {})
    if not isinstance(runs, dict) or not runs:
        raise ValueError("missing release runs")
    seen_paths, seen_pairs = set(), set()
    checked = []
    for name, entry in runs.items():
        record, facts = run_facts(root, name, entry, files)
        pair = (facts["family"], facts["seed"])
        path = checked_path(root, entry["path"]).resolve()
        if path in seen_paths or pair in seen_pairs:
            raise ValueError("repeated fitted run reference or family/seed")
        seen_paths.add(path)
        seen_pairs.add(pair)
        checked.append((entry, record, facts))
    for entry, record, facts in checked:
        check_estimator(root, entry, record, facts)
    return release


def verify_release(path, digest, root):
    check_hash(Path(path), digest)
    try:
        return validate_release(run_contract._json(Path(path)), root)
    except (
        KeyError,
        TypeError,
        AttributeError,
        OSError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        raise ValueError("invalid release structure or fitted artifact") from error


def inspect_legacy_release(
    path, digest, root, *, historical_source_root, protocol_path
):
    """Check v1 listed bytes and actual mappings; never authorize new inference.

    Source files come from an explicitly supplied historical snapshot, because
    current corrected code is not the incompletely frozen historical program.
    """
    root, source_root = Path(root), Path(historical_source_root)
    check_hash(Path(path), digest)
    release = run_contract._json(Path(path))
    if type(release.get("schema_version")) is not int or release["schema_version"] != 1:
        raise ValueError("legacy inspection requires schema 1")
    check_hash(Path(protocol_path), release["protocol_sha256"])
    for name, value in release["files"].items():
        base = source_root if name.startswith("src/") else root
        check_hash(checked_path(base, name), value)
    for name, entry in release["runs"].items():
        run_facts(root, name, entry, release["files"], legacy=True)
    validate_matrix(release["runs"])
    return release
