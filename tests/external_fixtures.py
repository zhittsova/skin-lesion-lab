"""Small, generated fitted artifacts and explicit independent release declarations."""

import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from src import calibration, classical, deep, external_release, splitting, uncertainty

SOURCE = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class ReleaseFixture:
    def __init__(self, root, identities=(("logistic", 17),), *, policy_version=1):
        self.root = root
        self.release = {
            "schema_version": 2,
            "reporting": {
                "draws": 5,
                "seed": 2026,
                "reference": "logistic",
                "metrics_version": 2,
            },
            "comparison": {
                "families": ["logistic", "efficientnet-full-unweighted"],
                "seeds": [17, 42, 73],
            },
            "runs": {},
            "files": {},
            "source_files": sorted(
                [str(p.relative_to(SOURCE)) for p in (SOURCE / "src").rglob("*.py")]
                + ["scripts/evaluate_external.py"]
            ),
            "environment_path": "environment.json",
        }
        for name in self.release["source_files"] + [
            "pyproject.toml",
            "uv.lock",
            ".python-version",
            "docs/evaluation-protocol.md",
        ]:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / name, target)
            self.pin(name)
        write(root / "environment.json", external_release.environment())
        self.pin("environment.json")
        (root / "protocol.md").write_text(
            "Generated external protocol fixture. No measured results.\n"
        )
        self.pin("protocol.md")
        self.release["protocol"] = {
            "path": "protocol.md",
            "sha256": digest(root / "protocol.md"),
        }
        for family, seed in identities:
            self.add_run(family, seed, policy_version=policy_version)
        ids = ["001", "000", "NA", "NULL", "7", "mixed_A"]
        rows = []
        for i, name in enumerate(ids):
            Image.new("RGB", (32, 32), (25 + i * 30, 35, 70)).save(root / f"{name}.jpg")
            rows.append(
                {
                    "image_id": name,
                    "group_id": ids[i],
                    "target": i % 2,
                    "reason": "retained",
                    "image_sha256": digest(root / f"{name}.jpg"),
                }
            )
        self.manifest = {
            "schema_version": 1,
            "purpose": "external",
            "rows": rows,
            "counts": {"input": len(rows), "retained": len(rows)},
        }
        self.manifest["cohort_sha256"] = splitting.canonical_hash(self.manifest)
        write(root / "manifest.json", self.manifest)
        write(
            root / "audit.json",
            {
                "clear": True,
                "manifest_sha256": digest(root / "manifest.json"),
                "exact": [],
                "near": [],
            },
        )
        self.save()

    def pin(self, name):
        self.release["files"][name] = digest(self.root / name)

    def add_run(self, family, seed, *, policy_version=1):
        name = f"{family}-{seed}"
        path = self.root / "fitted" / name
        (path / "models").mkdir(parents=True)
        is_deep = family.startswith(("cnn-", "efficientnet-"))
        architecture = (
            ("small_cnn" if family.startswith("cnn-") else "efficientnet_b0")
            if is_deep
            else None
        )
        mode = family.split("-")[1] if is_deep else None
        loss = family.split("-")[2] if is_deep else "unweighted"
        config = {
            "run_id": name,
            "seed": seed,
            "training_seed": seed,
            "decision_policy_version": policy_version,
        }
        if is_deep:
            config.update(
                architecture=architecture,
                fine_tune_backbone=mode == "full",
                pretrained=mode == "head",
                loss_strategy=loss,
                dropout=0.3,
                image_size=32,
                batch_size=3,
                mc_samples=2,
            )
        else:
            config["model"] = family
        partitions = {
            "train": ["train-a", "train-b"],
            "selection": ["select-a", "select-b"],
            "calibration": ["cal-a", "cal-b", "cal-c", "cal-d"],
            "development": ["dev-a", "dev-b"],
        }
        split = {
            "purpose": "development",
            "seed": 42,
            "cohort_hash": "generated-cohort",
            "grouping": "lesion_id",
            "partitions": partitions,
            "rows": [
                {
                    "image_id": i,
                    "lesion_id": "lesion-" + i,
                    "target": (j // 2 if role == "calibration" else j % 2),
                }
                for role, ids in partitions.items()
                for j, i in enumerate(ids)
            ],
            "groups": [
                {"group_id": "lesion-" + i, "image_ids": [i]}
                for ids in partitions.values()
                for i in ids
            ],
        }
        split["split_hash"] = splitting.canonical_hash(split)
        write(path / "inputs/split-manifest.json", split)
        weight = 2.0 if loss == "pos_weight" else 1.0
        scores = np.array([0.1, 0.2, 0.7, 0.9])
        model_info = None
        if is_deep:
            deep.set_seed(seed)
            model = deep.build_model(
                architecture,
                dropout=0.3,
                pretrained=False,
                freeze_backbone=architecture == "small_cnn",
            )
            estimator = f"models/{architecture}_mc_dropout.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "split_hash": split["split_hash"],
                    "cohort_hash": split["cohort_hash"],
                    **{
                        k: config[k]
                        for k in (
                            "architecture",
                            "image_size",
                            "dropout",
                            "seed",
                            "fine_tune_backbone",
                            "pretrained",
                            "loss_strategy",
                        )
                    },
                    "pos_weight": None if weight == 1 else weight,
                },
                path / estimator,
            )
            model_info = {
                "architecture": architecture,
                "checkpoint_sha256": digest(path / estimator),
            }
            raw = np.array([scores, scores + 0.01])
            policy = calibration.fit_policy(
                [0, 0, 1, 1],
                raw,
                image_ids=partitions["calibration"],
                split_hash=split["split_hash"],
                weight=weight,
                variances=raw.var(axis=0, ddof=1),
            )
            policy["model"] = model_info
            training_mode = (
                "small_cnn_from_scratch"
                if architecture == "small_cnn"
                else "random_full_training"
                if mode == "full"
                else "pretrained_head_training"
            )
            write(
                path / "results/deep_training_metadata.json",
                {
                    "architecture": architecture,
                    "mode": training_mode,
                    "randomness": {"seed": seed},
                    "loss": {
                        "strategy": loss,
                        "pos_weight": None if weight == 1 else weight,
                    },
                    "checkpoint": {
                        "path": estimator,
                        "sha256": digest(path / estimator),
                    },
                    "transforms": {
                        "eval": repr(deep.build_transforms(32, train=False))
                    },
                },
            )
        else:
            x = np.random.default_rng(seed).random((8, 128))
            y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
            if family == "logistic":
                fitted, _ = classical.fit_logistic(x, y, x, y, seed=seed)
                estimator = "models/logistic_model.pkl"
            elif family == "gmm":
                from sklearn.mixture import GaussianMixture

                fitted = {
                    k: GaussianMixture(
                        n_components=1, covariance_type="diag", random_state=seed + k
                    ).fit(x[y == k])
                    for k in (0, 1)
                }
                estimator = "models/bayesian_gmm_model.pkl"
            else:
                raise ValueError("unsupported fixture family")
            if policy_version == 2:
                policy = calibration.fit_policy(
                    [0, 0, 1, 1],
                    [-1000, -5, 5, 1000],
                    schema_version=2,
                    split_manifest=split,
                    image_ids=partitions["calibration"],
                    split_hash=split["split_hash"],
                )
            else:
                policy = calibration.fit_policy(
                    [0, 0, 1, 1],
                    scores,
                    image_ids=partitions["calibration"],
                    split_hash=split["split_hash"],
                )
            with (path / estimator).open("wb") as handle:
                pickle.dump(
                    {
                        "model_kind": family,
                        "split_hash": split["split_hash"],
                        "cohort_hash": split["cohort_hash"],
                        "fitted": fitted,
                        "decision_policy": policy,
                        "train_mean": np.zeros(128),
                        "train_std": np.ones(128),
                        "class_priors": np.array([0.5, 0.5]),
                    },
                    handle,
                )
        write(path / "models/decision_policy.json", policy)
        prediction_files = {}
        shared = []
        pipeline = f"deep_{architecture}" if is_deep else f"classical_{family}"
        for role, ids in partitions.items():
            role_scores = scores if role == "calibration" else np.array([0.2, 0.8])
            if is_deep:
                role_scores = np.array([role_scores, role_scores + 0.01])
            elif policy_version == 2:
                role_scores = (
                    np.array([-1000, -5, 5, 1000])
                    if role == "calibration"
                    else np.array([-10, 10])
                )
            variance = role_scores.var(axis=0, ddof=1) if is_deep else None
            applied = calibration.apply_policy(policy, role_scores, variances=variance)
            targets = [
                j // 2 if role == "calibration" else j % 2 for j in range(len(ids))
            ]
            frame = pd.DataFrame(
                {
                    "schema_version": 1,
                    "run_id": name,
                    "model_key": pipeline,
                    "role": role,
                    "split_hash": split["split_hash"],
                    "image_id": ids,
                    "lesion_id": ["lesion-" + i for i in ids],
                    "group_id": ["lesion-" + i for i in ids],
                    "target": targets,
                    "prob_melanoma": applied["calibrated_probability"],
                    "prediction": applied["prediction"],
                }
            )
            shared.append(frame.copy())
            frame["raw_score"] = (
                role_scores.mean(axis=0)
                if is_deep
                else calibration.expit(role_scores)
                if policy_version == 2
                else role_scores
            )
            frame["corrected_score"] = applied["corrected_score"]
            frame["review_recommended"] = applied["review_recommended"]
            if policy_version == 2:
                frame["calibration_score"] = role_scores
                frame["ranking_score"] = applied["ranking_score"]
            if is_deep:
                statistics = uncertainty.summarize_mc_dropout_probabilities(role_scores)
                frame["predictive_std"] = np.sqrt(statistics["variance"])
                for column in (
                    "predictive_entropy",
                    "expected_entropy",
                    "mutual_information",
                ):
                    frame[column] = statistics[column]
                (path / "arrays").mkdir(exist_ok=True)
                np.save(
                    path / f"arrays/{architecture}_{role}_mc_probabilities.npy",
                    role_scores,
                )
                if role == "development":
                    for suffix, values in (
                        ("probability", role_scores.mean(axis=0)),
                        ("uncertainty", np.sqrt(variance)),
                        ("y_true", np.array(targets)),
                    ):
                        np.save(
                            path / f"arrays/{architecture}_development_{suffix}.npy",
                            values,
                        )
            filename = f"results/tables/{architecture + '_' if is_deep else ''}predictions_{role}.csv"
            (path / filename).parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path / filename, index=False)
            prediction_files[role] = filename
        pd.concat(shared, ignore_index=True).to_csv(
            path / "predictions.csv", index=False
        )
        summary = {
            "decision_policy_version": policy_version,
            "cost_matrix": {
                "false_negative": policy["decision"]["cost_fn"],
                "false_positive": policy["decision"]["cost_fp"],
                "map_threshold": 0.5,
                "cost_formula_threshold": policy["decision"]["formula_threshold"],
                "selected_threshold_from_calibration"
                if is_deep
                else "cost_threshold": policy["decision"]["threshold"],
            },
        }
        write(
            path
            / f"results/{architecture + '_' if is_deep else ''}metrics_summary.json",
            summary,
        )
        (path / "inputs/source.diff").write_bytes(b"")
        source_name = "src/generated_fixture.py"
        source_path = path / "inputs/untracked" / source_name
        source_path.parent.mkdir(parents=True)
        source_path.write_text("# Generated fixture provenance.\n")
        record = {
            "schema_version": 1,
            "status": "completed",
            "run_id": name,
            "pipeline": pipeline,
            "config": config,
            "config_sha256": splitting.canonical_hash(config),
            "environment": {"software": {"numpy": np.__version__}},
            "split_hash": split["split_hash"],
            "cohort_hash": split["cohort_hash"],
            "split_seed": split["seed"],
            "grouping": split["grouping"],
            "inputs": {"split_manifest": digest(path / "inputs/split-manifest.json")},
            "source": {
                "dirty_diff_sha256": digest(path / "inputs/source.diff"),
                "untracked_files": {source_name: digest(source_path)},
            },
            "prediction_files": prediction_files,
            "prediction_roles": sorted(prediction_files),
            "array_schemas": {
                str(p.relative_to(path)): {
                    "shape": list(a.shape),
                    "dtype": str(a.dtype),
                }
                for p in path.rglob("*.npy")
                for a in [np.load(p, allow_pickle=False)]
            },
            "artifacts": {
                str(p.relative_to(path)): digest(p)
                for p in path.rglob("*")
                if p.is_file()
            },
        }
        write(path / "run.json", record)
        self.register_run(family, seed)

    def register_run(self, family, seed):
        """Declare an existing generated fit with independently supplied identity."""
        name = f"{family}-{seed}"
        path = self.root / "fitted" / name
        record = json.loads((path / "run.json").read_text())
        config = record["config"]
        is_deep = family.startswith(("cnn-", "efficientnet-"))
        architecture = (
            ("small_cnn" if family.startswith("cnn-") else "efficientnet_b0")
            if is_deep
            else None
        )
        mode = family.split("-")[1] if is_deep else None
        loss = family.split("-")[2] if is_deep else "unweighted"
        estimator = (
            f"models/{architecture}_mc_dropout.pt"
            if is_deep
            else "models/bayesian_gmm_model.pkl"
            if family == "gmm"
            else f"models/{family}_model.pkl"
        )
        relative = str(path.relative_to(self.root))
        for p in path.rglob("*"):
            if p.is_file():
                self.pin(str(p.relative_to(self.root)))
        self.release["runs"][name] = {
            "path": relative,
            "run_id": name,
            "run_sha256": digest(path / "run.json"),
            "config_sha256": record["config_sha256"],
            "family": family,
            "seed": seed,
            "pipeline": record["pipeline"],
            "architecture": architecture,
            "pretrained": config["pretrained"] if is_deep else None,
            "mode": mode,
            "loss_strategy": loss,
            "positive_weight": 2.0 if loss == "pos_weight" else 1.0,
            "policy_version": config["decision_policy_version"],
            "estimator": f"{relative}/{estimator}",
            "policy": f"{relative}/models/decision_policy.json",
            "preprocessing": {
                "kind": "transforms",
                "artifact": f"{relative}/results/deep_training_metadata.json",
            }
            if is_deep
            else {"kind": "embedded", "artifact": f"{relative}/{estimator}"},
            "device": "cpu",
        }

    def save(self):
        write(self.root / "release.json", self.release)
        return digest(self.root / "release.json")

    def kwargs(self, name=None, output="output"):
        return dict(
            root=self.root,
            release_path=self.root / "release.json",
            release_sha256=self.save(),
            manifest_path=self.root / "manifest.json",
            manifest_sha256=digest(self.root / "manifest.json"),
            audit_path=self.root / "audit.json",
            audit_sha256=digest(self.root / "audit.json"),
            images_dir=self.root,
            run_id=name or next(iter(self.release["runs"])),
            output=self.root / output,
        )
