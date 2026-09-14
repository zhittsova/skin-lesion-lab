import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from src import splitting


def cohort(n=20):
    frame = pd.DataFrame(
        [
            {
                "isic_id": f"I{label}_{i:03d}",
                "lesion_id": f"L{label}_{i // 2:03d}",
                "target": label,
                "dx": "mel" if label else "nv",
                "image_sha256": hashlib.sha256(f"{label}:{i}".encode()).hexdigest(),
                "patient_id": "",
                "duplicate_cluster_id": "",
            }
            for label in (0, 1)
            for i in range(n)
        ]
    )
    frame.attrs = {
        "source": "ham10000",
        "metadata_content_hash": "a" * 64,
        "label_policy_version": "melanoma-selected-benign-v1",
    }
    return frame


class SplitContractTests(unittest.TestCase):
    def test_full_coverage_classes_and_disjoint_groups(self):
        frame = cohort()
        manifest = splitting.create_manifest(frame)
        parts = splitting.manifest_indices(frame, manifest)
        self.assertEqual(
            set(parts), {"train", "selection", "calibration", "development"}
        )
        self.assertEqual(
            sorted(np.concatenate(list(parts.values()))), list(range(len(frame)))
        )
        groups = []
        for indices in parts.values():
            self.assertEqual(set(frame.iloc[indices].target), {0, 1})
            groups.append(set(frame.iloc[indices].lesion_id))
        for i, group in enumerate(groups):
            for other in groups[i + 1 :]:
                self.assertFalse(group & other)

    def test_permutation_preserves_manifest_and_index_id_order(self):
        frame = cohort()
        other = frame.sample(frac=1, random_state=17).reset_index(drop=True)
        first = splitting.create_manifest(frame)
        self.assertEqual(first, splitting.create_manifest(other))
        for role, indices in splitting.manifest_indices(other, first).items():
            self.assertEqual(
                other.iloc[indices].isic_id.tolist(), first["partitions"][role]
            )

    def test_invalid_contracts_fail(self):
        for column, value in (
            ("isic_id", ""),
            ("lesion_id", None),
            ("image_sha256", "bad"),
            ("target", 2),
            ("target", 0.5),
        ):
            frame = cohort().astype({"target": object})
            frame.loc[0, column] = value
            with (
                self.subTest(column=column, value=value),
                self.assertRaises(ValueError),
            ):
                splitting.create_manifest(frame)
        frame = cohort()
        frame.loc[1, "isic_id"] = frame.loc[0, "isic_id"]
        with self.assertRaisesRegex(ValueError, "duplicate image IDs"):
            splitting.create_manifest(frame)
        frame = cohort()
        frame.loc[0, "target"] = 1
        with self.assertRaisesRegex(ValueError, "mixed labels"):
            splitting.create_manifest(frame)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            splitting.create_manifest(cohort(n=6))
        for fractions in (
            (0.6, 0.2, 0.2, 0),
            (0.6, 0.2, 0.2, -0.1),
            (float("nan"), 0.1, 0.1, 0.1),
            (0.1, 0.1, 0.1, 0.1),
        ):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                splitting.create_manifest(cohort(), fractions=fractions)

    def test_transitive_duplicate_and_patient_groups_stay_together(self):
        frame = cohort(40)
        frame.loc[[0, 2], "duplicate_cluster_id"] = "D1"
        frame.loc[3, "image_sha256"] = frame.loc[4, "image_sha256"]
        manifest = splitting.create_manifest(frame)
        membership = {
            image: role for role, ids in manifest["partitions"].items() for image in ids
        }
        self.assertEqual(
            len({membership[frame.loc[i, "isic_id"]] for i in range(6)}), 1
        )
        frame["patient_id"] = frame.lesion_id
        frame.loc[2:3, "patient_id"] = frame.loc[0, "patient_id"]
        manifest = splitting.create_manifest(frame)
        self.assertEqual(manifest["grouping"], "patient+lesion+known-duplicates")
        frame.loc[0, "patient_id"] = ""
        with self.assertRaisesRegex(ValueError, "partial patient"):
            splitting.create_manifest(frame)
        frame = cohort(40)
        frame.loc[[0, 40], "duplicate_cluster_id"] = "conflict"
        with self.assertRaisesRegex(ValueError, "mixed labels"):
            splitting.create_manifest(frame)

    def test_patient_only_bridge_connects_independent_lesions(self):
        frame = cohort(40)
        frame["patient_id"] = frame.lesion_id
        frame.loc[4:5, "patient_id"] = frame.loc[0, "patient_id"]
        manifest = splitting.create_manifest(frame)
        groups = [set(group["image_ids"]) for group in manifest["groups"]]
        expected = set(frame.loc[[0, 1, 4, 5], "isic_id"])
        self.assertIn(expected, groups)
        assignments = {
            image: role for role, ids in manifest["partitions"].items() for image in ids
        }
        self.assertEqual(len({assignments[image] for image in expected}), 1)
        frame.loc[40:41, "patient_id"] = frame.loc[0, "patient_id"]
        with self.assertRaisesRegex(ValueError, "mixed labels"):
            splitting.create_manifest(frame)

    def test_stale_or_tampered_manifest_rejected_and_never_overwritten(self):
        frame = cohort()
        original = splitting.create_manifest(frame)
        for column, value in (
            ("target", 1),
            ("image_sha256", "b" * 64),
            ("dx", "bkl"),
            ("isic_id", "replacement"),
            ("duplicate_cluster_id", "new"),
        ):
            changed = frame.copy()
            changed.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                splitting.manifest_indices(changed, original)
        changed = frame.copy()
        changed.attrs["metadata_content_hash"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "manifest"):
            splitting.manifest_indices(changed, original)
        for change in ("missing", "overlap", "seed", "purpose", "hash"):
            bad = copy.deepcopy(original)
            if change == "missing":
                bad["partitions"]["train"].pop()
            elif change == "overlap":
                bad["partitions"]["train"].append(bad["partitions"]["development"][0])
            elif change == "purpose":
                bad["purpose"] = "confirmation"
            elif change == "seed":
                bad["seed"] += 1
            else:
                bad["cohort_hash"] = "b" * 64
            with self.subTest(change=change), self.assertRaises(ValueError):
                splitting.manifest_indices(frame, bad)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "split.json"
            splitting.save_manifest(original, path)
            self.assertEqual(splitting.load_manifest(frame, path), original)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                splitting.save_manifest(original, path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(json.loads(before), original)

    def test_legacy_split_rejects_mixed_labels_and_invalid_report(self):
        groups = np.array([f"L{i // 2}" for i in range(80)])
        labels = np.array([0] * 40 + [1] * 40)
        labels[0] = 1
        with self.assertRaisesRegex(ValueError, "mixed labels"):
            splitting.split_dataset(groups, labels)
        labels[0] = 0
        bad = {
            "train": np.arange(40),
            "val": np.arange(40, 60),
            "test": np.arange(59, 80),
        }
        with self.assertRaisesRegex(ValueError, "coverage|overlap"):
            splitting.get_split_report(labels, bad, groups)


if __name__ == "__main__":
    unittest.main()


class SplitPreparationTests(unittest.TestCase):
    def test_real_preparation_preserves_linkage_and_invalidates_changed_bytes(self):
        from PIL import Image
        from src.data import prepare_dataset

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            rows = cohort(20).rename(columns={"isic_id": "image_id"})
            rows["patient_id"] = rows.lesion_id
            rows.loc[[0, 2], "duplicate_cluster_id"] = "known_pair"
            metadata = root / "metadata.csv"
            columns = [
                "image_id",
                "lesion_id",
                "dx",
                "patient_id",
                "duplicate_cluster_id",
            ]
            rows[columns].to_csv(metadata, index=False)
            for i, image_id in enumerate(rows.image_id):
                Image.new("RGB", (4, 4), (i * 5, i * 3, 7)).save(
                    root / f"{image_id}.jpg"
                )

            def prepare():
                return prepare_dataset(
                    str(metadata),
                    str(root),
                    source="ham10000",
                    attrition_path=root / "audit.json",
                )[0]

            first = prepare()
            manifest = splitting.create_manifest(first)
            self.assertEqual(first.patient_id.tolist(), rows.patient_id.tolist())
            self.assertEqual(
                first.duplicate_cluster_id.tolist(), rows.duplicate_cluster_id.tolist()
            )
            rows[columns].iloc[::-1].to_csv(metadata, index=False)
            self.assertEqual(splitting.create_manifest(prepare()), manifest)
            Image.new("RGB", (4, 4), (255, 255, 255)).save(
                root / f"{rows.iloc[0].image_id}.jpg"
            )
            with self.assertRaisesRegex(ValueError, "manifest"):
                splitting.manifest_indices(prepare(), manifest)

    def test_metadata_only_change_invalidates_saved_manifest(self):
        a = pd.DataFrame({"image_id": ["a", "b"], "age": ["20", "30"]})
        self.assertEqual(
            splitting.metadata_hash(a), splitting.metadata_hash(a.iloc[::-1])
        )
        b = a.copy()
        b.loc[0, "age"] = "21"
        self.assertNotEqual(splitting.metadata_hash(a), splitting.metadata_hash(b))
