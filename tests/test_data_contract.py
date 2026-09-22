import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from src import data, splitting


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()

    def image(self, image_id):
        value = int(image_id[-7:])
        Image.new("RGB", (4, 4), (value * 27 % 256, value * 71 % 256, 0)).save(
            self.images / f"{image_id}.jpg"
        )

    def run_cohort(self, source, rows):
        metadata = self.root / "metadata.csv"
        with metadata.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return data.build_cohort(metadata, self.images, source=source)

    def test_ham_codes_and_attrition(self):
        codes = {"mel": 1, "nv": 0, "bkl": 0, "df": 0, "vasc": 0}
        excluded = {"bcc", "akiec", "scc", "mystery"}
        rows = []
        for i, code in enumerate([*codes, *excluded]):
            image_id = f"ISIC_{i:07d}"
            self.image(image_id)
            rows.append({"image_id": image_id, "lesion_id": f"L{i}", "dx": code})
        result = self.run_cohort("ham10000", rows)
        self.assertEqual(dict(zip(result.frame.dx, result.frame.target)), codes)
        self.assertEqual(result.counts["retained"], 5)
        self.assertEqual(result.counts["excluded_malignancy"], 2)
        self.assertEqual(result.counts["excluded_indeterminate"], 1)
        self.assertEqual(result.counts["unknown_diagnosis"], 1)
        self.assertEqual(len(result.outcomes), len(rows))

    def test_isic_hierarchy_and_conflicts(self):
        cases = [
            ("Benign", "Benign melanocytic proliferations", "Nevus", 0),
            (
                "Benign",
                "Benign epidermal proliferations",
                "Pigmented benign keratosis",
                0,
            ),
            ("Benign", "Benign soft tissue proliferations - Vascular", "", 0),
            (
                "Benign",
                "Benign soft tissue proliferations - Fibro-histiocytic",
                "Dermatofibroma",
                0,
            ),
            (
                "Malignant",
                "Malignant melanocytic proliferations (Melanoma)",
                "Melanoma, NOS",
                1,
            ),
            (
                "Malignant",
                "Malignant adnexal epithelial proliferations - Follicular",
                "Basal cell carcinoma",
                None,
            ),
            (
                "Malignant",
                "Malignant epidermal proliferations",
                "Squamous cell carcinoma, NOS",
                None,
            ),
            (
                "Indeterminate",
                "Indeterminate epidermal proliferations",
                "Solar or actinic keratosis",
                None,
            ),
        ]
        rows = []
        for i, (d1, d2, d3, _) in enumerate(cases):
            image_id = f"ISIC_{i:07d}"
            self.image(image_id)
            rows.append(
                {
                    "isic_id": image_id,
                    "lesion_id": f"L{i}",
                    "diagnosis_1": d1,
                    "diagnosis_2": d2,
                    "diagnosis_3": d3,
                }
            )
        result = self.run_cohort("isic2018_task3", rows)
        self.assertEqual(result.frame.target.tolist(), [0, 0, 0, 0, 1])
        self.assertEqual(result.counts["excluded_malignancy"], 2)
        self.assertEqual(result.counts["excluded_indeterminate"], 1)
        rows[0]["diagnosis_1"] = "Malignant"
        with self.assertRaisesRegex(ValueError, "conflicting diagnosis"):
            self.run_cohort("isic2018_task3", rows)

    def test_missing_columns_ids_and_duplicates_are_rejected(self):
        self.image("ISIC_0000001")
        valid = {"image_id": "ISIC_0000001", "lesion_id": "L1", "dx": "mel"}
        for rows in (
            [{"image_id": "ISIC_0000001", "dx": "mel"}],
            [valid, valid],
            [{**valid, "image_id": "../escape"}],
            [{**valid, "lesion_id": ""}],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.run_cohort("ham10000", rows)

    def test_duplicate_image_content_is_rejected(self):
        self.image("ISIC_0000001")
        (self.images / "ISIC_0000002.jpg").write_bytes(
            (self.images / "ISIC_0000001.jpg").read_bytes()
        )
        rows = [
            {"image_id": "ISIC_0000001", "lesion_id": "L1", "dx": "nv"},
            {"image_id": "ISIC_0000002", "lesion_id": "L2", "dx": "mel"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate image content"):
            self.run_cohort("ham10000", rows)

    def test_conflicting_diagnosis_fields_are_rejected(self):
        self.image("ISIC_0000001")
        isic = [
            {
                "isic_id": "ISIC_0000001",
                "lesion_id": "L1",
                "diagnosis_1": "Benign",
                "diagnosis_2": "Benign melanocytic proliferations",
                "diagnosis_3": "Melanoma, NOS",
            }
        ]
        with self.assertRaisesRegex(ValueError, "conflicting diagnosis"):
            self.run_cohort("isic2018_task3", isic)
        ham = [
            {
                "image_id": "ISIC_0000001",
                "lesion_id": "L1",
                "dx": "nv",
                "diagnosis_1": "Malignant",
            }
        ]
        with self.assertRaisesRegex(ValueError, "conflicting diagnosis"):
            self.run_cohort("ham10000", ham)
        ham[0].update({"diagnosis_1": "Benign", "diagnosis_3": "Melanoma, NOS"})
        with self.assertRaisesRegex(ValueError, "conflicting diagnosis"):
            self.run_cohort("ham10000", ham)
        isic[0]["diagnosis_3"] = "Nevus"
        isic[0]["dx"] = "mel"
        with self.assertRaisesRegex(ValueError, "conflicting diagnosis"):
            self.run_cohort("isic2018_task3", isic)

    def test_conflicting_labels_within_lesion_are_rejected(self):
        self.image("ISIC_0000001")
        self.image("ISIC_0000002")
        rows = [
            {"image_id": "ISIC_0000001", "lesion_id": "L1", "dx": "mel"},
            {"image_id": "ISIC_0000002", "lesion_id": "L1", "dx": "nv"},
        ]
        with self.assertRaisesRegex(ValueError, "conflicting lesion diagnoses"):
            self.run_cohort("ham10000", rows)

    def test_missing_and_corrupt_images_have_reasons(self):
        (self.images / "ISIC_0000002.jpg").write_bytes(b"bad image")
        rows = [
            {"image_id": "ISIC_0000001", "lesion_id": "L1", "dx": "nv"},
            {"image_id": "ISIC_0000002", "lesion_id": "L2", "dx": "mel"},
        ]
        with self.assertRaisesRegex(ValueError, "empty cohort"):
            self.run_cohort("ham10000", rows)
        self.image("ISIC_0000003")
        rows.append({"image_id": "ISIC_0000003", "lesion_id": "L3", "dx": "mel"})
        result = self.run_cohort("ham10000", rows)
        self.assertEqual(result.counts["missing_image"], 1)
        self.assertEqual(result.counts["corrupt_image"], 1)

    def test_reordering_and_source_schemas_agree(self):
        self.image("ISIC_0000001")
        self.image("ISIC_0000002")
        ham = [
            {"image_id": "ISIC_0000002", "lesion_id": "L2", "dx": "mel"},
            {"image_id": "ISIC_0000001", "lesion_id": "L1", "dx": "nv"},
        ]
        a = self.run_cohort("ham10000", ham)
        b = self.run_cohort("ham10000", list(reversed(ham)))
        isic = [
            {
                "isic_id": "ISIC_0000001",
                "lesion_id": "L1",
                "diagnosis_1": "Benign",
                "diagnosis_2": "Benign melanocytic proliferations",
                "diagnosis_3": "Nevus",
            },
            {
                "isic_id": "ISIC_0000002",
                "lesion_id": "L2",
                "diagnosis_1": "Malignant",
                "diagnosis_2": "Malignant melanocytic proliferations (Melanoma)",
                "diagnosis_3": "Melanoma, NOS",
            },
        ]
        c = self.run_cohort("isic2018_task3", isic)
        columns = ["isic_id", "lesion_id", "dx", "target"]
        self.assertEqual(
            a.frame[columns].to_dict("records"), b.frame[columns].to_dict("records")
        )
        self.assertEqual(
            a.frame[columns].to_dict("records"), c.frame[columns].to_dict("records")
        )
        self.assertEqual(str(a.frame.target.dtype), str(c.frame.target.dtype))

    def test_preparation_saves_complete_attrition(self):
        self.image("ISIC_0000001")
        rows = [
            {"image_id": "ISIC_0000001", "lesion_id": "L1", "dx": "mel"},
            {"image_id": "ISIC_0000002", "lesion_id": "L2", "dx": "bcc"},
        ]
        metadata = self.root / "metadata.csv"
        with metadata.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        audit_path = self.root / "output" / "cohort_attrition.json"
        frame, ids, labels = data.prepare_dataset(
            str(metadata),
            str(self.images),
            source="ham10000",
            attrition_path=audit_path,
        )
        audit = json.loads(audit_path.read_text())
        self.assertEqual(len(frame), 1)
        self.assertEqual(ids.tolist(), ["ISIC_0000001"])
        self.assertEqual(labels.tolist(), [1])
        self.assertEqual(
            audit["counts"], {"input": 2, "retained": 1, "excluded_malignancy": 1}
        )
        self.assertEqual(len(audit["outcomes"]), 2)
        self.assertEqual(len(audit["metadata_sha256"]), 64)

    def test_empty_cohort_still_saves_attrition(self):
        metadata = self.root / "metadata.csv"
        metadata.write_text("image_id,lesion_id,dx\nISIC_0000001,L1,nv\n")
        audit_path = self.root / "cohort_attrition.json"
        with self.assertRaisesRegex(ValueError, "empty cohort"):
            data.prepare_dataset(
                str(metadata),
                str(self.images),
                source="ham10000",
                attrition_path=audit_path,
            )
        audit = json.loads(audit_path.read_text())
        self.assertEqual(audit["counts"], {"input": 1, "missing_image": 1})

    def test_excluded_rows_preserve_chained_identity_links(self):
        rows = []
        for target in (0, 1):
            for index in range(12):
                image_id = f"I{target}_{index:02d}"
                Image.new("RGB", (4, 4), (target * 110, index * 18, 25)).save(
                    self.images / f"{image_id}.jpg"
                )
                rows.append(
                    {
                        "image_id": image_id,
                        "lesion_id": f"L{target}_{index:02d}",
                        "dx": "mel" if target else "nv",
                        "duplicate_cluster_id": "",
                    }
                )
        # A -> missing -> corrupt -> excluded diagnosis -> B. None of the
        # three bridge images is eligible, but each supplies a known link.
        rows[0]["duplicate_cluster_id"] = "first"
        rows[1]["duplicate_cluster_id"] = "last"
        rows.extend(
            [
                {
                    "image_id": "missing",
                    "lesion_id": "bridge1",
                    "dx": "nv",
                    "duplicate_cluster_id": "first",
                },
                {
                    "image_id": "corrupt",
                    "lesion_id": "bridge1",
                    "dx": "nv",
                    "duplicate_cluster_id": "middle",
                },
                {
                    "image_id": "excluded",
                    "lesion_id": "bridge2",
                    "dx": "bcc",
                    "duplicate_cluster_id": "middle",
                },
                {
                    "image_id": "excluded2",
                    "lesion_id": "bridge2",
                    "dx": "bcc",
                    "duplicate_cluster_id": "last",
                },
            ]
        )
        (self.images / "corrupt.jpg").write_bytes(b"not an image")
        frame = self.run_cohort("ham10000", rows).frame
        manifest = splitting.create_manifest(frame)
        membership = {
            image: group["group_id"]
            for group in manifest["groups"]
            for image in group["image_ids"]
        }
        roles = {
            image: role for role, ids in manifest["partitions"].items() for image in ids
        }
        self.assertEqual(membership["I0_00"], membership["I0_01"])
        self.assertEqual(roles["I0_00"], roles["I0_01"])
        self.assertNotEqual(membership["I0_00"], membership["I0_02"])
        self.assertEqual(
            manifest,
            splitting.create_manifest(self.run_cohort("ham10000", rows[::-1]).frame),
        )

        tampered = json.loads(json.dumps(manifest))
        tampered["identity_components"]["I0_01"] = "unlinked"
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            splitting.manifest_indices(frame, tampered)

        mixed = [dict(row) for row in rows]
        mixed[1]["duplicate_cluster_id"] = ""
        mixed[12]["duplicate_cluster_id"] = "last"
        with self.assertRaisesRegex(ValueError, "mixed labels"):
            splitting.create_manifest(self.run_cohort("ham10000", mixed).frame)

    def test_patient_and_exact_content_links_survive_exclusion(self):
        rows = []
        for target in (0, 1):
            for index in range(12):
                image_id = f"P{target}_{index:02d}"
                Image.new("RGB", (4, 4), (target * 110, index * 18, 35)).save(
                    self.images / f"{image_id}.jpg"
                )
                rows.append(
                    {
                        "image_id": image_id,
                        "lesion_id": f"L{target}_{index:02d}",
                        "patient_id": f"patient{target}_{index:02d}",
                        "duplicate_cluster_id": "",
                        "dx": "mel" if target else "nv",
                    }
                )
        rows[1]["duplicate_cluster_id"] = "pixel_bridge"
        rows.append(
            {
                "image_id": "excluded_pixel",
                "lesion_id": "other",
                "patient_id": "patient_extra",
                "duplicate_cluster_id": "pixel_bridge",
                "dx": "bcc",
            }
        )
        shutil.copyfile(self.images / "P0_00.jpg", self.images / "excluded_pixel.jpg")
        rows[2]["duplicate_cluster_id"] = "patient_bridge"
        rows.append(
            {
                "image_id": "missing_patient",
                "lesion_id": "another",
                "patient_id": "patient0_00",
                "duplicate_cluster_id": "patient_bridge",
                "dx": "nv",
            }
        )
        frame = self.run_cohort("ham10000", rows).frame
        manifest = splitting.create_manifest(frame)
        group = {
            image: item["group_id"]
            for item in manifest["groups"]
            for image in item["image_ids"]
        }
        self.assertEqual({group[f"P0_{i:02d}"] for i in (0, 1, 2)}, {group["P0_00"]})
        self.assertEqual(manifest["schema_version"], 2)
        rows[-1]["patient_id"] = ""
        with self.assertRaisesRegex(ValueError, "partial patient IDs"):
            self.run_cohort("ham10000", rows)


if __name__ == "__main__":
    unittest.main()
