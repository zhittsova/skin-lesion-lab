import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from src import external


class ExternalContractTests(unittest.TestCase):
    def test_release_rejects_changed_bytes_and_changed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from tests.external_fixtures import ReleaseFixture

            fixture = ReleaseFixture(root)
            release = root / "release.json"
            digest = fixture.save()
            external.verify_files(release, digest, root)
            artifact = root / "fitted/logistic-17/models/logistic_model.pkl"
            artifact.write_bytes(b"refitted")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                external.verify_files(release, digest, root)
            release.write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                external.verify_files(release, digest, root)

    def test_connected_groups_preserve_mixed_patient_and_excluded_links(self):
        rows = pd.DataFrame(
            [
                {
                    "image_id": "a",
                    "lesion_id": "L1",
                    "patient_id": "P1",
                    "rgb_sha256": "x",
                    "target": 0,
                },
                {
                    "image_id": "b",
                    "lesion_id": "L2",
                    "patient_id": "P1",
                    "rgb_sha256": "y",
                    "target": 1,
                },
                {
                    "image_id": "c",
                    "lesion_id": "L4",
                    "patient_id": "P2",
                    "rgb_sha256": "y",
                    "target": None,
                },
                {
                    "image_id": "d",
                    "lesion_id": "L3",
                    "patient_id": "P2",
                    "rgb_sha256": "w",
                    "target": 0,
                },
            ]
        )
        groups = external.connected_groups(rows)
        self.assertEqual(set(groups.values()), {"a"})
        self.assertEqual(groups, external.connected_groups(rows.iloc[::-1]))
        rows.loc[0, "patient_id"] = ""
        with self.assertRaisesRegex(ValueError, "patient"):
            external.connected_groups(rows)

    def test_shared_patient_draws_keep_entire_mixed_group(self):
        frame = pd.DataFrame(
            {
                "image_id": ["a", "b", "c"],
                "group_id": ["p", "p", "q"],
                "target": [0, 1, 0],
                "prob_melanoma": [0.2, 0.8, 0.3],
                "prediction": [0, 1, 0],
            }
        )
        draws = list(external.component_draws(frame, draws=50, seed=2026))
        self.assertTrue(any(list(d).count(0) == 2 for d in draws))
        for d in draws:
            self.assertEqual(list(d).count(0), list(d).count(1))
        models = {
            m: {s: frame.copy() for s in (17, 42, 73)} for m in ("logistic", "deep")
        }
        report = external.paired_report(models, reference="logistic", draws=50)
        self.assertEqual(report["models"]["deep"]["roc_auc"]["estimate"], 1.0)
        self.assertEqual(report["differences"]["deep"]["brier_score"]["estimate"], 0)
        self.assertGreater(report["models"]["deep"]["roc_auc"]["undefined_draws"], 0)
        self.assertIsNone(report["models"]["deep"]["roc_auc"]["interval"])
        models["deep"][73].loc[0, "image_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "alignment"):
            external.paired_report(models, reference="logistic", draws=5)

    def test_overlap_checks_ids_bytes_pixels_and_near_candidates(self):
        reference = [
            {
                "image_id": "old",
                "image_sha256": "bytes",
                "rgb_sha256": "pixels",
                "phash": 255,
            }
        ]
        rows = [
            {
                "image_id": "old",
                "image_sha256": "new",
                "rgb_sha256": "new",
                "phashes": [0],
            },
            {
                "image_id": "b",
                "image_sha256": "bytes",
                "rgb_sha256": "new",
                "phashes": [0],
            },
            {
                "image_id": "c",
                "image_sha256": "new",
                "rgb_sha256": "pixels",
                "phashes": [0],
            },
            {
                "image_id": "d",
                "image_sha256": "new",
                "rgb_sha256": "new",
                "phashes": [254],
            },
        ]
        audit = external.overlap_audit(rows, reference)
        self.assertEqual({r["image_id"] for r in audit["exact"]}, {"old", "b", "c"})
        self.assertEqual(
            audit["near"], [{"image_id": "d", "reference_id": "old", "distance": 1}]
        )
        self.assertFalse(audit["clear"])

    def test_prediction_contract_rejects_alignment_nonfinite_and_policy_mutation(self):
        policy = {
            "schema_version": 1,
            "score_transform": {"positive_weight": 1.0},
            "calibrator": {"slope": 1.0, "intercept": 0.0},
            "decision": {"threshold": 0.5},
            "referral": {"mc_passes": None, "variance_cutoff": None, "margin": 0.1},
        }
        original = copy.deepcopy(policy)
        cohort = pd.DataFrame(
            {"image_id": ["a", "b"], "group_id": ["a", "b"], "target": [0, 1]}
        )
        result = external.prediction_frame(
            cohort, [0.25, 0.75], policy, run_id="logistic-17"
        )
        np.testing.assert_array_equal(result.prediction, [0, 1])
        np.testing.assert_allclose(result.prob_melanoma, [0.25, 0.75])
        self.assertEqual(policy, original)
        changed = cohort.copy()
        changed.target = 1 - changed.target
        second = external.prediction_frame(
            changed, [0.25, 0.75], policy, run_id="logistic-17"
        )
        np.testing.assert_array_equal(result.prob_melanoma, second.prob_melanoma)
        for scores in ([0.5], [np.nan, 0.1], [1.1, 0.2]):
            with self.assertRaises(ValueError):
                external.prediction_frame(cohort, scores, policy, run_id="logistic-17")

    def test_fingerprints_detect_reencoded_pixels_and_rotated_candidates(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.random.default_rng(5).integers(
                0, 256, (37, 61, 3), dtype=np.uint8
            )
            Image.fromarray(pixels).save(root / "a.png")
            Image.fromarray(pixels).save(root / "b.bmp")
            Image.fromarray(np.rot90(pixels)).save(root / "c.png")
            a = external.fingerprint(root / "a.png", "a")
            b = external.fingerprint(root / "b.bmp", "b")
            c = external.fingerprint(root / "c.png", "c")
            self.assertNotEqual(a["image_sha256"], b["image_sha256"])
            self.assertEqual(a["rgb_sha256"], b["rgb_sha256"])
            self.assertIn(a["phash"], c["phashes"])

    def test_one_class_draw_keeps_defined_scores(self):
        frame = pd.DataFrame(
            {
                "image_id": ["a", "b"],
                "group_id": ["a", "b"],
                "target": [0, 0],
                "prob_melanoma": [0.25, 0.5],
                "prediction": [0, 1],
            }
        )
        result = external.endpoints(frame)
        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(result["sensitivity"])
        self.assertEqual(result["specificity"], 0.5)
        self.assertEqual(result["brier_score"], 0.15625)
        self.assertEqual(result["average_cost"], 0.5)

    def test_grouping_rejects_conflicting_lesion_owner_and_links_known_duplicates(self):
        frame = pd.DataFrame(
            {
                "image_id": ["a", "b"],
                "lesion_id": ["L1", "L1"],
                "patient_id": ["P1", "P2"],
                "rgb_sha256": ["x", "y"],
                "duplicate_cluster_id": ["", ""],
            }
        )
        with self.assertRaisesRegex(ValueError, "lesion.*patient"):
            external.connected_groups(frame)
        frame.loc[1, "lesion_id"] = "L2"
        frame["duplicate_cluster_id"] = "known"
        self.assertEqual(external.connected_groups(frame), {"a": "a", "b": "a"})

    def test_hiba_adapter_keeps_task_labels_and_records_exclusions(self):
        from PIL import Image

        labels = [
            (
                "melanoma",
                "Malignant",
                "Malignant melanocytic proliferations (Melanoma)",
                "Melanoma, NOS",
            ),
            ("nevus", "Benign", "Benign melanocytic proliferations", "Nevus"),
            (
                "actinic keratosis",
                "Indeterminate",
                "Indeterminate epidermal proliferations",
                "Solar or actinic keratosis",
            ),
            (
                "basal cell carcinoma",
                "Malignant",
                "Malignant adnexal epithelial proliferations - Follicular",
                "Basal cell carcinoma",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i, (dx, d1, d2, d3) in enumerate(labels + labels[:2]):
                image_id = f"I{i}"
                Image.new("RGB", (8, 8), (20 * i, 10 * i, 12)).save(
                    root / f"{image_id}.jpg"
                )
                rows.append(
                    {
                        "isic_id": image_id,
                        "lesion_id": f"L{i}",
                        "patient_id": "P1" if i < 2 else f"P{i}",
                        "diagnosis": dx,
                        "diagnosis_1": d1,
                        "diagnosis_2": d2,
                        "diagnosis_3": d3,
                        "image_type": "clinical: overview" if i == 4 else "dermoscopic",
                        "dermoscopic_type": "" if i == 4 else "contact polarized",
                        "copyright_license": "CC-BY",
                        "attribution": "Hospital Italiano de Buenos Aires",
                    }
                )
            (root / "I5.jpg").unlink()
            metadata = root / "metadata.csv"
            pd.DataFrame(rows).to_csv(metadata, index=False)
            manifest = external.hiba_manifest(metadata, root)
            self.assertEqual(manifest["purpose"], "external")
            self.assertEqual(
                [
                    (r["image_id"], r["target"])
                    for r in manifest["rows"]
                    if r["reason"] == "retained"
                ],
                [("I0", 1), ("I1", 0)],
            )
            self.assertEqual(
                manifest["counts"],
                {
                    "input": 6,
                    "retained": 2,
                    "excluded_indeterminate": 1,
                    "excluded_malignancy": 1,
                    "excluded_modality": 1,
                    "missing_image": 1,
                },
            )
            self.assertEqual(
                manifest["rows"][0]["group_id"], manifest["rows"][1]["group_id"]
            )
            rows[0]["diagnosis_1"] = "Benign"
            pd.DataFrame(rows).to_csv(metadata, index=False)
            with self.assertRaisesRegex(ValueError, "conflicting diagnosis"):
                external.hiba_manifest(metadata, root)

    def test_identical_pixels_with_conflicting_diagnoses_fail(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8), (25, 50, 75)).save(root / "a.jpg")
            (root / "b.jpg").write_bytes((root / "a.jpg").read_bytes())
            base = {
                "image_type": "dermoscopic",
                "dermoscopic_type": "contact polarized",
                "copyright_license": "CC-BY",
                "attribution": "Hospital Italiano de Buenos Aires",
            }
            rows = [
                {
                    **base,
                    "isic_id": "a",
                    "lesion_id": "a",
                    "patient_id": "p1",
                    "diagnosis": "melanoma",
                    "diagnosis_1": "Malignant",
                    "diagnosis_2": "Malignant melanocytic proliferations (Melanoma)",
                    "diagnosis_3": "Melanoma, NOS",
                },
                {
                    **base,
                    "isic_id": "b",
                    "lesion_id": "b",
                    "patient_id": "p2",
                    "diagnosis": "nevus",
                    "diagnosis_1": "Benign",
                    "diagnosis_2": "Benign melanocytic proliferations",
                    "diagnosis_3": "Nevus",
                },
            ]
            pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
            with self.assertRaisesRegex(ValueError, "duplicate.*diagnos"):
                external.hiba_manifest(root / "metadata.csv", root)


class IndependentUnequalGroupOracle(unittest.TestCase):
    def test_unequal_mixed_groups_keep_paired_seed_mean_auc(self):
        ids = list("abcdefg")
        groups = np.array(["P1", "P1", "P1", "P2", "P2", "P3", "P3"])
        targets = np.array([1, 0, 0, 1, 0, 1, 0])
        scores = {
            "logistic": {
                17: [0.4, 0.6, 0.2, 0.7, 0.5, 0.8, 0.1],
                42: [0.6, 0.5, 0.3, 0.8, 0.7, 0.4, 0.2],
                73: [0.5, 0.4, 0.6, 0.8, 0.2, 0.7, 0.1],
            },
            "deep": {
                17: [0.8, 0.5, 0.2, 0.7, 0.4, 0.6, 0.3],
                42: [0.8, 0.4, 0.3, 0.6, 0.2, 0.5, 0.7],
                73: [0.9, 0.2, 0.3, 0.7, 0.4, 0.8, 0.6],
            },
        }
        frames = {
            model: {
                seed: pd.DataFrame(
                    {
                        "image_id": ids,
                        "group_id": groups,
                        "target": targets,
                        "prob_melanoma": probability,
                        "prediction": np.asarray(probability) >= 0.5,
                    }
                )
                for seed, probability in runs.items()
            }
            for model, runs in scores.items()
        }

        def pair_count_auc(labels, probability):
            positive = probability[labels == 1]
            negative = probability[labels == 0]
            comparisons = positive[:, None] - negative[None, :]
            return np.mean((comparisons > 0) + 0.5 * (comparisons == 0))

        def seed_mean(model, positions):
            return np.mean(
                [
                    pair_count_auc(targets[positions], np.array(p)[positions])
                    for p in scores[model].values()
                ]
            )

        # Construct resamples independently of external.component_draws.
        rng = np.random.default_rng(2026)
        samples = {model: [] for model in scores}
        paired = []
        for _ in range(127):
            sampled_groups = rng.choice(["P1", "P2", "P3"], 3, replace=True)
            positions = np.array(
                [
                    i
                    for group in sampled_groups
                    for i, g in enumerate(groups)
                    if g == group
                ]
            )
            values = {model: seed_mean(model, positions) for model in scores}
            for model, value in values.items():
                samples[model].append(value)
            paired.append(values["deep"] - values["logistic"])

        report = external.paired_report(frames, draws=127, seed=2026)
        for model in scores:
            estimate = report["models"][model]["roc_auc"]
            self.assertAlmostEqual(estimate["estimate"], seed_mean(model, np.arange(7)))
            self.assertEqual(estimate["undefined_draws"], 0)
            np.testing.assert_allclose(
                estimate["interval"],
                np.quantile(samples[model], [0.025, 0.975]),
                rtol=0,
                atol=1e-14,
            )
        difference = report["differences"]["deep"]["roc_auc"]
        self.assertAlmostEqual(difference["estimate"], 1 / 9)
        np.testing.assert_allclose(
            difference["interval"],
            np.quantile(paired, [0.025, 0.975]),
            rtol=0,
            atol=1e-14,
        )
        np.testing.assert_allclose(difference["interval"], [-1 / 3, 1 / 3], atol=1e-14)
