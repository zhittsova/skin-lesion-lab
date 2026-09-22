"""External cohort count arithmetic from explicit generated manifest rows."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import external_inference, splitting
from tests.external_fixtures import ReleaseFixture, digest, write


def manifest(rows, counts):
    """Build a self-hashed manifest without using the production count helper."""
    value = {
        "schema_version": 1,
        "purpose": "external",
        "rows": rows,
        "counts": counts,
    }
    value["cohort_sha256"] = splitting.canonical_hash(value)
    return value


def explicit_rows():
    """Nine hand-counted rows spanning every supported final row reason."""
    reasons = [
        "retained",
        "retained",
        "excluded_malignancy",
        "excluded_modality",
        "excluded_indeterminate",
        "unknown_diagnosis",
        "missing_image",
        "corrupt_image",
        "duplicate_pixels",
    ]
    return [
        {
            "image_id": f"image-{index}",
            "group_id": f"group-{index}",
            "target": index % 2 if reason == "retained" else None,
            "reason": reason,
            "image_sha256": f"hash-{index}" if reason == "retained" else "",
        }
        for index, reason in enumerate(reasons)
    ]


EXPECTED_COUNTS = {
    "input": 9,
    "retained": 2,
    "excluded_malignancy": 1,
    "excluded_modality": 1,
    "excluded_indeterminate": 1,
    "unknown_diagnosis": 1,
    "missing_image": 1,
    "corrupt_image": 1,
    "duplicate_pixels": 1,
}


class ExternalManifestCountTests(unittest.TestCase):
    def test_mixed_final_reasons_match_independent_fixture_oracle(self):
        value = manifest(explicit_rows(), copy.deepcopy(EXPECTED_COUNTS))
        observed = external_inference.validate_manifest(value)
        self.assertEqual(observed["counts"], EXPECTED_COUNTS)
        self.assertEqual(
            len([row for row in observed["rows"] if row["reason"] == "retained"]),
            2,
        )

    def test_sparse_zero_exclusions_and_row_order_are_valid(self):
        rows = explicit_rows()[:2]
        sparse = manifest(rows, {"input": 2, "retained": 2})
        external_inference.validate_manifest(sparse)

        with_zero = manifest(
            list(reversed(rows)),
            {
                "input": 2,
                "retained": 2,
                "excluded_modality": 0,
                "duplicate_pixels": 0,
            },
        )
        external_inference.validate_manifest(with_zero)

    def test_declared_totals_must_match_rows_independently(self):
        rows = explicit_rows()
        variants = []
        wrong_input = copy.deepcopy(EXPECTED_COUNTS)
        wrong_input["input"] = 10
        variants.append(wrong_input)
        wrong_exclusion = copy.deepcopy(EXPECTED_COUNTS)
        wrong_exclusion["excluded_modality"] = 2
        variants.append(wrong_exclusion)
        self_consistent_but_wrong = copy.deepcopy(EXPECTED_COUNTS)
        self_consistent_but_wrong["retained"] = 3
        self_consistent_but_wrong["duplicate_pixels"] = 0
        variants.append(self_consistent_but_wrong)
        omitted_nonzero = copy.deepcopy(EXPECTED_COUNTS)
        del omitted_nonzero["missing_image"]
        variants.append(omitted_nonzero)

        for counts in variants:
            with (
                self.subTest(counts=counts),
                self.assertRaisesRegex(ValueError, "manifest counts"),
            ):
                external_inference.validate_manifest(manifest(rows, counts))

    def test_count_structure_values_and_reason_vocabulary_are_strict(self):
        rows = explicit_rows()
        malformed = []
        missing_counts = manifest(rows, copy.deepcopy(EXPECTED_COUNTS))
        del missing_counts["counts"]
        missing_counts["cohort_sha256"] = splitting.canonical_hash(missing_counts)
        malformed.append(missing_counts)
        malformed.append(manifest(rows, []))
        for required in ("input", "retained"):
            counts = copy.deepcopy(EXPECTED_COUNTS)
            del counts[required]
            malformed.append(manifest(rows, counts))
        for bad in (-1, True, 1.5, "1"):
            counts = copy.deepcopy(EXPECTED_COUNTS)
            counts["retained"] = bad
            malformed.append(manifest(rows, counts))
        counts = copy.deepcopy(EXPECTED_COUNTS)
        counts["unsupported"] = 0
        malformed.append(manifest(rows, counts))
        unknown_reason = explicit_rows()
        unknown_reason[2]["reason"] = "excluded_other"
        malformed.append(manifest(unknown_reason, copy.deepcopy(EXPECTED_COUNTS)))

        for value in malformed:
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    ValueError, "manifest counts|manifest rows|external manifest"
                ),
            ):
                external_inference.validate_manifest(value)

    def test_nonfinite_json_counts_fail_at_strict_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            for token in ("NaN", "Infinity", "-Infinity"):
                path.write_text(
                    '{"schema_version":1,"purpose":"external","rows":[],'
                    f'"counts":{{"input":{token},"retained":0}},'
                    '"cohort_sha256":"unused"}'
                )
                with (
                    self.subTest(token=token),
                    self.assertRaisesRegex(ValueError, "nonfinite"),
                ):
                    from src import run_contract

                    run_contract._json(path)

    def test_wrong_retained_count_rejects_before_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            fixture.manifest["counts"]["retained"] = 999
            fixture.manifest["cohort_sha256"] = splitting.canonical_hash(
                {
                    key: value
                    for key, value in fixture.manifest.items()
                    if key != "cohort_sha256"
                }
            )
            write(root / "manifest.json", fixture.manifest)
            write(
                root / "audit.json",
                {
                    "clear": True,
                    "manifest_sha256": digest(root / "manifest.json"),
                    "exact": [],
                    "near": [],
                },
            )
            kwargs = fixture.kwargs()
            with (
                patch.object(external_inference, "predict_raw") as predictor,
                self.assertRaisesRegex(ValueError, "manifest counts"),
            ):
                external_inference.evaluate_run(**kwargs)
            predictor.assert_not_called()
            self.assertFalse((root / "output").exists())

    def test_fully_excluded_manifest_does_not_become_inference_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            for row in fixture.manifest["rows"]:
                row["reason"] = "excluded_modality"
            fixture.manifest["counts"] = {
                "input": 6,
                "retained": 0,
                "excluded_modality": 6,
            }
            fixture.manifest["cohort_sha256"] = splitting.canonical_hash(
                {
                    key: value
                    for key, value in fixture.manifest.items()
                    if key != "cohort_sha256"
                }
            )
            write(root / "manifest.json", fixture.manifest)
            write(
                root / "audit.json",
                {
                    "clear": True,
                    "manifest_sha256": digest(root / "manifest.json"),
                    "exact": [],
                    "near": [],
                },
            )
            external_inference.validate_manifest(fixture.manifest)
            kwargs = fixture.kwargs()
            with (
                patch.object(external_inference, "predict_raw") as predictor,
                self.assertRaisesRegex(ValueError, "empty or duplicated"),
            ):
                external_inference.evaluate_run(**kwargs)
            predictor.assert_not_called()
            self.assertFalse((root / "output").exists())
