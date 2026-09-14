"""Strict, source-specific metadata normalization for the binary lesion task."""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

HAM_LABELS = {"mel": 1, "nv": 0, "bkl": 0, "df": 0, "vasc": 0}
HAM_EXCLUDED = {"bcc", "scc"}
ISIC_DIAGNOSES = {
    ("Benign", "Benign melanocytic proliferations", "Nevus"): ("nv", 0),
    ("Benign", "Benign epidermal proliferations", "Pigmented benign keratosis"): (
        "bkl",
        0,
    ),
    ("Benign", "Benign soft tissue proliferations - Vascular", ""): ("vasc", 0),
    (
        "Benign",
        "Benign soft tissue proliferations - Fibro-histiocytic",
        "Dermatofibroma",
    ): ("df", 0),
    ("Malignant", "Malignant melanocytic proliferations (Melanoma)", "Melanoma, NOS"): (
        "mel",
        1,
    ),
    (
        "Malignant",
        "Malignant adnexal epithelial proliferations - Follicular",
        "Basal cell carcinoma",
    ): ("bcc", None),
    (
        "Malignant",
        "Malignant epidermal proliferations",
        "Squamous cell carcinoma, NOS",
    ): ("scc", None),
    (
        "Indeterminate",
        "Indeterminate epidermal proliferations",
        "Solar or actinic keratosis",
    ): ("akiec", None),
}
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


@dataclass(frozen=True)
class CohortResult:
    frame: pd.DataFrame
    outcomes: list[dict[str, str]]
    counts: dict[str, int]


class EmptyCohortError(ValueError):
    def __init__(self, outcomes: list[dict[str, str]], counts: dict[str, int]):
        super().__init__("empty cohort")
        self.outcomes = outcomes
        self.counts = counts


def _field(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _validate_id(value: object, name: str) -> str:
    result = _field(value)
    if not ID_PATTERN.fullmatch(result):
        raise ValueError(f"invalid or missing {name}: {result!r}")
    return result


def _diagnosis(row: pd.Series, source: str) -> tuple[str, int | None, str]:
    if source == "ham10000":
        code = _field(row["dx"]).lower()
        if "diagnosis_1" in row.index:
            first = _field(row["diagnosis_1"])
            if first in {"Benign", "Malignant"} and code in HAM_LABELS:
                expected = "Malignant" if code == "mel" else "Benign"
                if first != expected:
                    raise ValueError(
                        f"conflicting diagnosis fields for {row['image_id']}"
                    )
        for field_index in (2, 3):
            column = f"diagnosis_{field_index}"
            if column in row.index:
                known = {
                    key[field_index - 1]: result[0]
                    for key, result in ISIC_DIAGNOSES.items()
                    if key[field_index - 1]
                }
                value = _field(row[column])
                if value in known and known[value] != code:
                    raise ValueError(
                        f"conflicting diagnosis fields for {row['image_id']}"
                    )
        if code in HAM_LABELS:
            return code, HAM_LABELS[code], "retained"
        if code in HAM_EXCLUDED:
            return code, None, "excluded_malignancy"
        if code == "akiec":
            return code, None, "excluded_indeterminate"
        return code, None, "unknown_diagnosis"

    diagnosis = tuple(
        _field(row[column]) for column in ("diagnosis_1", "diagnosis_2", "diagnosis_3")
    )
    if diagnosis in ISIC_DIAGNOSES:
        code, target = ISIC_DIAGNOSES[diagnosis]
        if (
            "dx" in row.index
            and _field(row["dx"])
            and _field(row["dx"]).lower() != code
        ):
            raise ValueError(f"conflicting diagnosis fields for {row['isic_id']}")
        if target is not None:
            return code, target, "retained"
        reason = "excluded_indeterminate" if code == "akiec" else "excluded_malignancy"
        return code, None, reason
    first, second, third = diagnosis
    second_codes = {item[1]: result[0] for item, result in ISIC_DIAGNOSES.items()}
    third_codes = {
        item[2]: result[0] for item, result in ISIC_DIAGNOSES.items() if item[2]
    }
    if first in {"Benign", "Malignant", "Indeterminate"}:
        if second and not second.startswith(first + " "):
            raise ValueError(f"conflicting diagnosis fields for {row['isic_id']}")
        if first == "Malignant" and third in {
            "Nevus",
            "Pigmented benign keratosis",
            "Dermatofibroma",
        }:
            raise ValueError(f"conflicting diagnosis fields for {row['isic_id']}")
        if first == "Benign" and ("Melanoma" in third or "carcinoma" in third.lower()):
            raise ValueError(f"conflicting diagnosis fields for {row['isic_id']}")
        if (
            second in second_codes
            and third in third_codes
            and second_codes[second] != third_codes[third]
        ):
            raise ValueError(f"conflicting diagnosis fields for {row['isic_id']}")
    return "unknown", None, "unknown_diagnosis"


def build_cohort(
    metadata_path: str | Path, images_dir: str | Path, *, source: str
) -> CohortResult:
    """Return sorted eligible rows and an outcome for every source row.

    `source` is selected by the caller from a verified metadata export, never
    inferred from column names or the directory name.
    """
    required = {
        "ham10000": {"image_id", "lesion_id", "dx"},
        "isic2018_task3": {
            "isic_id",
            "lesion_id",
            "diagnosis_1",
            "diagnosis_2",
            "diagnosis_3",
        },
    }
    if source not in required:
        raise ValueError(f"unsupported source: {source}")
    df = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    missing = required[source] - set(df.columns)
    if missing:
        raise ValueError(f"missing metadata columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("empty metadata")
    id_column = "image_id" if source == "ham10000" else "isic_id"
    image_ids = [_validate_id(value, id_column) for value in df[id_column]]
    lesion_ids = [_validate_id(value, "lesion_id") for value in df["lesion_id"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("duplicate image IDs")
    images_root = Path(images_dir).resolve()
    records = []
    outcomes = []
    seen_content = {}
    lesion_codes = {}
    for (_, row), image_id, lesion_id in zip(df.iterrows(), image_ids, lesion_ids):
        code, target, reason = _diagnosis(row, source)
        if lesion_id in lesion_codes and lesion_codes[lesion_id] != code:
            raise ValueError(f"conflicting lesion diagnoses for {lesion_id}")
        lesion_codes[lesion_id] = code
        if reason == "retained":
            image_path = images_root / f"{image_id}.jpg"
            if not image_path.is_file():
                reason = "missing_image"
            elif not image_path.resolve().is_relative_to(images_root):
                raise ValueError(f"unsafe image path: {image_id}")
            elif cv2.imread(str(image_path)) is None:
                reason = "corrupt_image"
            else:
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if digest in seen_content:
                    raise ValueError(
                        f"duplicate image content: {seen_content[digest]} and {image_id}"
                    )
                seen_content[digest] = image_id
        outcomes.append({"image_id": image_id, "reason": reason, "diagnosis": code})
        if reason == "retained":
            records.append(
                {
                    "isic_id": image_id,
                    "lesion_id": lesion_id,
                    "dx": code,
                    "target": target,
                }
            )
    outcomes.sort(key=lambda item: item["image_id"])
    counts = dict(sorted(Counter(item["reason"] for item in outcomes).items()))
    counts["input"] = len(outcomes)
    if not records:
        raise EmptyCohortError(outcomes, counts)
    frame = (
        pd.DataFrame.from_records(records).sort_values("isic_id").reset_index(drop=True)
    )
    frame["target"] = frame["target"].astype("int64")
    return CohortResult(frame=frame, outcomes=outcomes, counts=counts)
