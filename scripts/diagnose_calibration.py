"""Inspect saved v1 calibration ranking without fitting or modifying a run."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import calibration, reporting, run_contract  # noqa: E402


def diagnose(run_dir):
    path = Path(run_dir)
    record, _ = run_contract.validate_run(path)
    if record["pipeline"] != "classical_gmm":
        raise ValueError("clipping diagnosis requires a saved GMM run")
    policy = json.loads((path / "models/decision_policy.json").read_text())
    if policy["schema_version"] != 1:
        raise ValueError("clipping diagnosis requires legacy policy v1")
    file = path / record["prediction_files"]["development"]
    frame = run_contract._read_csv(file)
    raw = frame.raw_score.to_numpy()
    fitted = calibration.apply_policy(policy, raw)["calibrated_probability"]
    np.testing.assert_array_equal(fitted, frame.prob_melanoma.to_numpy())
    labels = frame.target.to_numpy()

    def ranking(values):
        _, counts = np.unique(values, return_counts=True)
        return {
            "roc_auc": float(roc_auc_score(labels, values))
            if len(np.unique(labels)) == 2
            else None,
            "distinct_scores": len(counts),
            "tied_pairs": int(np.sum(counts * (counts - 1) // 2)),
        }

    return {
        "run_id": record["run_id"],
        "policy_version": 1,
        "role": "development",
        "count": len(raw),
        "fit_performed": False,
        "raw_ranking": ranking(raw),
        "fitted_ranking": ranking(fitted),
        "below_clip": int(np.sum(raw < 1e-10)),
        "above_clip": int(np.sum(raw > 1 - 1e-10)),
        "exact_zero": int(np.sum(raw == 0)),
        "exact_one": int(np.sum(raw == 1)),
        "slope": policy["calibrator"]["slope"],
        "policy_sha256": run_contract.sha256(path / "models/decision_policy.json"),
        "predictions_sha256": run_contract.sha256(file),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or any(
        args.output.resolve().is_relative_to(p.resolve()) for p in args.run_dir
    ):
        raise ValueError("diagnosis requires a new output outside the saved runs")
    rows = [diagnose(path) for path in args.run_dir]
    reporting.save_json(
        {"schema_version": 1, "analysis": "saved_v1_clipping_no_refit", "rows": rows},
        args.output,
    )


if __name__ == "__main__":
    main()
