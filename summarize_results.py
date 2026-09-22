"""Recompute checked run metrics from prediction records alone."""

import argparse
import json
from pathlib import Path

from src import reporting, run_contract


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a completed run and recompute its binary metrics."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, help="Optional path outside the run directory."
    )
    args = parser.parse_args()
    record, _ = run_contract.validate_run(args.run_dir)
    result = {
        "schema_version": 2,
        "run_id": record["run_id"],
        "split_hash": record["split_hash"],
        **run_contract.recompute_report(args.run_dir),
    }
    if args.output:
        if args.output.resolve().is_relative_to(args.run_dir.resolve()):
            raise ValueError("cannot modify a completed run")
        reporting.save_json(result, args.output)
    else:
        print(json.dumps(reporting.to_builtin(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
