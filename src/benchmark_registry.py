"""Frozen benchmark schedules and checked completed-run inputs."""

import json
from datetime import datetime
from pathlib import Path

from src import run_contract, splitting


def save_plan(path, payload):
    """Create a plan once; its digest binds the complete schedule and protocol."""
    path = Path(path)
    encoded = json.dumps(
        {"plan": payload, "sha256": splitting.canonical_hash(payload)},
        indent=2,
        allow_nan=False,
    )
    with path.open("x", encoding="utf-8") as output:
        output.write(encoded + "\n")


def load_plan(path):
    envelope = json.loads(Path(path).read_text())
    if splitting.canonical_hash(envelope["plan"]) != envelope["sha256"]:
        raise ValueError("benchmark plan hash mismatch")
    return envelope["plan"]


def check_scheduled_run(plan, job, record):
    if datetime.fromisoformat(record["created_utc"]) <= datetime.fromisoformat(
        plan["created_utc"]
    ):
        raise ValueError("run must start after the frozen plan")
    if record["source"] != plan["source"] or record["inputs"] != plan["inputs"]:
        raise ValueError("run source or inputs differ from frozen plan")
    expected_pipeline = (
        "deep_" + job["config"]["architecture"]
        if job["pipeline"] == "deep"
        else "classical_" + job["config"]["model"]
    )
    if record["run_id"] != job["run_id"] or record["pipeline"] != expected_pipeline:
        raise ValueError("run identity differs from scheduled job")
    expected = {**job["config"], "source": plan["dataset_source"]}
    if job["pipeline"] == "deep":
        expected["device"] = plan["device"]
    for key, value in expected.items():
        if record["config"].get(key) != value:
            raise ValueError(f"run configuration differs from schedule: {key}")


def inventory(plan, runs_dir):
    """Account for all jobs; failed or invalid jobs cannot enter comparisons."""
    rows, models = [], {}
    expected_ids = {j["run_id"] for j in plan["jobs"]}
    runs_dir = Path(runs_dir)
    unexpected = (
        sorted(
            p.name
            for p in runs_dir.iterdir()
            if p.is_dir() and p.name not in expected_ids
        )
        if runs_dir.exists()
        else []
    )
    for job in plan["jobs"]:
        row = {
            "run_id": job["run_id"],
            "stratum": job["stratum"],
            "seed": job["seed"],
            "candidate_count": job["candidate_count"],
            "status": "not_started",
        }
        path = runs_dir / job["run_id"]
        if path.exists():
            try:
                record = json.loads((path / "run.json").read_text())
                row["status"] = record["status"]
                row["run_record_sha256"] = run_contract.sha256(path / "run.json")
                if record["status"] == "completed":
                    record, frame = run_contract.validate_run(path)
                    check_scheduled_run(plan, job, record)
                    development = frame.loc[frame.role == "development"]
                    manifest = json.loads(
                        (path / "inputs/split-manifest.json").read_text()
                    )
                    if development.empty or set(development.image_id) != set(
                        manifest["partitions"]["development"]
                    ):
                        raise ValueError("run lacks the full frozen development cohort")
                    models.setdefault(job["stratum"], {})[job["seed"]] = development
                    row["artifacts_verified"] = len(record["artifacts"])
                elif "failure" in record:
                    row["failure"] = record["failure"]
            except (OSError, ValueError, KeyError, TypeError) as error:
                row["status"] = "invalid"
                row["error"] = str(error)
        rows.append(row)
    return {
        "jobs": rows,
        "unexpected_run_directories": unexpected,
        "complete": not unexpected
        and bool(rows)
        and all(r["status"] == "completed" for r in rows),
    }, models


def check_evaluator_source(plan):
    current, _, _ = run_contract._source_record()
    if current != plan["source"]:
        raise ValueError("current evaluator source differs from frozen benchmark")
