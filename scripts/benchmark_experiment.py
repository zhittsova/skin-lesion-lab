"""Freeze a declared benchmark schedule or report checked saved runs."""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import benchmark, benchmark_registry, run_contract  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--directory", type=Path, required=True)
    freeze.add_argument("--metadata-path", type=Path, required=True)
    freeze.add_argument("--images-dir", type=Path, required=True)
    freeze.add_argument("--split-manifest", type=Path, required=True)
    freeze.add_argument(
        "--source", choices=["ham10000", "isic2018_task3"], required=True
    )
    freeze.add_argument("--device", choices=["cpu", "mps", "cuda"], required=True)
    freeze.add_argument("--max-seconds", type=int, required=True)
    freeze.add_argument("--allow-weight-download", action="store_true")
    for name in ("status", "report"):
        sub = commands.add_parser(name)
        sub.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.command == "freeze":
        if args.max_seconds <= 0:
            parser.error("--max-seconds must be positive")
        # A clean published source commit makes the schedule portable and auditable.
        source, diff, untracked = run_contract._source_record()
        if diff or untracked:
            raise ValueError(
                "commit reviewed implementation before freezing a benchmark"
            )
        protocol = (ROOT / "docs/evaluation-protocol.md").read_text()
        manifest = json.loads(args.split_manifest.read_text())
        if manifest["purpose"] != "development":
            raise ValueError("only a development manifest may enter this benchmark")
        paths = {
            "metadata": args.metadata_path.resolve(),
            "split_manifest": args.split_manifest.resolve(),
            "dependency_lock": ROOT / "uv.lock",
        }
        jobs = benchmark.planned_runs()
        plan = {
            "schema_version": 1,
            "purpose": "development",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "protocol": protocol,
            "protocol_sha256": run_contract.sha256(
                ROOT / "docs/evaluation-protocol.md"
            ),
            "inputs": {k: run_contract.sha256(p) for k, p in paths.items()},
            "dataset_source": args.source,
            "device": args.device,
            "budget_seconds": args.max_seconds,
            "allow_weight_download": args.allow_weight_download,
            "hardware": platform.platform(),
            "split_hash": manifest["split_hash"],
            "grouping": manifest["grouping"],
            "jobs": jobs,
        }
        for job in jobs:
            job["argv"] = benchmark.training_command(
                job,
                python=sys.executable,
                metadata=paths["metadata"],
                images=args.images_dir.resolve(),
                manifest=paths["split_manifest"],
                runs=directory / "runs",
                source=args.source,
                device=args.device,
            )
        directory.mkdir(parents=True, exist_ok=True)
        benchmark_registry.save_plan(directory / "plan.json", plan)
        print(
            json.dumps(
                {
                    "plan": str(directory / "plan.json"),
                    "runs": len(jobs),
                    "candidates": sum(j["candidate_count"] for j in jobs),
                    "training_started": False,
                }
            )
        )
        return
    plan = benchmark_registry.load_plan(directory / "plan.json")
    status, models = benchmark_registry.inventory(plan, directory / "runs")
    if args.command == "status":
        print(json.dumps(status, indent=2, allow_nan=False))
        return
    if not status["complete"]:
        print(json.dumps(status, indent=2, allow_nan=False))
        raise ValueError("complete checked matrix required before benchmark reporting")
    benchmark_registry.check_evaluator_source(plan)
    report = benchmark.paired_report(models, reference="logistic")
    report.update(
        plan_sha256=run_contract.sha256(directory / "plan.json"),
        protocol_sha256=plan["protocol_sha256"],
        grouping=plan["grouping"],
        primary_contrast="efficientnet_full_unweighted minus logistic",
        other_contrasts="exploratory",
        registry=status,
    )
    # Recompute calibration bins and referral summaries from the saved policies.
    report["calibration_and_referral"] = {
        job["run_id"]: run_contract.recompute_calibration_report(
            directory / "runs" / job["run_id"]
        )
        for job in plan["jobs"]
    }
    output = directory / "benchmark-report.json"
    with output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
