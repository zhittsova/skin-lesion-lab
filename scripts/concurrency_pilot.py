"""Measure overlapping GPU workloads before changing any full-run concurrency."""

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.resource_runtime import (  # noqa: E402
    device_lease,
    host_available_bytes,
    run_bounded,
)


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def capacity_reason(config, available, slots, host_per_job, gpu_per_job):
    cpu = slots * (config["threads_per_job"] + config["loader_workers"])
    if not config["allow_cpu_oversubscription"] and cpu > available["cpu_count"]:
        return "cpu_count"
    if slots * host_per_job + config["reserve_host_bytes"] > available["host_free"]:
        return "host_memory"
    if slots * gpu_per_job + config["reserve_device_bytes"] > available["gpu_free"]:
        return "device_memory"
    if available["disk_free"] < config["reserve_disk_bytes"]:
        return "disk_space"
    return None


def summarize_group(samples):
    if not samples or any(s.get("loss_finite") is not True for s in samples):
        raise ValueError("all samples must report finite loss")
    starts = [s["workload_started_unix_seconds"] for s in samples]
    ends = [s["workload_finished_unix_seconds"] for s in samples]
    if any(not math.isfinite(x) for x in starts + ends) or any(
        e <= s for s, e in zip(starts, ends)
    ):
        raise ValueError("invalid workload interval")
    overlap = min(ends) - max(starts)
    if overlap <= 0:
        raise ValueError("workload intervals did not overlap")
    elapsed = max(ends) - min(starts)
    return {
        "elapsed_seconds": elapsed,
        "overlap_seconds": overlap,
        "pilot_workloads_per_second": len(samples) / elapsed,
        "start_spread_seconds": max(starts) - min(starts),
    }


def summarize_trials(trials, repeats):
    rates = {}
    for trial in trials:
        if "measurement" in trial:
            rates.setdefault(trial["slots"], []).append(
                trial["measurement"]["pilot_workloads_per_second"]
            )
    baseline = statistics.median(rates[1]) if len(rates.get(1, [])) >= repeats else None
    levels = {}
    for slots, values in sorted(rates.items()):
        median = statistics.median(values)
        levels[str(slots)] = {
            "samples": len(values),
            "enough_repeats": len(values) >= repeats,
            "median_pilot_workloads_per_second": median,
            "speedup_vs_one": median / baseline if baseline else None,
        }
    eligible = [s for s in rates if baseline and len(rates[s]) >= repeats]
    best = (
        max(eligible, key=lambda s: statistics.median(rates[s])) if eligible else None
    )
    return {
        "levels": levels,
        "best_observed_slots": best,
        "full_run_authorized": False,
        "scope": "Generated-image EfficientNet full workload; provisional throughput comparison only",
    }


def validate(config):
    keys = {
        "version",
        "device",
        "workload",
        "concurrency_levels",
        "repeats",
        "batches",
        "dataset_batches",
        "threads_per_job",
        "loader_workers",
        "allow_cpu_oversubscription",
        "paid_budget",
        "pilot_budget_seconds",
        "group_budget_seconds",
        "ready_budget_seconds",
        "reserve_host_bytes",
        "reserve_device_bytes",
        "reserve_disk_bytes",
    }
    if set(config) != keys or config["version"] != 1 or config["paid_budget"] != 0:
        raise ValueError("invalid or paid concurrency config")
    if config["device"] != "cuda:0" or config["workload"] != "efficientnet_full":
        raise ValueError("this bounded pilot requires CUDA:0 and EfficientNet full")
    levels = config["concurrency_levels"]
    if not isinstance(levels, list) or levels not in ([1, 2], [1, 2, 3]):
        raise ValueError("concurrency levels must be [1,2] or [1,2,3]")
    for name in keys - {
        "version",
        "device",
        "workload",
        "concurrency_levels",
        "allow_cpu_oversubscription",
        "paid_budget",
    }:
        if type(config[name]) is not int or config[name] < (
            0 if name == "loader_workers" else 1
        ):
            raise ValueError("invalid positive integer: " + name)
    if type(config["allow_cpu_oversubscription"]) is not bool:
        raise ValueError("CPU oversubscription must be explicit")
    if (
        config["repeats"] < 2
        or config["batches"] < 16
        or config["pilot_budget_seconds"] > 1200
    ):
        raise ValueError("need two repeats, at least16 batches and at most1200 seconds")
    if (
        config["group_budget_seconds"] > config["pilot_budget_seconds"]
        or config["ready_budget_seconds"] >= config["group_budget_seconds"]
    ):
        raise ValueError("invalid group/readiness budgets")
    return config


def telemetry(output):
    csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=3,
    )
    free, used, utilization = [float(x.strip()) for x in csv.strip().split(",")]
    if not all(math.isfinite(x) and x >= 0 for x in (free, used, utilization)):
        raise ValueError("invalid GPU telemetry")
    return {
        "unix_seconds": time.time(),
        "gpu_free": int(free * 1024**2),
        "gpu_used": int(used * 1024**2),
        "gpu_utilization_percent": utilization,
        "host_free": host_available_bytes(),
        "disk_free": shutil.disk_usage(output).free,
        "cpu_count": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
    }


def run(config, output):
    from src.concurrency_runtime import run_group

    config = validate(config)
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + config["pilot_budget_seconds"]
    report = {
        "config": config,
        "trials": [],
        "status": "running",
        "full_run_started": False,
        "source_hashes": {
            n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
            for n in [
                "src/deep.py",
                "src/resource_runtime.py",
                "src/concurrency_runtime.py",
                "scripts/resource_pilot.py",
                "scripts/concurrency_pilot.py",
            ]
        },
        "limitations": [
            "Repeated generated images may benefit from filesystem cache",
            "Longer loops still include loader startup; workers held at configured value",
            "No accuracy or clinical result; no full-run admission",
            "Host/GPU memory sampled about once per second; transient peaks may be missed",
        ],
    }
    key = hashlib.sha256(config["device"].encode()).hexdigest()[:16]
    lock = (
        Path(tempfile.gettempdir()) / f"skin-lesion-resource-{os.getuid()}-{key}.lock"
    )
    host_per_job, gpu_per_job = 3 * 1024**3, 4 * 1024**3
    try:
        with device_lease(lock):
            acquisition = run_bounded(
                [
                    sys.executable,
                    str(ROOT / "scripts/resource_pilot.py"),
                    "inventory",
                    "--devices",
                    json.dumps([config["device"]]),
                    "--output",
                    str(output / "inventory.json"),
                ],
                output / "inventory.log",
                min(40, config["pilot_budget_seconds"] - 15),
                grace_seconds=0.5,
            )
            report["acquisition"] = acquisition
            if acquisition["returncode"] != 0:
                raise RuntimeError("CUDA acquisition failed")
            inventory = json.loads((output / "inventory.json").read_text())
            if not inventory["inventory"]["devices"][0]["acquired"]:
                raise RuntimeError("CUDA device unavailable")
            for repeat in range(config["repeats"]):
                levels = (
                    config["concurrency_levels"]
                    if repeat % 2 == 0
                    else list(reversed(config["concurrency_levels"]))
                )
                for slots in levels:
                    remaining = deadline - time.monotonic() - 15
                    if remaining < 30:
                        raise TimeoutError("pilot wall budget exhausted")
                    available = telemetry(output)
                    reason = capacity_reason(
                        config, available, slots, host_per_job, gpu_per_job
                    )
                    trial = {"slots": slots, "repeat": repeat, "before": available}
                    report["trials"].append(trial)
                    if reason:
                        trial.update(status="skipped", reason=reason)
                        save(output / "decision.json", report)
                        continue
                    if slots > 1 and not any(
                        t.get("measurement") and t["slots"] == 1
                        for t in report["trials"]
                    ):
                        trial.update(status="skipped", reason="baseline_required")
                        continue
                    group = output / f"repeat-{repeat}-slots-{slots}"
                    group.mkdir()
                    commands = []
                    for index in range(slots):
                        commands.append(
                            [
                                sys.executable,
                                str(ROOT / "scripts/resource_pilot.py"),
                                "sample",
                                "--device",
                                config["device"],
                                "--workload",
                                config["workload"],
                                "--workers",
                                str(config["loader_workers"]),
                                "--threads",
                                str(config["threads_per_job"]),
                                "--batches",
                                str(config["batches"]),
                                "--dataset-batches",
                                str(config["dataset_batches"]),
                                "--ready-file",
                                str(group / f"job-{index}.ready"),
                                "--start-file",
                                str(group / "start"),
                                "--output",
                                str(group / f"sample-{index}.json"),
                            ]
                        )
                    snapshots = []
                    last_sample = 0

                    def monitor():
                        nonlocal last_sample
                        now = time.monotonic()
                        if now - last_sample < 1:
                            return None
                        last_sample = now
                        sample = telemetry(output)
                        snapshots.append(sample)
                        save(group / "telemetry.json", snapshots)
                        for field, reserve in (
                            ("host_free", "reserve_host_bytes"),
                            ("gpu_free", "reserve_device_bytes"),
                            ("disk_free", "reserve_disk_bytes"),
                        ):
                            if sample[field] < config[reserve]:
                                return field + "_reserve_exhausted"
                        return None

                    print(
                        f"Repeat {repeat + 1}: measuring {slots} concurrent jobs",
                        flush=True,
                    )
                    outcome = run_group(
                        commands,
                        group,
                        min(remaining, config["group_budget_seconds"]),
                        min(config["ready_budget_seconds"], remaining / 2),
                        monitor=monitor,
                        monitor_timeout_seconds=4,
                    )
                    trial["execution"] = outcome
                    trial["status"] = outcome["status"]
                    if outcome["status"] != "completed":
                        raise RuntimeError(
                            "group stopped: " + str(outcome.get("reason"))
                        )
                    samples = [
                        json.loads((group / f"sample-{i}.json").read_text())
                        for i in range(slots)
                    ]
                    trial["measurement"] = summarize_group(samples)
                    # Keep the largest observed per-process footprint plus CUDA context allowance.
                    host_per_job = max(
                        host_per_job if slots != 1 else 0,
                        *(s["measurement"]["peak_host_bytes"] for s in samples),
                    )
                    gpu_per_job = max(
                        gpu_per_job if slots != 1 else 0,
                        *(
                            s["measurement"]["peak_device_bytes"] + 512 * 1024**2
                            for s in samples
                        ),
                    )
                    report["summary"] = summarize_trials(
                        report["trials"], config["repeats"]
                    )
                    save(output / "decision.json", report)
                    print(json.dumps(trial["measurement"]), flush=True)
            report["status"] = "completed"
    except (
        RuntimeError,
        TimeoutError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        report.update(status="blocked", reason=str(error))
    except BaseException:
        report.update(status="interrupted", reason="controller interrupted")
        raise
    finally:
        report["summary"] = summarize_trials(report["trials"], config["repeats"])
        save(output / "decision.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "summary": report["summary"],
                "reason": report.get("reason"),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output.resolve())


if __name__ == "__main__":

    def interrupted(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    main()
