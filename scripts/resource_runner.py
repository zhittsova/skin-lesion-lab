"""Execute an explicit command manifest only after a fresh resource decision."""

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.resource_plan import plan_capacity  # noqa: E402
from src.resource_runtime import (  # noqa: E402
    device_lease,
    host_available_bytes,
    run_bounded,
    stop_owned_group,
)


def validate_jobs(config, jobs):
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a nonempty list")
    ids = set()
    counts = collections.Counter()
    for job in jobs:
        if not isinstance(job, dict) or set(job) != {"id", "workload", "argv", "cwd"}:
            raise ValueError("each job requires id, workload, argv and cwd")
        if (
            not isinstance(job["id"], str)
            or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", job["id"])
            or job["id"] in ids
        ):
            raise ValueError("job IDs must be unique plain filenames")
        if not isinstance(job["workload"], str):
            raise ValueError("workload must be a string")
        if (
            not isinstance(job["argv"], list)
            or not job["argv"]
            or any(not isinstance(a, str) or not a for a in job["argv"])
        ):
            raise ValueError("argv must be a nonempty string list")
        if (
            not isinstance(job["cwd"], str)
            or not Path(job["cwd"]).is_absolute()
            or not Path(job["cwd"]).is_dir()
        ):
            raise ValueError("cwd must be an existing absolute directory")
        ids.add(job["id"])
        counts[job["workload"]] += 1
    if dict(counts) != config["jobs"]:
        raise ValueError(
            "manifest workload counts differ from measured resource config"
        )
    return jobs


def validate_resource_flags(argv, device, workers):
    for flag, expected in (("--device", device), ("--num-workers", str(workers))):
        occurrences = []
        for index, token in enumerate(argv):
            option = token.split("=", 1)[0]
            if option.startswith("--") and flag.startswith(option):
                if option != flag:
                    raise ValueError("abbreviated resource flags are not allowed")
                value = (
                    token.split("=", 1)[1]
                    if "=" in token
                    else (argv[index + 1] if index + 1 < len(argv) else None)
                )
                occurrences.append(value)
        if occurrences != [expected]:
            raise ValueError(f"exactly one {flag} {expected} is required")


def validate_profile(report, jobs, plan_path):
    from src import benchmark, benchmark_registry, run_contract

    if plan_path is None:
        raise ValueError("a frozen benchmark plan is required")
    root = Path(__file__).resolve().parents[1]
    plan = benchmark_registry.load_plan(plan_path)
    benchmark_registry.check_evaluator_source(plan)
    expected_profile = {
        "batch_size": 64,
        "image_size": 128,
        "epochs": 20,
        "candidates": 2,
        "mc_samples": 30,
        "counts": {
            "train": 5491,
            "selection": 1366,
            "calibration": 922,
            "development": 1395,
        },
        "deep_source_sha256": run_contract.sha256(root / "src/deep.py"),
    }
    if report.get("pilot_profile") != expected_profile:
        raise ValueError("pilot profile differs from the measured workload")
    scheduled = {j["run_id"]: j for j in plan["jobs"]}
    canonical = {j["run_id"]: j for j in benchmark.planned_runs()}
    workers = report["decision"]["selected_loader_workers"]
    device = report["decision"]["selected_device"].split(":")[0]
    if plan["device"] != device:
        raise ValueError("frozen device differs from the resource decision")
    for job in jobs:
        declared = scheduled.get(job["id"])
        known = canonical.get(job["id"])
        if not known or known["pipeline"] != "deep" or not declared:
            raise ValueError("executor supports only measured, scheduled deep jobs")
        expected_config = {**known["config"], "num_workers": workers}
        if declared["config"] != expected_config:
            raise ValueError("frozen job exceeds or differs from the measured profile")
        workload = (
            "cnn"
            if expected_config["architecture"] == "small_cnn"
            else (
                "efficientnet_full"
                if expected_config["fine_tune_backbone"]
                else "efficientnet_head"
            )
        )
        if job["workload"] != workload or Path(job["cwd"]).resolve() != root:
            raise ValueError("workload or checkout differs from the measured profile")
        argv = declared["argv"]
        paths = {
            name: Path(argv[argv.index(flag) + 1])
            for name, flag in (
                ("metadata", "--metadata-path"),
                ("images", "--images-dir"),
                ("split_manifest", "--split-manifest"),
                ("runs", "--runs-dir"),
            )
        }
        for name, path in (
            ("metadata", paths["metadata"]),
            ("split_manifest", paths["split_manifest"]),
            ("dependency_lock", root / "uv.lock"),
        ):
            if run_contract.sha256(path) != plan["inputs"][name]:
                raise ValueError("frozen input changed: " + name)
        manifest = json.loads(paths["split_manifest"].read_text())
        if {k: len(v) for k, v in manifest["partitions"].items()} != expected_profile[
            "counts"
        ]:
            raise ValueError("cohort size differs from the timed workload")
        expected_argv = benchmark.training_command(
            declared,
            python=sys.executable,
            metadata=paths["metadata"],
            images=paths["images"],
            manifest=paths["split_manifest"],
            runs=paths["runs"],
            source=plan["dataset_source"],
            device=device,
        )
        if argv != expected_argv or job["argv"] != expected_argv:
            raise ValueError("command differs from canonical frozen job")


def acquire_for_run(config, output):
    path = output / "acquisition.json"
    argv = [
        sys.executable,
        str(Path(__file__).with_name("resource_pilot.py")),
        "inventory",
        "--devices",
        json.dumps(config["preferred_devices"]),
        "--output",
        str(path),
    ]
    attempt = run_bounded(
        argv,
        output / "acquisition.log",
        min(30, config["pilot_budget_seconds"]),
        grace_seconds=1,
    )
    if attempt["returncode"] != 0:
        raise RuntimeError("resource acquisition failed; see acquisition.log")
    return json.loads(path.read_text())["inventory"]


def execute(report, jobs, output, benchmark_plan=None):
    started = time.monotonic()
    config = report["config"]
    validate_jobs(config, jobs)
    selected = report["decision"]["selected_device"]
    if report["decision"]["status"] != "ready":
        raise ValueError("pilot decision is blocked")
    if selected not in {"cpu", "mps:0", "cuda:0"}:
        raise ValueError("executor currently supports the first accelerator only")
    validate_profile(report, jobs, benchmark_plan)
    cleanup_reserve = 5 + 3 * config["max_concurrent_jobs"]
    deadline = started + config["wall_budget_seconds"] - cleanup_reserve
    if deadline - time.monotonic() <= 3:
        raise ValueError("wall budget cannot cover acquisition and cleanup")
    output.mkdir(parents=True, exist_ok=False)
    key = hashlib.sha256(selected.encode()).hexdigest()[:16]
    lock = (
        Path(tempfile.gettempdir()) / f"skin-lesion-resource-{os.getuid()}-{key}.lock"
    )
    with device_lease(lock):
        inventory = acquire_for_run(
            {
                **config,
                "preferred_devices": [selected],
                "pilot_budget_seconds": min(
                    config["pilot_budget_seconds"], deadline - time.monotonic() - 3
                ),
            },
            output,
        )
        config = {**config, "as_of_unix_seconds": time.time()}
        decision = plan_capacity(config, inventory, report["measurements"])
        if decision["status"] != "ready" or decision["selected_device"] != selected:
            raise ValueError(
                "fresh resource check blocked execution: " + json.dumps(decision)
            )
        if (
            decision["selected_loader_workers"]
            != report["decision"]["selected_loader_workers"]
        ):
            raise ValueError("measured loader choice changed; prepare a fresh manifest")
        # Bind resource-sensitive training flags; do not silently rewrite a
        # frozen experiment to match a new device or loader seed policy.
        expected_device = selected.split(":")[0]
        for job in jobs:
            validate_resource_flags(
                job["argv"], expected_device, decision["selected_loader_workers"]
            )
        rows = [{"id": j["id"], "status": "pending"} for j in jobs]
        state = {
            "decision": decision,
            "jobs": rows,
            "status": "running",
            "pid": os.getpid(),
        }
        active = {}
        env = dict(
            os.environ,
            OMP_NUM_THREADS=str(config["threads_per_job"]),
            OPENBLAS_NUM_THREADS=str(config["threads_per_job"]),
            VECLIB_MAXIMUM_THREADS=str(config["threads_per_job"]),
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            PYTHONUNBUFFERED="1",
        )

        def save():
            state["updated_unix_seconds"] = time.time()
            temporary = output / "state.tmp"
            temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
            temporary.replace(output / "state.json")

        try:
            while active or any(r["status"] == "pending" for r in rows):
                if (
                    time.monotonic() >= deadline
                    or shutil.disk_usage(output).free < config["reserve_disk_bytes"]
                ):
                    raise RuntimeError("wall-time or disk limit reached")
                for index, (process, log) in list(active.items()):
                    if process.poll() is not None:
                        stop_owned_group(process, grace_seconds=1)
                        rows[index].update(
                            status="completed" if process.returncode == 0 else "failed",
                            returncode=process.returncode,
                        )
                        log.close()
                        del active[index]
                for index, (job, row) in enumerate(zip(jobs, rows, strict=True)):
                    if len(active) >= decision["attainable_parallelism"]:
                        break
                    if row["status"] != "pending":
                        continue
                    required = decision["workload_plans"][job["workload"]][
                        "peak_host_bytes"
                    ]
                    if selected.startswith("mps"):
                        required += decision["workload_plans"][job["workload"]][
                            "peak_device_bytes"
                        ]
                    if host_available_bytes() < config["reserve_host_bytes"] + required:
                        if not active:
                            raise RuntimeError(
                                "available memory fell below measured requirement"
                            )
                        break
                    log = (output / f"{job['id']}.log").open("x")
                    try:
                        process = subprocess.Popen(
                            job["argv"],
                            cwd=job["cwd"],
                            env=env,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    except BaseException:
                        log.close()
                        raise
                    active[index] = (process, log)
                    row.update(status="running", pid=process.pid)
                save()
                time.sleep(1)
            state["status"] = (
                "completed"
                if all(r["status"] == "completed" for r in rows)
                else "completed_with_failures"
            )
        except BaseException as error:
            state.update(status="stopped", reason=str(error))
            raise
        finally:
            for index, (process, log) in active.items():
                stop_owned_group(process, grace_seconds=1)
                log.close()
                rows[index].update(status="interrupted", returncode=process.returncode)
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--decision-sha256", required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--benchmark-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    encoded = args.decision.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != args.decision_sha256:
        parser.error("resource decision digest differs")
    execute(
        json.loads(encoded),
        json.loads(args.jobs.read_text()),
        args.output,
        args.benchmark_plan,
    )


if __name__ == "__main__":

    def interrupted(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    main()
