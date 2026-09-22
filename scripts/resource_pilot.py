"""Probe a current-host allocation and time generated-image workloads before training."""

import argparse
import contextlib
import hashlib
import json
import os
import platform
import resource
import signal
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.resource_runtime import (  # noqa: E402
    device_lease,
    estimate_job_seconds,
    probe_inventory,
    run_bounded,
)


def write_json(path, payload):
    with Path(path).open("x") as output:
        output.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample(args):
    # Set before importing torch, matching deterministic GPU requirements.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from PIL import Image
    from src import deep
    from torch.utils.data import DataLoader

    torch.set_num_threads(args.threads)
    deep.set_seed(17)
    device = torch.device(args.device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    rng = np.random.default_rng(17)
    with tempfile.TemporaryDirectory(prefix="skin-lesion-pilot-") as tmp:
        images = Path(tmp)
        unique_batches = args.dataset_batches or args.batches
        ids = [
            f"generated-{i}" for i in range(args.batch_size * max(2, unique_batches))
        ]
        unique_images = len(ids)
        for image_id in ids:
            Image.fromarray(rng.integers(0, 256, (450, 600, 3), dtype=np.uint8)).save(
                images / f"{image_id}.jpg", quality=85
            )

        # Reuse synthetic content while preserving MC's unique ID/path contract.
        for index in range(len(ids), args.batch_size * args.batches):
            name = f"generated-{index}"
            os.link(
                images / f"generated-{index % unique_images}.jpg",
                images / f"{name}.jpg",
            )
            ids.append(name)

        def loader(batches, train):
            names = ids[: args.batch_size * batches]
            dataset = deep.SkinLesionImageDataset(
                names,
                [i % 2 for i in range(len(names))],
                images,
                deep.build_transforms(args.image_size, train),
            )
            return DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.workers,
                worker_init_fn=deep.seed_worker,
                generator=torch.Generator().manual_seed(17),
                pin_memory=device.type == "cuda",
            )

        model = deep.build_model(
            architecture="small_cnn" if args.workload == "cnn" else "efficientnet_b0",
            dropout=0.3,
            pretrained=args.workload != "cnn",
            freeze_backbone=args.workload != "efficientnet_full",
        ).to(device)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=0.001,
            weight_decay=1e-4,
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        deep.train_one_epoch(model, loader(2, True), criterion, optimizer, device)
        synchronize()
        if args.ready_file:
            write_json(args.ready_file, {"pid": os.getpid(), "ready": True})
            while not args.start_file.exists():
                time.sleep(0.05)
            start_at = json.loads(args.start_file.read_text())["start_unix_seconds"]
            while time.time() < start_at:
                time.sleep(min(0.02, max(0, start_at - time.time())))
        workload_started = time.time()
        start = time.monotonic()
        loss = deep.train_one_epoch(
            model, loader(args.batches, True), criterion, optimizer, device
        )
        synchronize()
        train_seconds = (time.monotonic() - start) / args.batches
        # Includes both deterministic evaluation and stochastic inference; use
        # the slower batch time when extrapolating either phase.
        start = time.monotonic()
        deep.evaluate_loss(model, loader(args.batches, False), criterion, device)
        synchronize()
        eval_seconds = (time.monotonic() - start) / args.batches
        start = time.monotonic()
        deep.predict_with_mc_dropout(model, loader(args.batches, False), device, 2)
        synchronize()
        mc_seconds = (time.monotonic() - start) / (2 * args.batches)
        workload_finished = time.time()
        peak_device = 0
        if device.type == "cuda":
            peak_device = torch.cuda.max_memory_reserved(device)
        elif device.type == "mps":
            peak_device = torch.mps.driver_allocated_memory()
        unit = 1 if platform.system() == "Darwin" else 1024
        peak_host = unit * (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            + max(1, args.workers)
            * resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        )
        measured = {
            "measured_at_unix_seconds": time.time(),
            "seconds_per_job": estimate_job_seconds(
                train_batch_seconds=train_seconds,
                eval_batch_seconds=max(eval_seconds, mc_seconds),
                counts={
                    "train": 5491,
                    "selection": 1366,
                    "calibration": 922,
                    "development": 1395,
                },
                batch_size=args.batch_size,
            ),
            # Conservative allowance for short-probe memory underestimation.
            "peak_host_bytes": int(peak_host * 1.5),
            "peak_device_bytes": int(peak_device * 1.5),
            "required_disk_bytes": 512 * 1024**2,
        }
        write_json(
            args.output,
            {
                "measurement": measured,
                "workload_started_unix_seconds": workload_started,
                "workload_finished_unix_seconds": workload_finished,
                "timed_batches_per_phase": args.batches,
                "unique_images": unique_images,
                "image_paths": len(ids),
                "timing": {
                    "train_batch_seconds": train_seconds,
                    "eval_batch_seconds": eval_seconds,
                    "mc_batch_seconds": mc_seconds,
                },
                "loss_finite": bool(np.isfinite(loss)),
                "workload": args.workload,
                "workers": args.workers,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "torch": torch.__version__,
                "python": platform.python_version(),
                "device": str(device),
                "generated_images": True,
                "gpu_name": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else str(device),
                "limitations": [
                    "Short generated-JPEG probe, not a real-data benchmark or accuracy result",
                    "Per-job estimate excludes startup, hashing, plots and final report",
                    "Generated JPEGs do not establish real-data disk throughput",
                    "Memory allowance is 1.5 times sampled peak; not a reservation",
                    "MPS driver memory is an end-of-probe observation, not a peak counter",
                ],
            },
        )


def pilot(args):
    import shutil

    from src.resource_plan import plan_capacity, validate_config
    from src.resource_runtime import host_available_bytes

    config = validate_config(json.loads(args.config.read_text()))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + config["pilot_budget_seconds"]
    results, attempts, errors = {}, [], {}

    def bounded_probe(devices, name):
        remaining = deadline - time.monotonic() - 15
        if remaining <= 0:
            raise TimeoutError("pilot budget exhausted before device probe")
        path = output / (name + ".json")
        argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            "inventory",
            "--devices",
            json.dumps(devices),
            "--output",
            str(path),
        ]
        attempt = run_bounded(argv, output / (name + ".log"), min(45, remaining))
        attempts.append({**attempt, "phase": name})
        if attempt["returncode"] != 0:
            raise RuntimeError("device probe failed or timed out; see " + name + ".log")
        result = json.loads(path.read_text())
        errors.update(result["errors"])
        return result["inventory"]

    inventory = {
        "measured_at_unix_seconds": time.time(),
        "available_cpu_count": 0,
        "available_host_bytes": 0,
        "available_disk_bytes": shutil.disk_usage(output).free,
        "devices": [],
    }
    with contextlib.ExitStack() as stack:
        leased = []
        for device_id in sorted(config["preferred_devices"]):
            key = hashlib.sha256(device_id.encode()).hexdigest()[:16]
            lock = (
                Path(tempfile.gettempdir())
                / f"skin-lesion-resource-{os.getuid()}-{key}.lock"
            )
            try:
                stack.enter_context(device_lease(lock))
                leased.append(device_id)
            except BlockingIOError:
                errors[device_id] = "resource leased by another cooperating runner"
        try:
            if leased:
                inventory = bounded_probe(leased, "inventory-before")
            for device in inventory["devices"]:
                if not device["acquired"]:
                    continue
                if (
                    inventory["available_host_bytes"]
                    < config["reserve_host_bytes"] + 1024**3
                    or inventory["available_disk_bytes"]
                    < config["reserve_disk_bytes"] + 512 * 1024**2
                    or (
                        device["kind"] == "cuda"
                        and device["free_bytes"]
                        < config["reserve_device_bytes"] + 512 * 1024**2
                    )
                ):
                    errors[device["id"]] = (
                        "insufficient headroom for even the bounded pilot"
                    )
                    continue
                for workload in config["jobs"]:
                    if workload not in {
                        "cnn",
                        "efficientnet_head",
                        "efficientnet_full",
                    }:
                        raise ValueError(f"unsupported pilot workload: {workload}")
                    for workers in config["loader_workers_candidates"]:
                        if (
                            workers + config["threads_per_job"]
                            > inventory["available_cpu_count"]
                        ):
                            continue
                        # Reserve 30s for refresh and 15s for group cleanup.
                        remaining = deadline - time.monotonic() - 60
                        if remaining <= 0:
                            break
                        if (
                            host_available_bytes()
                            < config["reserve_host_bytes"] + 1024**3
                        ):
                            errors[device["id"]] = (
                                "host headroom fell below pilot minimum"
                            )
                            break
                        name = f"{device['id'].replace(':', '-')}-{workload}-w{workers}"
                        sample_path = output / f"{name}.json"
                        argv = [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "sample",
                            "--device",
                            device["id"],
                            "--workload",
                            workload,
                            "--workers",
                            str(workers),
                            "--threads",
                            str(config["threads_per_job"]),
                            "--batches",
                            str(args.batches),
                            "--output",
                            str(sample_path),
                        ]
                        attempt = run_bounded(argv, output / f"{name}.log", remaining)
                        attempt.update(
                            device=device["id"], workload=workload, workers=workers
                        )
                        attempts.append(attempt)
                        if attempt["returncode"] == 0:
                            sample_result = json.loads(sample_path.read_text())
                            if sample_result["loss_finite"]:
                                results.setdefault(workload, {}).setdefault(
                                    device["id"], {}
                                )[str(workers)] = sample_result["measurement"]
                        write_json(output / f"{name}-attempt.json", attempt)
            if leased:
                inventory = bounded_probe(leased, "inventory-after")
        except (TimeoutError, RuntimeError) as error:
            errors["pilot"] = str(error)
            for device in inventory["devices"]:
                device["acquired"] = False
        config["as_of_unix_seconds"] = time.time()
        decision = plan_capacity(config, inventory, results)
        write_json(
            output / "decision.json",
            {
                "config": config,
                "config_file_sha256": digest(args.config),
                "inventory": inventory,
                "measurements": results,
                "decision": decision,
                "attempts": attempts,
                "acquisition_errors": errors,
                "pilot_source_sha256": digest(__file__),
                "pilot_profile": {
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
                    "deep_source_sha256": digest(ROOT / "src/deep.py"),
                },
                "full_run_started": False,
            },
        )
    print(json.dumps(decision, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("pilot")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--batches", type=int, default=4)
    inventory_parser = commands.add_parser("inventory")
    inventory_parser.add_argument("--devices", required=True)
    inventory_parser.add_argument("--output", type=Path, required=True)
    child = commands.add_parser("sample")
    child.add_argument("--device", required=True)
    child.add_argument(
        "--workload",
        choices=["cnn", "efficientnet_head", "efficientnet_full"],
        required=True,
    )
    child.add_argument("--workers", type=int, default=0)
    child.add_argument("--threads", type=int, default=1)
    child.add_argument("--batches", type=int, default=4)
    child.add_argument("--batch-size", type=int, default=64)
    child.add_argument("--image-size", type=int, default=128)
    child.add_argument("--dataset-batches", type=int)
    child.add_argument("--ready-file", type=Path)
    child.add_argument("--start-file", type=Path)
    child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        inventory, errors = probe_inventory(
            json.loads(args.devices), args.output.parent
        )
        write_json(args.output, {"inventory": inventory, "errors": errors})
        return
    if args.batches <= 0:
        parser.error("batches must be positive")
    if args.command == "sample":
        if (
            args.workers < 0
            or args.threads < 1
            or args.batch_size < 1
            or args.image_size < 32
            or (args.dataset_batches is not None and args.dataset_batches < 1)
            or bool(args.ready_file) != bool(args.start_file)
        ):
            parser.error("invalid sample shape or worker/thread count")
        sample(args)
    else:
        pilot(args)


if __name__ == "__main__":

    def interrupted(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    main()
