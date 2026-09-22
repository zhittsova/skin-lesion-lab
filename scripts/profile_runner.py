"""Bounded phase profiling of the exact deep helpers and input-cache candidates."""

import argparse
import copy
import hashlib
import json
import math
import os
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.concurrency_pilot import save, telemetry  # noqa: E402
from src.concurrency_runtime import run_group  # noqa: E402
from src.profile_estimate import estimate_job  # noqa: E402
from src.profile_telemetry import ProcessTreeSampler  # noqa: E402
from src.resource_runtime import device_lease, run_bounded  # noqa: E402

PHASES = ("train", "selection_loss", "selection_probability", "mc")
COUNTS = {"train": 5491, "selection": 1366, "calibration": 922, "development": 1395}
VARIANTS = {
    "jpeg_w0": ("none", 0, False),
    "jpeg_w1_persistent": ("none", 1, True),
    "jpeg_w2_persistent": ("none", 2, True),
    "decoded_w0": ("decoded", 0, False),
    "resized_w0": ("resized", 0, False),
}


def fit_phase_rates(rows):
    rates, quality = {}, {}
    for phase in PHASES:
        by_length, by_repeat = {}, {}
        for row in rows:
            if row["phase"] != phase:
                continue
            seconds = row["seconds_per_pass"]
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("invalid timing")
            by_length.setdefault(row["batches"], []).append(seconds)
            pair = by_repeat.setdefault(row["repeat"], {})
            if row["batches"] in pair:
                raise ValueError("duplicate timing")
            pair[row["batches"]] = seconds
        if len(by_length) != 2 or any(len(v) < 2 for v in by_length.values()):
            raise ValueError("each phase needs two lengths and two repetitions")
        short, long = sorted(by_length)
        if any(set(v) != {short, long} for v in by_repeat.values()):
            raise ValueError("incomplete paired repeats")
        slopes = [(v[long] - v[short]) / (long - short) for v in by_repeat.values()]
        median_slope = statistics.median(slopes)
        if min(slopes) <= 0 or (max(slopes) - min(slopes)) / median_slope > 0.35:
            raise ValueError("unstable per-repeat phase slopes: " + phase)
        a, b = (statistics.median(by_length[n]) for n in (short, long))
        slope = (b - a) / (long - short)
        if not math.isfinite(slope) or slope <= 0:
            raise ValueError("phase fit has no positive measured slope: " + phase)
        raw_intercept = a - short * slope
        if raw_intercept < -0.15 * a:
            raise ValueError("unstable negative phase intercept: " + phase)
        rates[phase] = {
            "steady_seconds_per_batch": slope,
            "startup_seconds_per_pass": max(0.0, raw_intercept),
        }
        quality[phase] = {
            "per_repeat_slopes": slopes,
            "max_relative_slope_spread": 0.35,
            "raw_intercept_seconds": raw_intercept,
            "intercept_clamped": raw_intercept < 0,
            "repeat_spread_seconds": max(max(v) - min(v) for v in by_length.values()),
            "median_seconds_by_batches": {
                str(k): statistics.median(v) for k, v in by_length.items()
            },
        }
    return {"rates": rates, "quality": quality}


def cache_projection(*, cache_bytes, unique_contents, full_images, processes, workers):
    if unique_contents <= 0 or full_images < 1 or processes < 1 or workers < 0:
        raise ValueError("invalid cache projection counts")
    return {
        "pixel_bytes": math.ceil(
            cache_bytes / unique_contents * full_images * processes * (1 + workers)
        ),
        "includes_object_or_model_overhead": False,
        "worker_copy_assumption": "one parent plus a copy per worker in each process",
    }


class PassSampler:
    def __init__(self, count, shuffle):
        self.count, self.shuffle = count, shuffle

    def __iter__(self):
        import torch

        if self.shuffle:
            return iter(
                torch.randperm(
                    self.count, generator=torch.Generator().manual_seed(17)
                ).tolist()
            )
        return iter(range(self.count))

    def __len__(self):
        return self.count


class InstrumentedLoader:
    def __init__(self, loader):
        self.loader, self.observations = loader, []

    def __iter__(self):
        import torch

        began = time.perf_counter()
        iterator = iter(self.loader)
        self.iterator_startup_seconds = time.perf_counter() - began
        while True:
            with torch.profiler.record_function("profile.loader_wait"):
                start = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
                waited = time.perf_counter() - start
            detail = {k: float(v.sum()) for k, v in batch[4].items()}
            self.observations.append({"loader_wait_seconds": waited, **detail})
            yield batch[:4]


def warm_persistent_loaders(loaders, samplers, batch_size):
    """Observe cold versus reused worker startup once, outside fitted passes."""
    observations = []
    for loader, sampler in zip(loaders, samplers):
        sampler.count = 2 * batch_size
        first_batch = []
        for _ in range(2):
            began = time.perf_counter()
            iterator = iter(loader)
            next(iterator)
            first_batch.append(time.perf_counter() - began)
            for _ in iterator:
                pass
        observations.append(
            {
                "cold_first_batch_seconds": first_batch[0],
                "warm_first_batch_seconds": first_batch[1],
                "extra_startup_seconds": max(0, first_batch[0] - first_batch[1]),
            }
        )
    return observations


def prepare_inputs(output, real_data=None, count=1536):
    import numpy as np
    from PIL import Image

    began = time.monotonic()
    images = output / "profile-images"
    images.mkdir()
    if real_data:
        manifest = json.loads((real_data / "manifest.json").read_text())
        if (
            manifest.get("role") != "train"
            or manifest.get("purpose") != "performance_only"
        ):
            raise ValueError(
                "real-data pack must contain performance-only training rows"
            )
        originals = manifest["records"]
        for row in originals:
            path = real_data / row["filename"]
            if path.parent.resolve() != real_data.resolve() or path.is_symlink():
                raise ValueError("image filename escapes data directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("real-data image hash mismatch")
    else:
        raw = output / "generated-source"
        raw.mkdir()
        rng = np.random.default_rng(17)
        originals = []
        for i in range(256):
            p = raw / f"source-{i}.jpg"
            Image.fromarray(rng.integers(0, 256, (450, 600, 3), dtype=np.uint8)).save(
                p, quality=85
            )
            originals.append({"filename": p.name, "label": i % 2})
        real_data = raw
    if not 1 <= len(originals) <= 1024 or not 1 <= count <= 4096:
        raise ValueError("input pack/count outside bounded protocol")
    records, raw_pixels, sizes = [], [], []
    for row in originals:
        p = real_data / row["filename"]
        with Image.open(p) as im:
            raw_pixels.append(im.width * im.height * 3)
        sizes.append(p.stat().st_size)
    for i in range(count):
        row = originals[i % len(originals)]
        name = f"profile-{i}.jpg"
        os.link(real_data / row["filename"], images / name)
        records.append(
            {"image_id": f"profile-{i}", "filename": name, "label": row["label"]}
        )
    data = {
        "records": records,
        "source_kind": "real_train_subset"
        if (real_data / "manifest.json").exists()
        else "generated",
        "source_records": len(originals),
        "full_counts": COUNTS,
        "source_pixel_bytes_mean": statistics.mean(raw_pixels),
        "source_file_bytes_mean": statistics.mean(sizes),
        "preparation_seconds": time.monotonic() - began,
        "manifest_sha256": hashlib.sha256(
            (real_data / "manifest.json").read_bytes()
        ).hexdigest()
        if (real_data / "manifest.json").exists()
        else None,
    }
    save(output / "inputs.json", data)
    return data


def child(args):
    import numpy as np
    import torch
    from src import deep
    from src.profile_data import ProfileDataset
    from torch.utils.data import DataLoader

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    deep.set_seed(17)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    config = json.loads(args.config.read_text())
    payload = json.loads(args.inputs.read_text())
    records = payload["records"]
    mode, workers, persistent = VARIANTS[args.variant]
    instrument = args.diagnostic
    datasets = [
        ProfileDataset(
            records,
            args.images,
            config["image_size"],
            train,
            cache_mode=mode,
            instrument=instrument,
        )
        for train in (True, False)
    ]
    samplers = [
        PassSampler(config["lengths"][-1] * config["batch_size"], train)
        for train in (True, False)
    ]
    loaders = [
        DataLoader(
            ds,
            batch_size=config["batch_size"],
            sampler=sampler,
            num_workers=workers,
            persistent_workers=persistent,
            pin_memory=device.type == "cuda",
            worker_init_fn=deep.seed_worker,
            generator=torch.Generator().manual_seed(17 + i),
        )
        for i, (ds, sampler) in enumerate(zip(datasets, samplers))
    ]
    began = time.monotonic()
    model = deep.build_model(
        "small_cnn" if args.workload == "cnn" else "efficientnet_b0",
        dropout=0.3,
        pretrained=args.workload != "cnn",
        freeze_backbone=args.workload != "efficientnet_full",
    ).to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=0.001,
        weight_decay=0.0001,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    model_setup_seconds = time.monotonic() - began
    # Warm kernels and optimizer allocations on a fixed tensor, outside measurements.
    warm = [
        (
            torch.zeros(
                config["batch_size"], 3, config["image_size"], config["image_size"]
            ),
            torch.tensor(
                [i % 2 for i in range(config["batch_size"])], dtype=torch.float32
            ),
            [],
            [],
        )
        for _ in range(2)
    ]
    deep.train_one_epoch(model, warm, criterion, optimizer, device)
    sync()
    model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    def reset():
        model.load_state_dict(model_state)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        deep.set_seed(17)
        sync()

    def invoke(phase, loader):
        if phase == "train":
            value = deep.train_one_epoch(model, loader, criterion, optimizer, device)
        elif phase == "selection_loss":
            value = deep.evaluate_loss(model, loader, criterion, device)
        elif phase == "selection_probability":
            value = deep.predict_probabilities(model, loader, device)
        else:
            value = deep.predict_with_mc_dropout(model, loader, device, 2)
        if isinstance(value, dict):
            field = (
                "probability"
                if phase == "selection_probability"
                else "all_probabilities"
            )
            if not np.isfinite(value[field]).all():
                raise ValueError("non-finite inference")
        elif not math.isfinite(value):
            raise ValueError("non-finite loss")

    worker_startup = (
        warm_persistent_loaders(loaders, samplers, config["batch_size"])
        if persistent
        else []
    )
    save(args.ready_file, {"pid": os.getpid(), "ready": True})
    while not args.start_file.exists():
        time.sleep(0.05)
    start_at = json.loads(args.start_file.read_text())["start_unix_seconds"]
    while time.time() < start_at:
        time.sleep(0.01)
    result = {
        "workload_start_unix_seconds": time.time(),
        "persistent_worker_startup": worker_startup,
        "workload": args.workload,
        "variant": args.variant,
        "source_kind": payload["source_kind"],
        "timings": [],
        "cache_build_seconds": sum(d.cache_build_seconds for d in datasets),
        "cache_pixel_bytes_per_dataset": datasets[0].cache_bytes,
        "model_setup_seconds": model_setup_seconds,
        "torch": torch.__version__,
        "python": sys.version,
        "mode": "diagnostic" if instrument else "throughput",
        "full_run_started": False,
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if not instrument:
        for repeat in range(config["repeats"]):
            lengths = (
                config["lengths"]
                if repeat % 2 == 0
                else list(reversed(config["lengths"]))
            )
            for batches in lengths:
                for sampler in samplers:
                    sampler.count = batches * config["batch_size"]
                for phase in PHASES:
                    reset()
                    start = time.perf_counter()
                    invoke(phase, loaders[0 if phase == "train" else 1])
                    sync()
                    elapsed = time.perf_counter() - start
                    result["timings"].append(
                        {
                            "repeat": repeat,
                            "batches": batches,
                            "phase": phase,
                            "seconds": elapsed,
                            "seconds_per_pass": elapsed / (2 if phase == "mc" else 1),
                        }
                    )
                    save(args.output, result)
        try:
            result["fit"] = fit_phase_rates(result["timings"])
            result["deep_phase_estimate"] = estimate_job(result["fit"]["rates"], COUNTS)
            # Two candidate train/selection loaders and calibration/development loaders.
            overhead = (
                2 * worker_startup[0]["extra_startup_seconds"]
                + 4 * worker_startup[1]["extra_startup_seconds"]
                if worker_startup
                else 0
            )
            result["worker_startup_allowance_seconds"] = overhead
            result["deep_phase_estimate"]["total_seconds"] += overhead
        except ValueError as error:
            result["fit_error"] = str(error)
        # Measure local checkpoint and hash overhead separately from model phases.
        checkpoint = args.output.parent / "checkpoint-probe.pt"
        start = time.perf_counter()
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            checkpoint,
        )
        with checkpoint.open("r+b") as f:
            os.fsync(f.fileno())
        result["checkpoint_write_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        torch.load(checkpoint, map_location=device, weights_only=True)
        sync()
        result["checkpoint_reload_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        result["checkpoint_sha256"] = hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()
        result["checkpoint_hash_seconds"] = time.perf_counter() - start
        result["checkpoint_bytes"] = checkpoint.stat().st_size
        checkpoint.unlink()
    else:
        for sampler in samplers:
            sampler.count = 4 * config["batch_size"]
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        result["diagnostics"] = {}
        for phase in PHASES:
            wrapped = InstrumentedLoader(loaders[0 if phase == "train" else 1])
            detail = {}
            reset()
            try:
                with torch.profiler.profile(
                    activities=activities, profile_memory=True, record_shapes=False
                ) as prof:
                    with torch.profiler.record_function("profile.exact_" + phase):
                        invoke(phase, wrapped)
                        sync()
                prof.export_chrome_trace(
                    str(args.output.parent / (phase + "-trace.json"))
                )
                (args.output.parent / (phase + "-operators.txt")).write_text(
                    prof.key_averages().table(
                        sort_by="self_cpu_time_total", row_limit=50
                    )
                )
                detail["operators"] = [
                    {
                        "name": e.key,
                        "count": e.count,
                        "self_cpu_us": e.self_cpu_time_total,
                        "self_device_us": getattr(e, "self_device_time_total", 0),
                    }
                    for e in prof.key_averages()
                ]
            except RuntimeError as error:
                detail["profiler_error"] = str(error)
            detail["loader_observations"] = wrapped.observations
            detail["iterator_startup_seconds"] = getattr(
                wrapped, "iterator_startup_seconds", None
            )
            resident = []
            for batch in loaders[0 if phase == "train" else 1]:
                resident.append((batch[0].to(device), batch[1], batch[2], batch[3]))
            reset()
            start = time.perf_counter()
            invoke(phase, resident)
            sync()
            detail["resident_input_seconds"] = time.perf_counter() - start
            detail["resident_input_batches"] = len(resident)
            result["diagnostics"][phase] = detail
            save(args.output, result)
    result["workload_end_unix_seconds"] = time.time()
    result["cache_projection_one_job"] = cache_projection(
        cache_bytes=datasets[0].cache_bytes,
        unique_contents=payload["source_records"],
        full_images=sum(COUNTS.values()),
        processes=1,
        workers=workers,
    )
    result["cache_projection_two_jobs"] = cache_projection(
        cache_bytes=datasets[0].cache_bytes,
        unique_contents=payload["source_records"],
        full_images=sum(COUNTS.values()),
        processes=2,
        workers=workers,
    )
    result["gpu_reserved_peak_bytes"] = (
        torch.cuda.max_memory_reserved() if device.type == "cuda" else 0
    )
    result["main_cpu_seconds"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_utime
        + resource.getrusage(resource.RUSAGE_SELF).ru_stime
    )
    result["limitations"] = [
        "Phase fit excludes unmeasured calibration, plots, full-cohort validation and transfer",
        "Repeated subset fits filesystem cache; full dataset throughput remains conditional",
        "Persistent workers change RNG scheduling; model seed/reset and transforms remain fixed",
        "Resident-input and instrumented trace timings are diagnostic only",
    ]
    save(args.output, result)


def cpu_snapshot():
    fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    values = list(map(int, fields))
    return sum(values[:8]), values[3] + values[4]


def required_headroom(config, slots):
    if slots not in (1, 2):
        raise ValueError("only one or two measured slots")
    return {
        "host_free": config["reserve_host_bytes"] + slots * 3 * 1024**3,
        "gpu_free": config["reserve_device_bytes"] + slots * 4 * 1024**3,
        "disk_free": config["reserve_disk_bytes"],
    }


def run(config, output, real_data=None):
    if config != DEFAULT_CONFIG:
        raise ValueError(
            "this bounded profiling protocol requires the published config"
        )
    output.mkdir(parents=True, exist_ok=False)
    save(output / "config.json", config)
    deadline = time.monotonic() + config["pilot_budget_seconds"]
    report = {
        "config": config,
        "status": "running",
        "cases": [],
        "full_run_started": False,
        "full_run_admitted": False,
    }
    key = hashlib.sha256(b"cuda:0").hexdigest()[:16]
    lock = (
        Path(tempfile.gettempdir()) / f"skin-lesion-resource-{os.getuid()}-{key}.lock"
    )
    try:
        with device_lease(lock):
            probe = run_bounded(
                [
                    sys.executable,
                    str(ROOT / "scripts/resource_pilot.py"),
                    "inventory",
                    "--devices",
                    '["cuda:0"]',
                    "--output",
                    str(output / "inventory.json"),
                ],
                output / "inventory.log",
                40,
                grace_seconds=0.5,
            )
            if (
                probe["returncode"] != 0
                or not json.loads((output / "inventory.json").read_text())["inventory"][
                    "devices"
                ][0]["acquired"]
            ):
                raise RuntimeError("CUDA acquisition failed")
            prepare_argv = [
                sys.executable,
                str(Path(__file__).resolve()),
                "prepare",
                "--output",
                str(output),
                "--count",
                str(config["batch_size"] * max(config["lengths"])),
            ]
            if real_data:
                prepare_argv.extend(["--data", str(real_data.resolve())])
            prepared = run_bounded(
                prepare_argv,
                output / "prepare.log",
                min(120, max(1, deadline - time.monotonic() - 20)),
                grace_seconds=0.5,
            )
            report["input_preparation"] = prepared
            if prepared["returncode"] != 0:
                raise RuntimeError("bounded input preparation failed")
            payload = json.loads((output / "inputs.json").read_text())
            report["inputs"] = {k: v for k, v in payload.items() if k != "records"}
            cases = [("efficientnet_full", v, False, 1) for v in VARIANTS]
            winner = None
            index = 0
            while index < len(cases):
                workload, variant, diagnostic, slots = cases[index]
                index += 1
                remaining = deadline - time.monotonic() - 20
                if remaining < 45:
                    raise TimeoutError("profile wall budget exhausted")
                group = output / f"{index:02d}-{workload}-{variant}"
                group.mkdir()
                case = {
                    "workload": workload,
                    "variant": variant,
                    "diagnostic": diagnostic,
                    "slots": slots,
                    "directory": group.name,
                }
                report["cases"].append(case)
                available = telemetry(output)
                needs = required_headroom(config, slots)
                case["admission"] = {"required": needs, "observed": available}
                if any(available[k] < value for k, value in needs.items()):
                    raise RuntimeError("insufficient profile headroom")
                argv = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "sample",
                    "--config",
                    str(output / "config.json"),
                    "--inputs",
                    str(output / "inputs.json"),
                    "--images",
                    str(output / "profile-images"),
                    "--workload",
                    workload,
                    "--variant",
                    variant,
                    "--device",
                    "cuda:0",
                    "--ready-file",
                    str(group / "job-0.ready"),
                    "--start-file",
                    str(group / "start"),
                    "--output",
                    str(group / "sample.json"),
                ]
                if diagnostic:
                    argv.append("--diagnostic")
                commands = []
                for slot in range(slots):
                    child_dir = group / f"child-{slot}"
                    child_dir.mkdir()
                    command = list(argv)
                    command[command.index("--ready-file") + 1] = str(
                        group / f"job-{slot}.ready"
                    )
                    command[command.index("--output") + 1] = str(
                        child_dir / "sample.json"
                    )
                    commands.append(command)
                process_sampler = ProcessTreeSampler()
                last_sample = 0.0
                previous_cpu = None
                snapshots = []

                def monitor():
                    nonlocal last_sample, previous_cpu
                    now = time.monotonic()
                    if now - last_sample < 1:
                        return None
                    last_sample = now
                    sample = telemetry(output)
                    sample.update(process_sampler.sample(os.getpid()))
                    current = cpu_snapshot()
                    if previous_cpu and current[0] > previous_cpu[0]:
                        sample["host_cpu_busy_percent"] = 100 * (
                            1
                            - (current[1] - previous_cpu[1])
                            / (current[0] - previous_cpu[0])
                        )
                    previous_cpu = current
                    snapshots.append(sample)
                    save(group / "telemetry.json", snapshots)
                    for field, reserve in [
                        ("host_free", "reserve_host_bytes"),
                        ("gpu_free", "reserve_device_bytes"),
                        ("disk_free", "reserve_disk_bytes"),
                    ]:
                        if sample[field] < config[reserve]:
                            return field + "_reserve_exhausted"
                    return None

                print(
                    f"Profiling {workload}/{variant}"
                    + (" diagnostic" if diagnostic else ""),
                    flush=True,
                )
                case["execution"] = run_group(
                    commands,
                    group,
                    min(remaining, config["case_budget_seconds"]),
                    min(90, remaining / 2),
                    monitor=monitor,
                    monitor_timeout_seconds=4,
                )
                if case["execution"]["status"] != "completed":
                    case["status"] = "failed"
                    save(output / "decision.json", report)
                    raise RuntimeError("profiling case failed; retained logs")
                result = json.loads((group / "child-0" / "sample.json").read_text())
                case["children"] = [
                    json.loads((group / f"child-{slot}" / "sample.json").read_text())
                    for slot in range(slots)
                ]
                case["status"] = "completed"
                if slots == 1 and not diagnostic and "deep_phase_estimate" in result:
                    case["deep_phase_seconds"] = result["deep_phase_estimate"][
                        "total_seconds"
                    ]
                    case["fit_quality"] = result["fit"]["quality"]
                save(output / "decision.json", report)
                if index == len(VARIANTS):
                    valid = [c for c in report["cases"] if "deep_phase_seconds" in c]
                    winner = (
                        min(valid, key=lambda c: c["deep_phase_seconds"])["variant"]
                        if valid
                        else "jpeg_w0"
                    )
                    report["screening_status"] = (
                        "provisional" if valid else "insufficient_evidence"
                    )
                    report["screening_winner"] = winner if valid else None
                    report["followup_variant"] = winner
                    cases.extend(
                        (model, v, False, 1)
                        for model in ["cnn", "efficientnet_head"]
                        for v in dict.fromkeys(["jpeg_w0", winner])
                    )
                    cases.extend(
                        ("efficientnet_full", v, True, 1)
                        for v in dict.fromkeys(["jpeg_w0", winner])
                    )
                    # Repeat the same winner at one/two slots, reverse order on repeat two.
                    cases.extend(
                        ("efficientnet_full", winner, False, n) for n in [1, 2, 2, 1]
                    )
            report["status"] = "completed"
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        report.update(status="blocked", reason=str(error))
    except BaseException:
        report.update(status="interrupted", reason="controller interrupted")
        raise
    finally:
        estimates = []
        for c in report["cases"]:
            if not c["diagnostic"] and "deep_phase_seconds" in c:
                estimates.append(
                    {
                        "id": c["workload"] + "-" + c["variant"],
                        "seconds": c["deep_phase_seconds"],
                    }
                )
        report["phase_only_estimates"] = estimates
        report["next_action"] = (
            "Review phase fits, cache memory and overhead before real full-job validation; no automatic full run"
        )
        save(output / "decision.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "screening_winner": report.get("screening_winner"),
                "phase_only_estimates": report["phase_only_estimates"],
                "reason": report.get("reason"),
            },
            indent=2,
        )
    )


DEFAULT_CONFIG = {
    "version": 1,
    "paid_budget": 0,
    "pilot_budget_seconds": 2700,
    "case_budget_seconds": 420,
    "lengths": [6, 24],
    "repeats": 2,
    "batch_size": 64,
    "image_size": 128,
    "reserve_host_bytes": 2147483648,
    "reserve_device_bytes": 1073741824,
    "reserve_disk_bytes": 4294967296,
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    a = commands.add_parser("run")
    a.add_argument("--config", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--data", type=Path)
    a = commands.add_parser("prepare")
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--data", type=Path)
    a.add_argument("--count", type=int, required=True)
    a = commands.add_parser("sample")
    for name in ["config", "inputs", "images", "output", "ready-file", "start-file"]:
        a.add_argument("--" + name, type=Path, required=True)
    a.add_argument(
        "--workload",
        choices=["cnn", "efficientnet_head", "efficientnet_full"],
        required=True,
    )
    a.add_argument("--variant", choices=VARIANTS, required=True)
    a.add_argument("--device", default="cuda:0")
    a.add_argument("--diagnostic", action="store_true")
    args = p.parse_args()
    if args.command == "prepare":
        prepare_inputs(args.output, args.data, args.count)
    elif args.command == "sample":
        child(args)
    else:
        config = json.loads(args.config.read_text())
        output = args.output.resolve()

        run(config, output, args.data)


if __name__ == "__main__":
    import signal

    def interrupted(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        raise KeyboardInterrupt("profiling interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    main()
