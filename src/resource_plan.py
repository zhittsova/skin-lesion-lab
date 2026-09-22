"""Pure, deterministic capacity planning for a measured benchmark run.

``validate_config`` accepts this exact JSON-shaped configuration::

    {"version": 1, "preferred_devices": ["mps:0", "cpu"],
     "wall_budget_seconds": 3600, "pilot_budget_seconds": 300,
     "paid_budget": 0, "reserve_host_bytes": 2147483648,
     "reserve_disk_bytes": 5368709120, "reserve_device_bytes": 1073741824,
     "max_concurrent_jobs": 4, "threads_per_job": 2,
     "loader_workers_candidates": [0, 2],
     "measurement_max_age_seconds": 300, "as_of_unix_seconds": 1000,
     "jobs": {"deep": 18}}

Inventory records contain ``measured_at_unix_seconds``, available CPU, host and
disk quantities, and explicit devices with ``id``, ``kind``, ``acquired``,
``free_bytes`` and ``total_bytes``. Measurements are nested as
``workload -> device id -> loader workers -> observation``. Each observation
contains its timestamp, seconds per job, peak host and device bytes, and total
required disk bytes per completed job. The returned decision is JSON-shaped.
"""

import math
from copy import deepcopy
from typing import Any

_CONFIG_KEYS = {
    "version",
    "preferred_devices",
    "wall_budget_seconds",
    "pilot_budget_seconds",
    "paid_budget",
    "reserve_host_bytes",
    "reserve_disk_bytes",
    "reserve_device_bytes",
    "max_concurrent_jobs",
    "threads_per_job",
    "loader_workers_candidates",
    "measurement_max_age_seconds",
    "as_of_unix_seconds",
    "jobs",
}
_INVENTORY_KEYS = {
    "measured_at_unix_seconds",
    "available_cpu_count",
    "available_host_bytes",
    "available_disk_bytes",
    "devices",
}
_DEVICE_KEYS = {"id", "kind", "acquired", "free_bytes", "total_bytes"}
_OBSERVATION_KEYS = {
    "measured_at_unix_seconds",
    "seconds_per_job",
    "peak_host_bytes",
    "peak_device_bytes",
    "required_disk_bytes",
}


def _exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing or unknown:
        raise ValueError(
            f"{label} keys differ: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        qualifier = "positive " if positive else "nonnegative "
        raise ValueError(f"{label} must be a finite {qualifier}number")
    return value


def _device_kind(device_id: str) -> str:
    if not isinstance(device_id, str):
        raise ValueError("device id must be a string")
    if device_id == "cpu":
        return "cpu"
    pieces = device_id.split(":")
    if len(pieces) == 2 and pieces[0] in {"mps", "cuda"} and pieces[1].isdigit():
        return pieces[0]
    raise ValueError(f"invalid device id: {device_id!r}")


def validate_config(config: dict) -> dict:
    """Validate and return a detached, normalized v1 resource configuration."""
    _exact_keys(config, _CONFIG_KEYS, "config")
    if isinstance(config["version"], bool) or config["version"] != 1:
        raise ValueError("only resource config version 1 is supported")
    devices = config["preferred_devices"]
    if (
        not isinstance(devices, list)
        or not devices
        or any(not isinstance(item, str) for item in devices)
        or len(set(devices)) != len(devices)
    ):
        raise ValueError("preferred_devices must be a nonempty unique string list")
    for item in devices:
        _device_kind(item)
    wall = _number(config["wall_budget_seconds"], "wall_budget_seconds", positive=True)
    pilot = _number(
        config["pilot_budget_seconds"], "pilot_budget_seconds", positive=True
    )
    if pilot > wall:
        raise ValueError("pilot budget cannot exceed wall budget")
    paid = _number(config["paid_budget"], "paid_budget")
    if paid != 0:
        raise ValueError("paid compute is unsupported")
    for key in ("reserve_host_bytes", "reserve_disk_bytes", "reserve_device_bytes"):
        _integer(config[key], key)
    _integer(config["max_concurrent_jobs"], "max_concurrent_jobs", minimum=1)
    _integer(config["threads_per_job"], "threads_per_job", minimum=1)
    workers = config["loader_workers_candidates"]
    if (
        not isinstance(workers, list)
        or not workers
        or len(set(workers)) != len(workers)
    ):
        raise ValueError("loader_workers_candidates must be nonempty and unique")
    for worker in workers:
        _integer(worker, "loader worker count")
    _number(
        config["measurement_max_age_seconds"],
        "measurement_max_age_seconds",
        positive=True,
    )
    _number(config["as_of_unix_seconds"], "as_of_unix_seconds")
    jobs = config["jobs"]
    if not isinstance(jobs, dict) or not jobs:
        raise ValueError("jobs must be a nonempty object")
    for workload, count in jobs.items():
        if not isinstance(workload, str) or not workload:
            raise ValueError("workload names must be nonempty strings")
        _integer(count, f"jobs[{workload!r}]", minimum=1)
    return deepcopy(config)


def _validate_inventory(inventory: dict) -> dict[str, dict]:
    _exact_keys(inventory, _INVENTORY_KEYS, "inventory")
    _number(inventory["measured_at_unix_seconds"], "inventory timestamp")
    _integer(inventory["available_cpu_count"], "available_cpu_count")
    _integer(inventory["available_host_bytes"], "available_host_bytes")
    _integer(inventory["available_disk_bytes"], "available_disk_bytes")
    if not isinstance(inventory["devices"], list):
        raise ValueError("devices must be a list")
    indexed = {}
    for item in inventory["devices"]:
        _exact_keys(item, _DEVICE_KEYS, "device")
        device_id = item["id"]
        if not isinstance(device_id, str) or device_id in indexed:
            raise ValueError("device ids must be unique strings")
        if item["kind"] != _device_kind(device_id):
            raise ValueError("device kind does not match its id")
        if not isinstance(item["acquired"], bool):
            raise ValueError("device acquired must be boolean")
        free = _integer(item["free_bytes"], "device free_bytes")
        total = _integer(item["total_bytes"], "device total_bytes")
        if free > total:
            raise ValueError("device free_bytes cannot exceed total_bytes")
        indexed[device_id] = item
    return indexed


def _validate_measurements(measurements: dict) -> dict:
    if not isinstance(measurements, dict):
        raise ValueError("measurements must be an object")
    normalized = {}
    for workload, by_device in measurements.items():
        if (
            not isinstance(workload, str)
            or not workload
            or not isinstance(by_device, dict)
        ):
            raise ValueError("measurements require workload and device objects")
        normalized[workload] = {}
        for device_id, by_worker in by_device.items():
            _device_kind(device_id)
            if not isinstance(by_worker, dict):
                raise ValueError("device measurements must be an object")
            parsed = {}
            for raw_worker, record in by_worker.items():
                if isinstance(raw_worker, int) and not isinstance(raw_worker, bool):
                    worker = raw_worker
                elif isinstance(raw_worker, str) and raw_worker.isdigit():
                    worker = int(raw_worker)
                else:
                    raise ValueError(
                        "measurement worker keys must be nonnegative integers"
                    )
                _integer(worker, "measurement worker count")
                if worker in parsed:
                    raise ValueError("duplicate normalized measurement worker count")
                _exact_keys(record, _OBSERVATION_KEYS, "measurement")
                _number(record["measured_at_unix_seconds"], "measurement timestamp")
                _number(record["seconds_per_job"], "seconds_per_job", positive=True)
                _number(record["peak_host_bytes"], "peak_host_bytes", positive=True)
                _number(record["peak_device_bytes"], "peak_device_bytes")
                _integer(record["required_disk_bytes"], "required_disk_bytes")
                parsed[worker] = record
            normalized[workload][device_id] = parsed
    return normalized


def _fresh(timestamp: float, config: dict) -> bool:
    age = config["as_of_unix_seconds"] - timestamp
    return 0 <= age <= config["measurement_max_age_seconds"]


def _candidate(config, inventory, device, worker, records):
    kind = device["kind"]
    cpu_per_job = config["threads_per_job"] + worker
    if inventory["available_cpu_count"] < cpu_per_job:
        return None, "insufficient_cpu"
    usable_host = inventory["available_host_bytes"] - config["reserve_host_bytes"]
    if usable_host < 0:
        return None, "insufficient_host_memory"
    memory_per_job = max(
        record["peak_host_bytes"]
        + (record["peak_device_bytes"] if kind == "mps" else 0)
        for record in records.values()
    )
    if usable_host < memory_per_job:
        code = (
            "insufficient_shared_mps_memory"
            if kind == "mps"
            else "insufficient_host_memory"
        )
        return None, code
    if kind == "cpu" and any(
        record["peak_device_bytes"] != 0 for record in records.values()
    ):
        return None, "cpu_measurement_has_device_memory"
    if kind == "cuda":
        usable_device = device["free_bytes"] - config["reserve_device_bytes"]
        if usable_device < max(r["peak_device_bytes"] for r in records.values()):
            return None, "insufficient_device_memory"
    if kind == "cpu":
        slots = min(
            config["max_concurrent_jobs"],
            sum(config["jobs"].values()),
            inventory["available_cpu_count"] // cpu_per_job,
            int(usable_host // memory_per_job),
        )
    else:
        slots = min(config["max_concurrent_jobs"], 1)
    if slots < 1:
        return None, "zero_capacity"
    sequential = sum(
        config["jobs"][workload] * record["seconds_per_job"]
        for workload, record in records.items()
    )
    return {"slots": slots, "sequential": sequential, "records": records}, None


def plan_capacity(config: dict, inventory: dict, measurements: dict) -> dict:
    """Return a measured ready/blocked decision without acquiring or running work."""
    config = validate_config(config)
    devices = _validate_inventory(inventory)
    measurements = _validate_measurements(measurements)
    per_device_slots = {device_id: 0 for device_id in config["preferred_devices"]}
    device_reasons = {device_id: [] for device_id in config["preferred_devices"]}
    base = {
        "status": "blocked",
        "reasons": [],
        "selected_device": None,
        "selected_loader_workers": None,
        "attainable_parallelism": 0,
        "per_device_slots": per_device_slots,
        "device_reasons": device_reasons,
        "sequential_makespan_seconds": None,
        "parallel_makespan_seconds": None,
        "parallel_speedup_assumed": False,
        "total_jobs": sum(config["jobs"].values()),
        "workload_plans": {},
    }
    if not _fresh(inventory["measured_at_unix_seconds"], config):
        base["reasons"] = ["inventory_not_fresh"]
        return base

    selected = None
    for device_id in config["preferred_devices"]:
        device = devices.get(device_id)
        if device is None:
            device_reasons[device_id].append("device_missing")
            continue
        if not device["acquired"]:
            device_reasons[device_id].append("device_not_acquired")
            continue
        choices = []
        failed_codes = set()
        for worker in config["loader_workers_candidates"]:
            records = {}
            for workload in config["jobs"]:
                record = measurements.get(workload, {}).get(device_id, {}).get(worker)
                if record is None or not _fresh(
                    record["measured_at_unix_seconds"], config
                ):
                    break
                records[workload] = record
            if len(records) != len(config["jobs"]):
                continue
            choice, failure = _candidate(config, inventory, device, worker, records)
            if choice is not None:
                choices.append((choice["sequential"], worker, choice))
            else:
                failed_codes.add(failure)
        if not choices:
            device_reasons[device_id].extend(
                sorted(failed_codes) if failed_codes else ["missing_fresh_measurement"]
            )
            continue
        _, worker, choice = min(choices, key=lambda item: (item[0], item[1]))
        per_device_slots[device_id] = choice["slots"]
        if selected is None:
            selected = (device_id, worker, choice)

    if selected is None:
        base["reasons"] = ["no_usable_device"]
        return base
    device_id, worker, choice = selected
    sequential = choice["sequential"]
    total_disk = sum(
        config["jobs"][workload] * record["required_disk_bytes"]
        for workload, record in choice["records"].items()
    )
    usable_disk = inventory["available_disk_bytes"] - config["reserve_disk_bytes"]
    reasons = []
    if total_disk > usable_disk:
        reasons.append("insufficient_disk")
    if sequential > config["wall_budget_seconds"]:
        reasons.append("wall_budget_exceeded")
    workload_plans = {
        workload: {
            "jobs": config["jobs"][workload],
            "seconds_per_job": record["seconds_per_job"],
            "peak_host_bytes": record["peak_host_bytes"],
            "peak_device_bytes": record["peak_device_bytes"],
            "required_disk_bytes_per_job": record["required_disk_bytes"],
        }
        for workload, record in choice["records"].items()
    }
    base.update(
        {
            "status": "blocked" if reasons else "ready",
            "reasons": reasons,
            "selected_device": device_id,
            "selected_loader_workers": worker,
            "attainable_parallelism": choice["slots"],
            "sequential_makespan_seconds": sequential,
            # No concurrent timing was measured, so v1 makes no speedup claim.
            "parallel_makespan_seconds": sequential,
            "workload_plans": workload_plans,
        }
    )
    return base
