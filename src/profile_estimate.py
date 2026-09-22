"""Deterministic workload estimates from measured per-pass timing profiles.

``estimate_job(profile, counts, ...)`` requires four profile entries: ``train``,
``selection_loss``, ``selection_probability`` and ``mc``. Each entry contains
``steady_seconds_per_batch`` and ``startup_seconds_per_pass``. Counts contain
``train``, ``selection``, ``calibration`` and ``development`` image totals.

The accounting mirrors ``train_deep_pipeline``: every candidate epoch has one
training pass plus separate selection-loss and selection-probability passes.
After the winning checkpoint is loaded, calibration and development each have
``mc_samples`` stochastic passes. There is no train prediction or additional
deterministic calibration/development inference pass.

``pack_jobs`` considers complete jobs in supplied order with concurrency one.
It stops at the first job that does not fit; it never assumes parallel speedup.
"""

import math
from copy import deepcopy

_PROFILE_PHASES = {
    "train",
    "selection_loss",
    "selection_probability",
    "mc",
}
_RATE_KEYS = {"steady_seconds_per_batch", "startup_seconds_per_pass"}
_COUNT_KEYS = {"train", "selection", "calibration", "development"}
_EXCLUDES = [
    "dataset and split loading, validation, and image discovery",
    "model and optimizer construction plus pretrained-weight acquisition",
    "checkpoint serialization, reload, copy, and hashing",
    "policy fitting, metrics, arrays, tables, plots, and report I/O",
    "unmeasured failures, retries, thermal throttling, and external contention",
]


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be finite and {bound}")
    return value


def _positive_integer(value, label, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _validate_profile(profile):
    _exact(profile, _PROFILE_PHASES, "profile")
    for name, rate in profile.items():
        _exact(rate, _RATE_KEYS, f"profile[{name!r}]")
        _number(
            rate["steady_seconds_per_batch"],
            f"profile[{name!r}].steady_seconds_per_batch",
            positive=True,
        )
        _number(
            rate["startup_seconds_per_pass"],
            f"profile[{name!r}].startup_seconds_per_pass",
        )


def _phase(rate, batches, passes):
    steady = passes * batches * rate["steady_seconds_per_batch"]
    startup = passes * rate["startup_seconds_per_pass"]
    return {
        "batches_per_pass": batches,
        "passes": passes,
        "steady_seconds": steady,
        "startup_seconds": startup,
        "total_seconds": steady + startup,
    }


def estimate_job(
    profile,
    counts,
    batch_size=64,
    epochs=20,
    candidates=2,
    mc_samples=30,
):
    """Estimate one complete deep job from measured batch and startup rates."""
    _validate_profile(profile)
    _exact(counts, _COUNT_KEYS, "counts")
    for name, count in counts.items():
        _positive_integer(count, f"counts[{name!r}]")
    for name, value in (
        ("batch_size", batch_size),
        ("epochs", epochs),
        ("candidates", candidates),
    ):
        _positive_integer(value, name)
    _positive_integer(mc_samples, "mc_samples", minimum=2)

    batches = {name: math.ceil(count / batch_size) for name, count in counts.items()}
    candidate_epoch_passes = candidates * epochs
    breakdown = {
        "train": _phase(profile["train"], batches["train"], candidate_epoch_passes),
        "selection_loss": _phase(
            profile["selection_loss"],
            batches["selection"],
            candidate_epoch_passes,
        ),
        "selection_probability": _phase(
            profile["selection_probability"],
            batches["selection"],
            candidate_epoch_passes,
        ),
        "mc_calibration": _phase(profile["mc"], batches["calibration"], mc_samples),
        "mc_development": _phase(profile["mc"], batches["development"], mc_samples),
    }
    return {
        "total_seconds": sum(item["total_seconds"] for item in breakdown.values()),
        "breakdown": breakdown,
        "settings": {
            "counts": deepcopy(counts),
            "batch_size": batch_size,
            "epochs": epochs,
            "candidates": candidates,
            "mc_samples": mc_samples,
        },
        "concurrency_speedup_assumed": False,
        "excludes": list(_EXCLUDES),
    }


def pack_jobs(
    job_estimates,
    available_seconds,
    reserve_seconds,
    concurrency=1,
):
    """Select the next whole sequential jobs that fit after the time reserve."""
    if isinstance(concurrency, bool) or concurrency != 1:
        raise ValueError("only measured sequential concurrency=1 is supported")
    available = _number(available_seconds, "available_seconds")
    reserve = _number(reserve_seconds, "reserve_seconds")
    if not isinstance(job_estimates, list):
        raise ValueError("job_estimates must be a list")
    jobs = []
    ids = set()
    for job in job_estimates:
        _exact(job, {"id", "seconds"}, "job estimate")
        if not isinstance(job["id"], str) or not job["id"] or job["id"] in ids:
            raise ValueError("job ids must be nonempty unique strings")
        _number(job["seconds"], "job seconds", positive=True)
        ids.add(job["id"])
        jobs.append(dict(job))

    usable = max(0, available - reserve)
    selected = []
    used = 0
    for index, job in enumerate(jobs):
        if used + job["seconds"] > usable:
            deferred = jobs[index:]
            reason = (
                "reserve_exceeds_available"
                if reserve > available
                else "next_job_does_not_fit"
            )
            break
        selected.append(job)
        used += job["seconds"]
    else:
        deferred, reason = [], None
    return {
        "concurrency": 1,
        "available_seconds": available,
        "reserve_seconds": reserve,
        "usable_seconds": usable,
        "used_seconds": used,
        "remaining_seconds": usable - used,
        "selected": selected,
        "deferred": deferred,
        "reason": reason,
        "concurrency_speedup_assumed": False,
    }
