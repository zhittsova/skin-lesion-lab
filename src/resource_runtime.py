"""Current-host resource probes and bounded, exclusive pilot execution."""

import contextlib
import fcntl
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path


@contextlib.contextmanager
def device_lease(path):
    """Exclude cooperating runners; this does not reserve hardware from other apps."""
    with Path(path).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def stop_owned_group(process, grace_seconds=5):
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        # Also stop descendants if the group leader exited before its workers.
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            continue
        if sig == signal.SIGKILL:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return


def run_bounded(argv, log_path, timeout_seconds, *, grace_seconds=5, env=None):
    """Launch without a shell; preserve logs and reap the owned process group."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    started = time.monotonic()
    with Path(log_path).open("x") as output:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_owned_group(process, grace_seconds)
        except BaseException:
            stop_owned_group(process, grace_seconds)
            raise
        finally:
            # A leader may exit successfully while workers still hold the GPU.
            stop_owned_group(process, grace_seconds)
    return {
        "pid": process.pid,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
    }


def estimate_job_seconds(
    *,
    train_batch_seconds,
    eval_batch_seconds,
    counts,
    batch_size=64,
    epochs=20,
    candidates=2,
    mc_samples=30,
):
    """Extrapolate observed batch times; excludes startup, plots and hashing."""
    for value in (train_batch_seconds, eval_batch_seconds):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("batch times must be finite and positive")
    for value in (batch_size, epochs, candidates, mc_samples, *counts.values()):
        if type(value) is not int or value <= 0:
            raise ValueError("counts must be positive integers")
    batches = {
        key: math.ceil(counts[key] / batch_size)
        for key in ("train", "selection", "calibration", "development")
    }
    return (
        candidates
        * epochs
        * (
            batches["train"] * train_batch_seconds
            + 2 * batches["selection"] * eval_batch_seconds
        )
        + mc_samples
        * (batches["calibration"] + batches["development"])
        * eval_batch_seconds
    )


def host_available_bytes():
    if platform.system() == "Linux":
        text = Path("/proc/meminfo").read_text()
        match = re.search(r"^MemAvailable:\s+(\d+) kB", text, re.M)
        if match:
            return int(match[1]) * 1024
    elif platform.system() == "Darwin":
        text = subprocess.check_output(["vm_stat"], text=True)
        size = int(re.search(r"page size of (\d+) bytes", text)[1])
        pages = dict(re.findall(r"^(Pages [^:]+):\s+(\d+)", text, re.M))
        # Conservative: do not assume inactive or compressed memory is reclaimable.
        return size * (int(pages["Pages free"]) + int(pages["Pages speculative"]))
    raise RuntimeError("cannot measure available host memory")


def probe_inventory(preferred_devices, output_dir):
    """Attempt a small allocation and synchronized operation on every requested device."""
    import torch

    devices = []
    errors = {}
    for device_id in preferred_devices:
        kind = device_id.split(":")[0]
        entry = {
            "id": device_id,
            "kind": kind,
            "acquired": False,
            "free_bytes": 0,
            "total_bytes": 0,
        }
        try:
            device = torch.device(device_id)
            if kind == "cuda":
                index = device.index or 0
                free, total = torch.cuda.mem_get_info(index)
                entry.update(free_bytes=free, total_bytes=total)
            elif kind == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS unavailable")
            value = torch.ones(8, device=device).square().sum().item()
            if value != 8:
                raise RuntimeError("device computation failed")
            entry["acquired"] = True
        except (RuntimeError, ValueError, AssertionError) as error:
            errors[device_id] = str(error)
        devices.append(entry)
    count = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )
    inventory = {
        "measured_at_unix_seconds": time.time(),
        "available_cpu_count": count or 1,
        "available_host_bytes": host_available_bytes(),
        "available_disk_bytes": shutil.disk_usage(output_dir).free,
        "devices": devices,
    }
    return inventory, errors
