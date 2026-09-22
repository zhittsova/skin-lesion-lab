"""Bounded process-group coordination for concurrent resource pilots."""

import contextlib
import json
import math
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from src.resource_runtime import stop_owned_group

_POLL_SECONDS = 0.1
_START_DELAY_SECONDS = 0.5
_STOP_GRACE_SECONDS = 0.2
_CLEANUP_RESERVE_SECONDS = 0.7


def _positive_seconds(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")


def _nonnegative_seconds(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")


def _argument(argv, flag):
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"each command requires exactly one {flag}")
    return Path(argv[positions[0] + 1])


def _validate(
    commands,
    output,
    timeout_seconds,
    ready_timeout_seconds,
    monitor,
    monitor_timeout_seconds,
):
    _positive_seconds(timeout_seconds, "timeout_seconds")
    _positive_seconds(ready_timeout_seconds, "ready_timeout_seconds")
    _nonnegative_seconds(monitor_timeout_seconds, "monitor_timeout_seconds")
    if timeout_seconds <= _CLEANUP_RESERVE_SECONDS + monitor_timeout_seconds:
        raise ValueError("timeout_seconds leaves no monitor and cleanup reserve")
    if monitor is not None and not callable(monitor):
        raise ValueError("monitor must be callable or None")
    output = Path(output).resolve()
    if not output.is_dir():
        raise ValueError("output must be a precreated directory")
    if not isinstance(commands, (list, tuple)) or not commands:
        raise ValueError("commands must be a nonempty sequence")
    expected_start = output / "start"
    prepared = []
    ready_paths = set()
    for argv in commands:
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(value, str) or not value for value in argv)
        ):
            raise ValueError("each command must be a nonempty argv string list")
        ready = _argument(argv, "--ready-file").resolve()
        start = _argument(argv, "--start-file").resolve()
        if (
            ready.parent != output
            or not ready.name.startswith("job-")
            or not ready.name.endswith(".ready")
        ):
            raise ValueError("ready files must be output/job-N.ready")
        if start != expected_start:
            raise ValueError("all commands must use output/start")
        if ready in ready_paths:
            raise ValueError("ready files must be unique")
        ready_paths.add(ready)
        prepared.append((list(argv), ready, ready.with_suffix(".log")))
    artifacts = [expected_start, output / "state.json"]
    artifacts.extend(path for _, ready, log in prepared for path in (ready, log))
    existing = [path for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError(str(existing[0]))
    return output, expected_start, prepared


def _write_atomic(path, payload, *, exclusive=False):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
            os.unlink(temporary)
        else:
            os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _children(processes, prepared):
    return [
        {
            "index": index,
            "pid": process.pid,
            "returncode": process.poll(),
            "ready_file": str(prepared[index][1]),
            "log_file": str(prepared[index][2]),
        }
        for index, process in enumerate(processes)
    ]


def _stop_all(processes):
    def stop(process):
        try:
            stop_owned_group(process, _STOP_GRACE_SECONDS)
        except PermissionError:
            # A platform may deny group signaling after the leader has crossed
            # an exit boundary. Reap it, or at least stop the direct child.
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    process.wait(timeout=_STOP_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    process.wait()

    threads = [
        threading.Thread(
            target=stop,
            args=(process,),
            daemon=True,
        )
        for process in processes
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def run_group(
    commands,
    output: Path,
    timeout_seconds,
    ready_timeout_seconds,
    monitor=None,
    monitor_timeout_seconds=0,
):
    """Run commands behind one barrier; monitor_timeout_seconds reserves budget."""
    output, start_path, prepared = _validate(
        commands,
        output,
        timeout_seconds,
        ready_timeout_seconds,
        monitor,
        monitor_timeout_seconds,
    )
    began = time.monotonic()
    operation_deadline = (
        began + timeout_seconds - _CLEANUP_RESERVE_SECONDS - monitor_timeout_seconds
    )
    ready_deadline = min(began + ready_timeout_seconds, operation_deadline)
    processes = []
    release_unix = None
    start_unix = None
    status, reason, phase = "timed_out", "total_timeout", "launching"

    def state(current_status="running", current_reason=None):
        _write_atomic(
            output / "state.json",
            {
                "status": current_status,
                "phase": phase,
                "reason": current_reason,
                "elapsed_seconds": time.monotonic() - began,
                "release_unix_seconds": release_unix,
                "start_unix_seconds": start_unix,
                "children": _children(processes, prepared),
            },
        )

    try:
        with contextlib.ExitStack() as stack:
            logs = [stack.enter_context(log.open("x")) for _, _, log in prepared]
            for (argv, _, _), log in zip(prepared, logs, strict=True):
                processes.append(
                    subprocess.Popen(
                        argv,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
            phase = "waiting_ready"
            state()
            while True:
                codes = [process.poll() for process in processes]
                if any(code is not None for code in codes):
                    status = "failed"
                    reason = (
                        "child_nonzero_exit"
                        if any(code not in (None, 0) for code in codes)
                        else "child_exited_before_ready"
                    )
                    break
                now = time.monotonic()
                if now >= operation_deadline:
                    break
                if now >= ready_deadline:
                    status, reason = "blocked", "ready_timeout"
                    break
                monitor_reason = monitor() if monitor is not None else None
                if monitor_reason is not None:
                    if not isinstance(monitor_reason, str) or not monitor_reason:
                        raise TypeError(
                            "monitor reason must be a nonempty string or None"
                        )
                    status, reason = "blocked", monitor_reason
                    break
                now = time.monotonic()
                codes = [process.poll() for process in processes]
                if any(code is not None for code in codes):
                    status = "failed"
                    reason = (
                        "child_nonzero_exit"
                        if any(code not in (None, 0) for code in codes)
                        else "child_exited_before_ready"
                    )
                    break
                if now >= operation_deadline:
                    break
                if now >= ready_deadline:
                    status, reason = "blocked", "ready_timeout"
                    break
                if all(ready.is_file() for _, ready, _ in prepared):
                    start_unix = time.time() + _START_DELAY_SECONDS
                    _write_atomic(
                        start_path,
                        {"start_unix_seconds": start_unix},
                        exclusive=True,
                    )
                    release_unix = time.time()
                    phase = "released"
                    state()
                    break
                time.sleep(min(_POLL_SECONDS, max(0, ready_deadline - now)))

            while phase == "released" and status == "timed_out":
                codes = [process.poll() for process in processes]
                if any(code not in (None, 0) for code in codes):
                    status, reason = "failed", "child_nonzero_exit"
                    break
                if all(code == 0 for code in codes):
                    status, reason = "completed", None
                    break
                if time.monotonic() >= operation_deadline:
                    break
                monitor_reason = monitor() if monitor is not None else None
                now = time.monotonic()
                codes = [process.poll() for process in processes]
                if any(code not in (None, 0) for code in codes):
                    status, reason = "failed", "child_nonzero_exit"
                    break
                if all(code == 0 for code in codes):
                    status, reason = "completed", None
                    break
                if now >= operation_deadline:
                    break
                if monitor_reason is not None:
                    if not isinstance(monitor_reason, str) or not monitor_reason:
                        raise TypeError(
                            "monitor reason must be a nonempty string or None"
                        )
                    status, reason = "blocked", monitor_reason
                    break
                state()
                time.sleep(min(_POLL_SECONDS, max(0, operation_deadline - now)))
        phase = "cleanup"
        state(status, reason)
    except BaseException:
        phase = "cleanup"
        with contextlib.suppress(Exception):
            state("failed", "interrupted")
        raise
    finally:
        _stop_all(processes)

    phase = status
    state(status, reason)
    children = _children(processes, prepared)
    return {
        "status": status,
        "reason": reason,
        "pids": [child["pid"] for child in children],
        "returncodes": [child["returncode"] for child in children],
        "children": children,
        "elapsed_seconds": time.monotonic() - began,
        "release_unix_seconds": release_unix,
        "start_unix_seconds": start_unix,
    }
