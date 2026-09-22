"""Real-process tests for the bounded concurrency barrier."""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src import concurrency_runtime
from src.concurrency_runtime import run_group

CHILD = r"""
import argparse, json, os, pathlib, subprocess, sys, time
p = argparse.ArgumentParser()
p.add_argument('--ready-file', type=pathlib.Path, required=True)
p.add_argument('--start-file', type=pathlib.Path, required=True)
p.add_argument('--result-file', type=pathlib.Path)
p.add_argument('--peer-file', type=pathlib.Path)
p.add_argument('--mode', choices=('work', 'fail', 'hang', 'heartbeat'), required=True)
a = p.parse_args()
if a.mode == 'hang':
    time.sleep(30)
a.ready_file.write_text(str(os.getpid()))
while not a.start_file.exists():
    time.sleep(.01)
target = json.loads(a.start_file.read_text())['start_unix_seconds']
while time.time() < target:
    time.sleep(.005)
started = time.time()
if a.mode == 'fail':
    deadline = time.time() + 2
    while a.peer_file and not a.peer_file.exists() and time.time() < deadline:
        time.sleep(.01)
    sys.exit(7 if not a.peer_file or a.peer_file.exists() else 9)
if a.mode == 'heartbeat':
    code = "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]);\nwhile True:\n p.open('a').write('.')\n time.sleep(.02)"
    descendant = subprocess.Popen([sys.executable, '-c', code, str(a.result_file)])
    a.result_file.with_suffix('.pid').write_text(str(descendant.pid))
    while True: time.sleep(1)
time.sleep(.3)
a.result_file.write_text(json.dumps({'start': started, 'end': time.time()}))
"""


def command(output, index, mode="work"):
    argv = [
        sys.executable,
        "-c",
        CHILD,
        "--ready-file",
        str(output / f"job-{index}.ready"),
        "--start-file",
        str(output / "start"),
        "--result-file",
        str(output / f"job-{index}.result"),
        "--mode",
        mode,
    ]
    if mode == "fail":
        argv.extend(["--peer-file", str(output / "job-1.pid")])
    return argv


class ConcurrencyRuntimeTests(unittest.TestCase):
    def test_ready_barrier_releases_children_into_actual_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result = run_group(
                [command(output, 0), command(output, 1)],
                output,
                timeout_seconds=5,
                ready_timeout_seconds=2,
            )
            intervals = [
                json.loads((output / f"job-{index}.result").read_text())
                for index in range(2)
            ]
            self.assertEqual(result["status"], "completed")
            self.assertIsNone(result["reason"])
            self.assertTrue(all(code == 0 for code in result["returncodes"]))
            self.assertLess(
                max(i["start"] for i in intervals), min(i["end"] for i in intervals)
            )
            payload = json.loads((output / "start").read_text())
            self.assertEqual(
                result["start_unix_seconds"], payload["start_unix_seconds"]
            )
            self.assertLess(
                result["release_unix_seconds"], result["start_unix_seconds"]
            )
            state = json.loads((output / "state.json").read_text())
            self.assertEqual(state["status"], "completed")

    def test_nonzero_child_cancels_sleeping_sibling_and_descendants(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result = run_group(
                [command(output, 0, "fail"), command(output, 1, "heartbeat")],
                output,
                timeout_seconds=5,
                ready_timeout_seconds=2,
            )
            heartbeat = output / "job-1.result"
            before = heartbeat.stat().st_size if heartbeat.exists() else 0
            time.sleep(0.15)
            after = heartbeat.stat().st_size if heartbeat.exists() else 0
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["reason"], "child_nonzero_exit")
            self.assertIn(7, result["returncodes"])
            self.assertEqual(before, after)
            self.assertTrue(heartbeat.with_suffix(".pid").exists())
            for pid in result["pids"]:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

    def test_hung_readiness_blocks_without_releasing_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result = run_group(
                [command(output, 0), command(output, 1, "hang")],
                output,
                timeout_seconds=4,
                ready_timeout_seconds=0.4,
            )
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["reason"], "ready_timeout")
            self.assertFalse((output / "start").exists())
            self.assertLess(result["elapsed_seconds"], 3)
            for pid in result["pids"]:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

    def test_monitor_reason_blocks_and_stops_released_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)

            def monitor():
                return "host_pressure" if (output / "start").exists() else None

            result = run_group(
                [command(output, 0, "heartbeat")],
                output,
                timeout_seconds=5,
                ready_timeout_seconds=2,
                monitor=monitor,
            )
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["reason"], "host_pressure")

    def test_total_deadline_includes_cleanup_reserve(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result = run_group(
                [command(output, 0, "heartbeat")],
                output,
                timeout_seconds=1.5,
                ready_timeout_seconds=1,
            )
            self.assertEqual(result["status"], "timed_out")
            self.assertEqual(result["reason"], "total_timeout")
            self.assertLessEqual(result["elapsed_seconds"], 1.5)

    def test_slow_bounded_monitor_cannot_release_past_ready_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)

            def monitor():
                time.sleep(0.3)
                return None

            result = run_group(
                [command(output, 0, "heartbeat")],
                output,
                timeout_seconds=2,
                ready_timeout_seconds=0.1,
                monitor=monitor,
                monitor_timeout_seconds=0.4,
            )
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["reason"], "ready_timeout")
            self.assertFalse((output / "start").exists())

    def test_monitor_exception_still_stops_the_owned_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            pid_path = output / "job-0.ready"

            def monitor():
                if pid_path.exists():
                    raise KeyboardInterrupt
                return None

            with self.assertRaises(KeyboardInterrupt):
                run_group(
                    [command(output, 0, "heartbeat")],
                    output,
                    timeout_seconds=5,
                    ready_timeout_seconds=2,
                    monitor=monitor,
                )
            pid = int(pid_path.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_state_write_failure_still_stops_a_live_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            ready = output / "job-0.ready"
            real_write = concurrency_runtime._write_atomic

            def fail_state(path, payload, **kwargs):
                if path.name == "state.json":
                    deadline = time.monotonic() + 1
                    while not ready.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    raise OSError("injected state persistence failure")
                return real_write(path, payload, **kwargs)

            with (
                patch("src.concurrency_runtime._write_atomic", side_effect=fail_state),
                self.assertRaisesRegex(OSError, "injected state"),
            ):
                run_group(
                    [command(output, 0, "heartbeat")],
                    output,
                    timeout_seconds=5,
                    ready_timeout_seconds=2,
                )
            pid = int(ready.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_existing_log_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            log = output / "job-0.log"
            log.write_text("keep")
            with self.assertRaises(FileExistsError):
                run_group(
                    [command(output, 0)],
                    output,
                    timeout_seconds=5,
                    ready_timeout_seconds=2,
                )
            self.assertEqual(log.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
