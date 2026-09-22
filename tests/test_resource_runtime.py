"""Resource acquisition and subprocess boundaries use real OS primitives."""

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from src.resource_runtime import device_lease, estimate_job_seconds, run_bounded


class ResourceRuntimeTests(unittest.TestCase):
    def test_successful_leader_cannot_leave_a_running_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            heartbeat = Path(tmp) / "child.txt"
            child = "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]);\nwhile True:\n with p.open('a') as f: f.write('.')\n time.sleep(.02)"
            code = "import subprocess,sys,pathlib,time; p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]);\nwhile not pathlib.Path(sys.argv[2]).exists(): time.sleep(.01)"
            result = run_bounded(
                [sys.executable, "-c", code, child, str(heartbeat)],
                Path(tmp) / "log",
                5,
            )

            def cleanup():
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(result["pid"], signal.SIGKILL)

            self.addCleanup(cleanup)
            time.sleep(0.1)
            before = heartbeat.stat().st_size
            time.sleep(0.15)
            self.assertEqual(before, heartbeat.stat().st_size)

    def test_lease_excludes_another_process_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.lock"
            code = (
                "from src.resource_runtime import device_lease; import sys; "
                "\nwith device_lease(sys.argv[1]): print('acquired')"
            )
            with device_lease(path):
                result = subprocess.run(
                    [sys.executable, "-c", code, str(path)], capture_output=True
                )
                self.assertNotEqual(result.returncode, 0)
            with device_lease(path):
                pass

    def test_timeout_stops_process_group_and_preserves_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run.log"
            result = run_bounded(
                [
                    sys.executable,
                    "-c",
                    "import time; print('started', flush=True); time.sleep(30)",
                ],
                log,
                timeout_seconds=3,
                grace_seconds=0.1,
            )
            self.assertTrue(result["timed_out"])
            self.assertIsNotNone(result["returncode"])
            with self.assertRaises(ProcessLookupError):
                os.kill(result["pid"], 0)
            self.assertIn("started", log.read_text())

    def test_existing_log_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run.log"
            log.write_text("old")
            with self.assertRaises(FileExistsError):
                run_bounded([sys.executable, "-c", "pass"], log, 1)
            self.assertEqual(log.read_text(), "old")

    def test_estimate_includes_two_selection_passes_and_mc(self):
        # Two candidates, three epochs: train 3 batches, select 2 batches twice.
        # Final inference: calibration 1 + development 2 batches, four passes.
        seconds = estimate_job_seconds(
            train_batch_seconds=2,
            eval_batch_seconds=1,
            counts={"train": 5, "selection": 3, "calibration": 2, "development": 4},
            batch_size=2,
            epochs=3,
            candidates=2,
            mc_samples=4,
        )
        self.assertEqual(seconds, 72)


if __name__ == "__main__":
    unittest.main()
