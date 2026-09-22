"""Pure fake-/proc tests for recursive process-tree telemetry."""

import tempfile
import unittest
from pathlib import Path

from src.profile_telemetry import ProcessTreeSampler


def write_process(
    proc,
    pid,
    *,
    children=(),
    user=0,
    system=0,
    child_user=0,
    child_system=0,
    start=1,
    rss_pages=1,
    comm="worker (profile)",
):
    tail = ["S", *(["0"] * 21)]
    tail[11] = str(user)
    tail[12] = str(system)
    tail[13] = str(child_user)
    tail[14] = str(child_system)
    tail[19] = str(start)
    tail[21] = str(rss_pages)
    directory = proc / str(pid)
    (directory / "task" / str(pid)).mkdir(parents=True, exist_ok=True)
    (directory / "stat").write_text(f"{pid} ({comm}) " + " ".join(tail))
    (directory / "task" / str(pid) / "children").write_text(
        " ".join(map(str, children))
    )


class Clock:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class ProcessTreeSamplerTests(unittest.TestCase):
    def test_recursive_tree_cpu_and_rss_use_per_pid_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            write_process(
                proc,
                100,
                children=(101, 102),
                user=70,
                system=30,
                child_user=10_000,
                child_system=20_000,
                start=10,
                rss_pages=10,
            )
            write_process(
                proc, 101, children=(103,), user=40, system=10, start=11, rss_pages=4
            )
            write_process(proc, 102, user=20, system=10, start=12, rss_pages=3)
            write_process(proc, 103, user=10, system=10, start=13, rss_pages=2)
            sampler = ProcessTreeSampler(
                proc_root=proc,
                clock=Clock(10, 12),
                clock_ticks=100,
                page_size=4096,
            )
            first = sampler.sample(100)
            self.assertIsNone(first["tree_cpu_percent"])
            self.assertEqual(first["live_pids"], [100, 101, 102, 103])
            self.assertEqual(first["tree_rss_bytes"], 19 * 4096)

            # PID 102 disappears, while PID 104 starts with large lifetime CPU.
            # Per-PID deltas count 50 + 50 + 30 ticks and count neither event as
            # a negative or a historical CPU spike.
            (proc / "102" / "stat").unlink()
            write_process(
                proc,
                100,
                children=(101, 104),
                user=100,
                system=50,
                child_user=99_000,
                child_system=99_000,
                start=10,
                rss_pages=11,
            )
            write_process(
                proc, 101, children=(103,), user=80, system=20, start=11, rss_pages=5
            )
            write_process(proc, 103, user=30, system=20, start=13, rss_pages=3)
            write_process(proc, 104, user=999, system=999, start=14, rss_pages=7)
            second = sampler.sample(100)
            self.assertEqual(second["tree_cpu_percent"], 65.0)
            self.assertEqual(second["live_pids"], [100, 101, 103, 104])
            self.assertEqual(second["tree_rss_bytes"], 26 * 4096)

    def test_pid_reuse_is_a_new_process_not_a_negative_or_historical_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            write_process(proc, 7, user=100, system=50, start=1, rss_pages=2)
            sampler = ProcessTreeSampler(
                proc_root=proc,
                clock=Clock(1, 2),
                clock_ticks=100,
                page_size=1024,
            )
            sampler.sample(7)
            write_process(proc, 7, user=5_000, system=5_000, start=2, rss_pages=3)
            observed = sampler.sample(7)
            self.assertEqual(observed["tree_cpu_percent"], 0.0)
            self.assertEqual(observed["tree_rss_bytes"], 3 * 1024)

    def test_disappeared_and_malformed_pids_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            write_process(proc, 20, children=(21, 22), user=10, start=1)
            write_process(proc, 21, user=5, start=2)
            (proc / "22").mkdir()
            sampler = ProcessTreeSampler(
                proc_root=proc,
                clock=Clock(1, 2),
                clock_ticks=100,
                page_size=4096,
            )
            self.assertEqual(sampler.sample(20)["live_pids"], [20, 21])
            (proc / "20" / "stat").unlink()
            observed = sampler.sample(20)
            self.assertEqual(
                observed,
                {"tree_cpu_percent": 0.0, "tree_rss_bytes": 0, "live_pids": []},
            )

    def test_rejects_invalid_root_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            sampler = ProcessTreeSampler(proc_root=Path(tmp))
            for pid in (0, -1, True, "1"):
                with self.subTest(pid=pid), self.assertRaises(ValueError):
                    sampler.sample(pid)


if __name__ == "__main__":
    unittest.main()
