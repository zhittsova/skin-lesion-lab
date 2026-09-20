"""Linux process-tree CPU and resident-memory telemetry."""

import math
import os
import time
from pathlib import Path


class ProcessTreeSampler:
    """Sample one process and recursive live descendants from Linux ``/proc``.

    CPU deltas are tracked by ``(pid, starttime)`` and use only each process's
    own user and system ticks. Child cumulative fields are deliberately ignored.
    One fully occupied CPU core is reported as 100 percent.
    """

    def __init__(
        self,
        proc_root=Path("/proc"),
        clock=time.monotonic,
        clock_ticks=None,
        page_size=None,
    ):
        if not callable(clock):
            raise ValueError("clock must be callable")
        clock_ticks = os.sysconf("SC_CLK_TCK") if clock_ticks is None else clock_ticks
        page_size = os.sysconf("SC_PAGE_SIZE") if page_size is None else page_size
        for name, value in (("clock_ticks", clock_ticks), ("page_size", page_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.proc_root = Path(proc_root)
        self.clock = clock
        self.clock_ticks = clock_ticks
        self.page_size = page_size
        self._previous = {}
        self._previous_time = None

    def _stat(self, pid):
        try:
            text = (self.proc_root / str(pid) / "stat").read_text()
            prefix, suffix = text.rsplit(")", 1)
            if int(prefix.split(" ", 1)[0]) != pid:
                return None
            fields = suffix.split()
            if len(fields) <= 21:
                return None
            user_ticks = int(fields[11])
            system_ticks = int(fields[12])
            starttime = int(fields[19])
            rss_pages = int(fields[21])
            if min(user_ticks, system_ticks, starttime, rss_pages) < 0:
                return None
            return starttime, user_ticks + system_ticks, rss_pages * self.page_size
        except OSError, ValueError:
            return None

    def _children(self, pid):
        path = self.proc_root / str(pid) / "task" / str(pid) / "children"
        try:
            values = path.read_text().split()
        except OSError:
            return []
        children = []
        for value in values:
            try:
                child = int(value)
            except ValueError:
                continue
            if child > 0:
                children.append(child)
        return children

    def _tree(self, root_pid):
        pending = [root_pid]
        visited = set()
        current = {}
        while pending:
            pid = pending.pop()
            if pid in visited:
                continue
            visited.add(pid)
            stat = self._stat(pid)
            if stat is None:
                continue
            starttime, ticks, rss_bytes = stat
            current[(pid, starttime)] = (ticks, rss_bytes)
            pending.extend(self._children(pid))
        return current

    def sample(self, root_pid):
        """Return JSON-compatible CPU percent, RSS bytes, and sorted live PIDs."""
        if isinstance(root_pid, bool) or not isinstance(root_pid, int) or root_pid <= 0:
            raise ValueError("root_pid must be a positive integer")
        current = self._tree(root_pid)
        now = self.clock()
        if not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError("clock must return a finite number")
        cpu_percent = None
        if self._previous_time is not None:
            elapsed = now - self._previous_time
            if elapsed <= 0:
                raise ValueError("monotonic clock did not advance")
            delta_ticks = 0
            for identity, (ticks, _) in current.items():
                previous = self._previous.get(identity)
                if previous is not None and ticks >= previous:
                    delta_ticks += ticks - previous
            cpu_percent = delta_ticks / self.clock_ticks / elapsed * 100
        self._previous = {identity: ticks for identity, (ticks, _) in current.items()}
        self._previous_time = now
        return {
            "tree_cpu_percent": cpu_percent,
            "tree_rss_bytes": sum(rss for _, rss in current.values()),
            "live_pids": sorted(pid for pid, _ in current),
        }
