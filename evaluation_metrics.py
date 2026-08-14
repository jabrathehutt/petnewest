from __future__ import annotations

import os
import platform
import resource
import threading
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


@dataclass
class ResourceSnapshot:
    wall_seconds: float
    rss_start_bytes: int
    rss_end_bytes: int
    peak_rss_bytes: int
    peak_rss_delta_bytes: int
    peak_python_bytes: int


class ResourceMonitor:
    """Measure wall time, process RSS, and peak Python allocations."""

    def __init__(self, sample_interval: float = 0.01) -> None:
        self.sample_interval = sample_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0
        self._peak_rss = 0
        self._rss_start = 0
        self.result: Optional[ResourceSnapshot] = None

    @staticmethod
    def current_rss() -> int:
        if psutil is not None:
            try:
                return int(psutil.Process(os.getpid()).memory_info().rss)
            except psutil.Error:
                pass
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(raw if platform.system() == "Darwin" else raw * 1024)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self._peak_rss = max(self._peak_rss, self.current_rss())
            self._stop.wait(self.sample_interval)

    def __enter__(self) -> "ResourceMonitor":
        self._stop.clear()
        self._start_time = time.perf_counter()
        self._rss_start = self.current_rss()
        self._peak_rss = self._rss_start
        tracemalloc.start()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        wall = time.perf_counter() - self._start_time
        _, peak_python = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        rss_end = self.current_rss()
        self._peak_rss = max(self._peak_rss, rss_end)
        self.result = ResourceSnapshot(
            wall_seconds=float(wall),
            rss_start_bytes=int(self._rss_start),
            rss_end_bytes=int(rss_end),
            peak_rss_bytes=int(self._peak_rss),
            peak_rss_delta_bytes=max(0, int(self._peak_rss) - int(self._rss_start)),
            peak_python_bytes=int(peak_python),
        )


def dataframe_to_csv(rows: Iterable[Dict[str, Any]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def summarize_numeric(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [column for column in numeric if column not in group_columns]
    if not numeric:
        return frame[group_columns].drop_duplicates().reset_index(drop=True)
    grouped = frame.groupby(group_columns, dropna=False)[numeric]
    means = grouped.mean().add_suffix("_mean")
    stds = grouped.std(ddof=0).add_suffix("_std")
    return means.join(stds).reset_index()
