from __future__ import annotations

import csv
import os
import time
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TimingRecorder:
    def __init__(self, root: str | os.PathLike[str], role: str, rank: int):
        self.path = Path(root) / "timing" / f"{role}_rank_{rank}.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["time", "role", "rank", "client_id", "round", "phase", "seconds", "extra"]
                )

        self.role = role
        self.rank = rank
        self.client_id = os.environ.get("CLIENT_ID", "")

    @contextmanager
    def measure(self, phase: str, round_id: int | str = "", extra: str = ""):
        _sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            elapsed = time.perf_counter() - start
            with open(self.path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        self.role,
                        self.rank,
                        self.client_id,
                        round_id,
                        phase,
                        f"{elapsed:.6f}",
                        extra,
                    ]
                )

@contextmanager
def measure_torch_collectives(timer: TimingRecorder, round_id: int | str):
    import torch.distributed as dist

    ops = [
        "all_reduce",
        "all_gather",
        "all_gather_into_tensor",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "broadcast",
        "barrier",
    ]

    originals = {}
    totals = defaultdict(float)
    counts = defaultdict(int)

    def wrap(name, fn):
        def wrapped(*args, **kwargs):
            _sync()
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                _sync()
                totals[name] += time.perf_counter() - start
                counts[name] += 1
        return wrapped

    for name in ops:
        if hasattr(dist, name):
            originals[name] = getattr(dist, name)
            setattr(dist, name, wrap(name, originals[name]))

    try:
        yield totals
    finally:
        for name, fn in originals.items():
            setattr(dist, name, fn)

        for name, seconds in totals.items():
            with open(timer.path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        timer.role,
                        timer.rank,
                        timer.client_id,
                        round_id,
                        f"torchtitan_collective_{name}",
                        f"{seconds:.6f}",
                        f"count={counts[name]}",
                    ]
                )

    
