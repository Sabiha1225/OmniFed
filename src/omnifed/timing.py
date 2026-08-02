from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch


TIMING_COLUMNS = [
    "time",
    "role",
    "rank",
    "client_id",
    "round",
    "iteration",
    "global_step",
    "phase",
    "seconds",
    "extra",
]


def _sync_gpu() -> None:
    """Wait for outstanding GPU work before reading the clock."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TimingRecorder:
    def __init__(
        self,
        root: str | os.PathLike[str],
        role: str,
        rank: int,
    ):
        self.role = role
        self.rank = rank
        self.client_id = os.environ.get("CLIENT_ID", "")

        timing_root = Path(root) / "timing"

        if role == "client":
            if self.client_id == "":
                raise ValueError(
                    "CLIENT_ID must be set before creating a "
                    "client TimingRecorder"
                )

            self.path = (
                timing_root
                / f"client_{self.client_id}"
                / f"rank_{rank}.csv"
            )
        else:
            self.path = timing_root / f"{role}_rank_{rank}.csv"

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_file()

    def _initialize_file(self) -> None:
        """Create the CSV file and write its header."""
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, newline="") as file:
                existing_header = next(csv.reader(file), [])

            if existing_header != TIMING_COLUMNS:
                raise RuntimeError(
                    f"Timing file {self.path} uses the old schema.\n"
                    f"Existing columns: {existing_header}\n"
                    f"Expected columns: {TIMING_COLUMNS}\n"
                    "Use a new job directory or rename the old timing file."
                )

            return

        with open(self.path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(TIMING_COLUMNS)

    def record(
        self,
        phase: str,
        seconds: float,
        round_id: int | str = "",
        iteration: int | str = "",
        global_step: int | str = "",
        extra: str = "",
    ) -> None:
        """Write one completed timing measurement."""
        with open(self.path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    self.role,
                    self.rank,
                    self.client_id,
                    round_id,
                    iteration,
                    global_step,
                    phase,
                    f"{seconds:.6f}",
                    extra,
                ]
            )

    @contextmanager
    def measure(
        self,
        phase: str,
        round_id: int | str = "",
        extra: str = "",
        *,
        iteration: int | str = "",
        global_step: int | str = "",
    ):
        """Measure a CPU/GPU operation and record its elapsed wall time."""
        _sync_gpu()
        start = time.perf_counter()

        try:
            yield
        finally:
            _sync_gpu()
            seconds = time.perf_counter() - start

            self.record(
                phase=phase,
                seconds=seconds,
                round_id=round_id,
                iteration=iteration,
                global_step=global_step,
                extra=extra,
            )


def _communication_category(event_name: str) -> str | None:
    """Classify a Torch profiler event as a communication operation."""
    name = event_name.lower()

    if "all_reduce" in name or "allreduce" in name:
        return "all_reduce"

    if "all_gather" in name or "allgather" in name:
        return "all_gather"

    if "reduce_scatter" in name or "reducescatter" in name:
        return "reduce_scatter"

    if "broadcast" in name:
        return "broadcast"

    if (
        "send" in name
        or "recv" in name
        or "sendrecv" in name
        or "send_recv" in name
    ):
        return "send_recv"

    # PyTorch uses NCCL names with ROCm/RCCL in several profiler outputs.
    if "nccl" in name or "rccl" in name:
        return "other_collective"

    return None


def _event_device_time_us(event) -> float:
    """Read device time from different PyTorch profiler versions."""
    for attribute in (
        "self_device_time_total",
        "device_time_total",
        "self_cuda_time_total",
        "cuda_time_total",
    ):
        value = getattr(event, attribute, None)

        if value is not None and float(value) > 0:
            return float(value)

    return 0.0


@contextmanager
def profile_torchtitan_communication(
    timer: TimingRecorder,
    round_id: int,
    iteration: int,
    global_step: int,
    enabled: bool = True,
):
    """
    Profile GPU communication during one TorchTitan training iteration.

    This captures TP, FSDP and PP communication visible to torch.profiler.
    """
    if not enabled:
        yield
        return

    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]

    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    _sync_gpu()

    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        yield

    _sync_gpu()

    totals_us: dict[str, float] = defaultdict(float)
    event_counts: dict[str, int] = defaultdict(int)

    for event in profiler.key_averages():
        category = _communication_category(event.key)

        if category is None:
            continue

        device_time_us = _event_device_time_us(event)

        # Device time is preferred. CPU time is only a fallback when the
        # profiler did not expose a device duration for this event.
        if torch.cuda.is_available():
            # On Frontier/ROCm, report GPU collective time.
            # Ignore CPU-only collective launch events.
            measured_us = device_time_us
        else:
            # CPU fallback for environments without a GPU.
            measured_us = float(
                getattr(event, "self_cpu_time_total", 0.0)
            )

        if measured_us <= 0:
            continue

        totals_us[category] += measured_us
        event_counts[category] += int(
            getattr(event, "count", 1)
        )

    total_communication_us = sum(totals_us.values())

    timer.record(
        phase="torchtitan_iteration_communication_total",
        seconds=total_communication_us / 1_000_000.0,
        round_id=round_id,
        iteration=iteration,
        global_step=global_step,
        extra=f"num_categories={len(totals_us)}",
    )

    for category, microseconds in sorted(totals_us.items()):
        timer.record(
            phase=f"torchtitan_iteration_{category}",
            seconds=microseconds / 1_000_000.0,
            round_id=round_id,
            iteration=iteration,
            global_step=global_step,
            extra=f"count={event_counts[category]}",
        )