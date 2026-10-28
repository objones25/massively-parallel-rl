"""Training clock: counts only wall time spent inside ``measure()``.

Everything outside (checkpoint I/O, logging, evaluation, uploads) is off the clock, so runs at
different env counts are compared on equal training time.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from mprl.negative_space import require

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


def _no_sync() -> None:
    return None


class TrainingClock:
    def __init__(
        self,
        elapsed_s: float = 0.0,
        *,
        now: Callable[[], float] = time.perf_counter,
        sync: Callable[[], None] = _no_sync,
    ) -> None:
        require(elapsed_s >= 0.0, f"elapsed_s must be non-negative, got {elapsed_s}")
        self._elapsed_s = float(elapsed_s)
        self._now = now
        self._sync = sync
        self._inside = False

    @contextmanager
    def measure(self) -> Generator[None]:
        require(not self._inside, "nested measure(): the clock is already running")
        self._inside = True
        try:
            self._sync()
            start = self._now()
            try:
                yield
            finally:
                self._sync()
                delta = self._now() - start
                require(delta >= 0.0, f"negative interval {delta}: clock went backwards")
                self._elapsed_s += delta
        finally:
            self._inside = False

    @property
    def elapsed_s(self) -> float:
        return self._elapsed_s

    @property
    def elapsed_minutes(self) -> float:
        return self._elapsed_s / 60.0

    def state_dict(self) -> dict[str, float]:
        return {"elapsed_s": self._elapsed_s}

    @classmethod
    def from_state_dict(
        cls,
        d: dict[str, float],
        *,
        now: Callable[[], float] = time.perf_counter,
        sync: Callable[[], None] = _no_sync,
    ) -> TrainingClock:
        require("elapsed_s" in d, f"state dict lacks 'elapsed_s': {sorted(d)}")
        return cls(d["elapsed_s"], now=now, sync=sync)


def cuda_sync_for(device: str) -> Callable[[], None]:
    """``torch.cuda.synchronize`` for ``cuda:*`` devices, a no-op otherwise."""
    if device.startswith("cuda"):
        return torch.cuda.synchronize
    require(device == "cpu", f"unsupported device {device!r}")
    return _no_sync
