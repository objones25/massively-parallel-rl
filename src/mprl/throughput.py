"""Throughput test: env steps/s and whether each env count fits, on the real G1 env and runner.

Each count is built through ``mprl.train.built_runner`` (the construction path ``train_run`` uses)
and trained for ``iterations`` PPO iterations with ``learn_for`` and the same diagnostics hooks:

    steps_per_second[n] = n * num_steps_per_env (24) * iterations / training-clock seconds

``io_overhead_fraction`` is the off-clock time a sweep run pays per second of training clock, the
largest over the fitting counts:

    per-iteration off-clock (logging, diagnostics) / on-clock, over iterations 2..N
    + one checkpoint save's wall time / (checkpoint_every_minutes * 60)

Logging goes to TensorBoard here, not W&B, so W&B's own overhead is not included.

A count whose run raises the CUDA ``torch.OutOfMemoryError`` (verified in torch 2.14.1:
``torch.cuda.OutOfMemoryError is torch.OutOfMemoryError``, a ``RuntimeError`` subclass) is recorded
as ``fits[n] = False`` and the test continues; every other error propagates.

``measure_throughput`` takes ``eval_seconds_per_checkpoint`` from the caller; ``measure_eval_seconds``
measures it as the wall time of one ``evaluate_checkpoint`` on a count's throughput checkpoint (env
build included, as ``mprl.results`` builds one env per checkpoint; Warp kernels compiled on first use
are included too, so it errs high).
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import torch

from mprl import diagnostics
from mprl.budget import Measurements
from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.clock import TrainingClock, cuda_sync_for
from mprl.config import RunSpec
from mprl.evaluator import EvalConfig, evaluate_checkpoint
from mprl.negative_space import check_finite, expect, require
from mprl.train import TrainDeps, built_runner

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

log = logging.getLogger(__name__)

_SWEEP: Final = "throughput"
_NEVER_S: Final = 1e9  # budget and checkpoint period that max_iterations always beats
_DIAGNOSTICS_EVERY_K: Final = TrainDeps(None, "disabled", "cpu").diagnostics_every_k  # as train_run


@dataclass(frozen=True)
class CountMeasurement:
    steps_per_second: float
    io_overhead_fraction: float


class _TimedStore(CheckpointStore):
    """A ``CheckpointStore`` that records the wall time of each save."""

    def __init__(self, directory: Path) -> None:
        super().__init__(directory)
        self.save_seconds: list[float] = []

    def save_atomic(self, write: Callable[[Path], None], iteration: int, train_seconds: float) -> CheckpointRef:
        start = time.perf_counter()
        ref = super().save_atomic(write, iteration, train_seconds)
        self.save_seconds.append(time.perf_counter() - start)
        return ref


def measure_count(
    num_envs: int, iterations: int, device: str, work_dir: Path, checkpoint_every_minutes: float
) -> CountMeasurement:
    """Train ``iterations`` PPO iterations at ``num_envs`` and measure steps/s and off-clock overhead."""
    require(iterations >= 2, f"iterations must be >= 2 to time an iteration interval, got {iterations}")
    spec = RunSpec(_SWEEP, num_envs, 0, checkpoint_every_minutes, checkpoint_every_minutes)
    store = _TimedStore(work_dir / "checkpoints")
    marks: list[tuple[float, float]] = []  # (wall, training clock) at the end of each iteration
    with built_runner(spec, device, "tensorboard", work_dir / "tensorboard") as runner:
        clock = TrainingClock(sync=cuda_sync_for(device))
        diag, on_before_update, diag_on_iteration = diagnostics.for_runner(runner, _DIAGNOSTICS_EVERY_K)

        def on_iteration(it: int) -> None:
            diag_on_iteration(it)
            marks.append((time.perf_counter(), clock.elapsed_s))

        diag.attach()
        try:
            result = runner.learn_for(
                clock,
                _NEVER_S,
                _NEVER_S,
                store,
                fresh_start=True,
                max_iterations=iterations,
                on_iteration=on_iteration,
                on_before_update=on_before_update,
            )
        finally:
            diag.detach()
        steps_per_env = int(runner.cfg["num_steps_per_env"])
    require(result.iterations_run == iterations, f"ran {result.iterations_run} of {iterations} iterations")
    require(len(marks) == iterations, f"expected {iterations} iteration marks, got {len(marks)}")
    require(len(store.save_seconds) == 1, f"expected only the final save, got {len(store.save_seconds)}")
    wall = marks[-1][0] - marks[0][0]
    on_clock = marks[-1][1] - marks[0][1]
    require(on_clock > 0.0, f"no training clock between iterations 1 and {iterations}")
    per_iteration = max(0.0, wall - on_clock) / on_clock
    save = store.save_seconds[0] / (checkpoint_every_minutes * 60.0)
    sps = num_envs * steps_per_env * iterations / clock.elapsed_s
    check_finite(sps, "steps_per_second")
    return CountMeasurement(sps, per_iteration + save)


def count_dir(work_dir: Path, num_envs: int) -> Path:
    """Scratch dir of one count's measurement; its final checkpoint is under ``checkpoints/``."""
    return work_dir / f"n{num_envs:05d}"


def measure_eval_seconds(work_dir: Path, num_envs: int, device: str, cfg: EvalConfig) -> float:
    """Wall seconds of one ``evaluate_checkpoint`` with ``cfg`` on ``num_envs``'s throughput checkpoint."""
    ref = expect(
        CheckpointStore(count_dir(work_dir, num_envs) / "checkpoints").latest(),
        f"no throughput checkpoint for num_envs={num_envs} under {work_dir}",
    )
    start = time.perf_counter()
    evaluate_checkpoint(_SWEEP, ref, cfg, device)
    return time.perf_counter() - start


def _release_device_memory(device: str) -> None:
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def measure_throughput(
    env_counts: Sequence[int],
    iterations: int,
    device: str,
    *,
    work_dir: Path,
    eval_seconds_per_checkpoint: float,
    checkpoint_every_minutes: float = 3.0,
) -> Measurements:
    """Measure every count in ``env_counts`` (scratch files under ``work_dir``); OOM is recorded, not raised."""
    require(len(env_counts) > 0, "no env counts to measure")
    require(len(set(env_counts)) == len(env_counts), f"duplicate env counts: {list(env_counts)}")
    fits: dict[int, bool] = {}
    sps: dict[int, float] = {}
    overheads: list[float] = []
    for n in env_counts:
        measured: CountMeasurement | None = None
        try:
            measured = measure_count(n, iterations, device, count_dir(work_dir, n), checkpoint_every_minutes)
        except torch.OutOfMemoryError as error:
            # Log the text, not the exception: a handler that keeps LogRecords would keep its traceback.
            log.warning("num_envs=%d does not fit on %s: %s", n, device, str(error))
        # Released only here: inside the except, error.__traceback__ still pins the failed run's frames
        # (runner, policy, rollout tensors), so nothing could be freed. Python drops `error` at the
        # end of the except block.
        _release_device_memory(device)
        fits[n] = measured is not None
        if measured is None:
            continue
        sps[n] = measured.steps_per_second
        overheads.append(measured.io_overhead_fraction)
        log.info("num_envs=%d: %.0f env steps/s, off-clock overhead %.3f", n, sps[n], overheads[-1])
    return Measurements(eval_seconds_per_checkpoint, max(overheads, default=0.0), fits, sps)
