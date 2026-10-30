"""GPU tier (``pytest -m gpu`` on the pod): real G1 training at a sweep-sized env count, and checkpoints on the volume.

``test_cuda_training_iteration`` runs two PPO iterations of the shipped G1 config at ``num_envs=4096`` (the long
run's count) on ``cuda:0`` through ``built_runner`` / ``learn_for``, the path ``train_run`` takes, and checks that
every loss rsl_rl's logger reports is finite.

``test_checkpoint_round_trip_on_volume`` trains one iteration on ``cuda:0`` into a ``CheckpointStore`` under
``MPRL_VOLUME_ROOT`` (default ``/workspace/mprl``: the network volume), then loads that file into a fresh runner and
checks that the iteration, the training clock and the policy weights come back. It skips when the volume root's
parent directory does not exist (no volume mounted).

``test_results_path_on_cuda`` evaluates and renders a one-iteration checkpoint on ``cuda:0``: the ``results`` command's
GPU path (EGL rendering through the host's NVIDIA driver, CUDA state copied to the renderer), which no training run
exercises. Run it before ``results``, so a rendering failure costs seconds, not the evaluations before it.
"""

from __future__ import annotations

import math
import os
import secrets
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from rsl_rl.utils.log_writer import LogWriter
from support import LONG_S

from mprl.checkpoints import CheckpointStore
from mprl.clock import TrainingClock, cuda_sync_for
from mprl.config import RunSpec
from mprl.evaluator import EvalConfig, evaluate_checkpoint
from mprl.paths import volume_root
from mprl.render import render_rollout
from mprl.train import built_runner

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]

# Read at import: the autouse _clean_env fixture removes MPRL_* variables before each test.
_ENV = dict(os.environ)

DEVICE = "cuda:0"
SWEEP_ENVS = 4096
SMALL_ENVS = 64
LOSS_TAGS = ("Loss/value", "Loss/surrogate", "Loss/entropy")


class ValueWriter(LogWriter):
    """A real rsl_rl LogWriter that keeps every scalar it is given, by (tag, step)."""

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self.values: dict[tuple[str, int], float] = {}

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.values[(tag, global_step)] = float(scalar_value)


def _spec(num_envs: int) -> RunSpec:
    return RunSpec("gputest", num_envs, 0, LONG_S / 60.0, LONG_S / 60.0)


def test_cuda_training_iteration(tmp_path: Path) -> None:
    iterations = 2
    with built_runner(_spec(SWEEP_ENVS), DEVICE, {"class_name": ValueWriter}, tmp_path / "logs") as runner:
        clock = TrainingClock(sync=cuda_sync_for(DEVICE))
        result = runner.learn_for(
            clock, LONG_S, LONG_S, CheckpointStore(tmp_path / "checkpoints"), fresh_start=True, max_iterations=iterations
        )
        writer = runner.logger.writer
        policy_devices = {p.device.type for p in runner.alg.get_policy().parameters()}
    assert isinstance(writer, ValueWriter)
    losses = {(tag, it): writer.values.get((tag, it), math.nan) for tag in LOSS_TAGS for it in range(iterations)}

    assert result.iterations_run == iterations
    assert policy_devices == {"cuda"}
    assert {key: value for key, value in losses.items() if not math.isfinite(value)} == {}


@pytest.fixture
def volume_dir() -> Iterator[Path]:
    """A fresh directory under the volume root, removed afterwards; skips when no volume is mounted."""
    root = volume_root(_ENV)
    if not root.parent.is_dir():
        pytest.skip(f"no volume: {root.parent} does not exist")
    directory = root / f"gpu-test-{secrets.token_hex(6)}"
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def test_checkpoint_round_trip_on_volume(volume_dir: Path) -> None:
    store = CheckpointStore(volume_dir / "checkpoints")
    with built_runner(_spec(SMALL_ENVS), DEVICE, "tensorboard", volume_dir / "logs") as trained:
        clock = TrainingClock(sync=cuda_sync_for(DEVICE))
        final = trained.learn_for(clock, LONG_S, LONG_S, store, fresh_start=True, max_iterations=1).final
        saved = {k: v.detach().cpu().clone() for k, v in trained.alg.get_policy().state_dict().items()}

    with built_runner(_spec(SMALL_ENVS), DEVICE, "tensorboard", volume_dir / "logs-reload") as fresh:
        infos = fresh.load(str(final.path))
        restored_iteration = fresh.current_learning_iteration
        loaded = {k: v.detach().cpu() for k, v in fresh.alg.get_policy().state_dict().items()}
    restored = TrainingClock.from_state_dict(infos["clock"])

    assert store.latest() == final
    assert final.path.is_relative_to(volume_dir)
    assert restored_iteration == final.iteration == 0
    assert restored.elapsed_s == pytest.approx(clock.elapsed_s)
    assert sorted(loaded) == sorted(saved)
    assert [k for k in saved if not torch.equal(saved[k], loaded[k])] == []


def test_results_path_on_cuda(tmp_path: Path) -> None:
    with built_runner(_spec(SMALL_ENVS), DEVICE, "tensorboard", tmp_path / "logs") as trained:
        clock = TrainingClock(sync=cuda_sync_for(DEVICE))
        ref = trained.learn_for(
            clock, LONG_S, LONG_S, CheckpointStore(tmp_path / "checkpoints"), fresh_start=True, max_iterations=1
        ).final
    cfg = EvalConfig(commands=((0.5, 0.0, 0.0),), envs_per_command=2, seeds=(0,), horizon_s=0.2)

    result = evaluate_checkpoint("gputest", ref, cfg, DEVICE)
    frames = render_rollout(ref, (0.5, 0.0, 0.0), 0.2, device=DEVICE)

    assert 0.0 <= result.no_fall_fraction <= 1.0
    assert frames.shape == (10, 360, 480, 3)  # round(0.2 s / 0.02 s)
    assert frames.dtype == np.uint8
    assert frames.max() > 0
