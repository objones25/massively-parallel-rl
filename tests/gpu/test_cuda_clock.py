"""GPU tier (``pytest -m gpu`` on the pod): the training clock synchronizes CUDA around every measured block.

Without the synchronize, ``clock.measure()`` would stop while kernels are still queued and the clock
would under-count GPU training time, unfairly for whichever env count queues the most work.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mprl.checkpoints import CheckpointStore
from mprl.clock import TrainingClock, cuda_sync_for
from mprl.config import RunSpec
from mprl.train import built_runner

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]

DEVICE = "cuda:0"
LONG_S = 3600.0
ITERATIONS = 2
MEASURES_PER_ITERATION = 2  # rollout + compute_returns, then the PPO update
SYNCS_PER_MEASURE = 2  # before the start time and before the stop time


def test_learn_for_on_cuda_synchronizes_around_every_measure(tmp_path: Path) -> None:
    real_sync = cuda_sync_for(DEVICE)
    calls: list[None] = []

    def counting_sync() -> None:
        calls.append(None)
        real_sync()

    spec = RunSpec("gputest", 64, 0, LONG_S / 60.0, LONG_S / 60.0)
    clock = TrainingClock(sync=counting_sync)
    with built_runner(spec, DEVICE, "tensorboard", tmp_path / "logs") as runner:
        result = runner.learn_for(
            clock, LONG_S, LONG_S, CheckpointStore(tmp_path / "checkpoints"), fresh_start=True, max_iterations=ITERATIONS
        )

    assert real_sync is torch.cuda.synchronize
    assert result.iterations_run == ITERATIONS
    assert len(calls) == ITERATIONS * MEASURES_PER_ITERATION * SYNCS_PER_MEASURE
    assert clock.elapsed_s > 0.0
