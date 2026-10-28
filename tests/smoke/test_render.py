"""Smoke test: render a real 2-iteration G1 checkpoint on CPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.clock import TrainingClock, cuda_sync_for
from mprl.config import RunSpec
from mprl.negative_space import CheckFailed, expect
from mprl.render import render_rollout
from mprl.train import built_runner

pytestmark = pytest.mark.smoke

LONG_S = 3600.0
STEP_DT_S = 0.02  # decimation 4 x timestep 0.005 s


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> CheckpointRef:
    root = tmp_path_factory.mktemp("render")
    spec = RunSpec("rendertest", 4, 0, LONG_S / 60.0, LONG_S / 60.0)
    store = CheckpointStore(root / "checkpoints")
    with built_runner(spec, "cpu", "tensorboard", root / "logs") as runner:
        runner.learn_for(
            TrainingClock(sync=cuda_sync_for("cpu")), LONG_S, LONG_S, store, fresh_start=True, max_iterations=2
        )
    return expect(store.latest(), "the 2-iteration run saved no checkpoint")


def test_render_rollout_cpu(checkpoint: CheckpointRef):
    frames = render_rollout(checkpoint, (0.5, 0.0, 0.0), 0.2)

    assert frames.shape == (10, 360, 480, 3)  # round(0.2 s / 0.02 s)
    assert frames.dtype == np.uint8
    assert frames.max() > 0
    assert not np.array_equal(frames[0], frames[-1])  # the robot moved


def test_render_rollout_honours_size(checkpoint: CheckpointRef):
    frames = render_rollout(checkpoint, (0.0, 0.0, 0.5), 0.1, height=120, width=160)

    assert frames.shape == (5, 120, 160, 3)


def test_render_rollout_rejects_missing_checkpoint(tmp_path: Path):
    ghost = CheckpointRef(1, 1.0, tmp_path / "nope.pt")
    with pytest.raises(CheckFailed):
        render_rollout(ghost, (0.5, 0.0, 0.0), 0.1)
