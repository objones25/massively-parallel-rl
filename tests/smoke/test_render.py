"""Smoke test: render the shared real 2-iteration G1 checkpoint (``g1_checkpoint``) on CPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mprl.checkpoints import CheckpointRef
from mprl.negative_space import CheckFailed
from mprl.render import render_rollout

pytestmark = pytest.mark.smoke

STEP_DT_S = 0.02  # decimation 4 x timestep 0.005 s


@pytest.fixture
def checkpoint(g1_checkpoint: CheckpointRef) -> CheckpointRef:
    return g1_checkpoint


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
