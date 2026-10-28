from __future__ import annotations

import pytest

from mprl.negative_space import CheckFailed
from mprl.timed_runner import next_checkpoint_due


@pytest.mark.parametrize(
    ("elapsed_s", "every_s", "last_saved_s", "due"),
    [
        (179.0, 180.0, 0.0, False),
        (181.0, 180.0, 0.0, True),
        (361.0, 180.0, 180.0, True),
        (361.0, 180.0, 360.0, False),
        (180.0, 180.0, 0.0, True),  # reaching the multiple exactly counts as crossing it
        (540.0, 180.0, 361.0, True),  # last save off-grid: next due at the next multiple, 540
        (539.0, 180.0, 361.0, False),
    ],
)
def test_due_when_next_multiple_crossed(elapsed_s: float, every_s: float, last_saved_s: float, due: bool) -> None:
    assert next_checkpoint_due(elapsed_s, every_s, last_saved_s) is due


@pytest.mark.parametrize(
    ("elapsed_s", "every_s", "last_saved_s", "match"),
    [
        (10.0, 0.0, 0.0, "every_s"),
        (10.0, -1.0, 0.0, "every_s"),
        (10.0, 180.0, 20.0, "last_saved_s"),
        (-1.0, 180.0, 0.0, "elapsed_s"),
        (float("nan"), 180.0, 0.0, "elapsed_s"),
    ],
)
def test_rejects_impossible_inputs(elapsed_s: float, every_s: float, last_saved_s: float, match: str) -> None:
    with pytest.raises(CheckFailed, match=match):
        next_checkpoint_due(elapsed_s, every_s, last_saved_s)
