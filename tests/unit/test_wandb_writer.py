"""Unit tests for mprl.wandb_writer's pure parts (the offline runs are in tests/smoke/test_wandb_writer_offline.py)."""

from __future__ import annotations

import inspect
from unittest.mock import create_autospec

import pytest
import wandb

from mprl.wandb_writer import ResumableWandbWriter, logger_cfg


@pytest.mark.parametrize(
    ("resume_from_iteration", "run_keys"),
    [(None, {"id", "resume"}), (3, {"resume_from"})],
    ids=["fresh", "resumed"],
)
def test_wandb_init_accepts_the_keywords_we_pass(
    resume_from_iteration: int | None, run_keys: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The keywords the writer really passes (captured from its call), checked against wandb.init's signature."""
    init = create_autospec(wandb.init)  # an autospec double rejects a keyword wandb.init does not take
    monkeypatch.setattr(wandb, "init", init)

    ResumableWandbWriter("logs", "p", "rid", "grp", resume_from_iteration=resume_from_iteration, mode="disabled")

    passed = set(init.call_args.kwargs)
    common = {"project", "entity", "group", "job_type", "mode", "tags", "config"}
    assert passed == common | run_keys
    assert passed <= set(inspect.signature(wandb.init).parameters)


def test_logger_cfg_shape() -> None:
    cfg = logger_cfg("p", "rid", "grp", 3, "offline")
    assert cfg == {
        "class_name": "mprl.wandb_writer:ResumableWandbWriter",
        "project": "p",
        "run_id": "rid",
        "group": "grp",
        "resume_from_iteration": 3,
        "mode": "offline",
    }
