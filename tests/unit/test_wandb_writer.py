"""Unit tests for mprl.wandb_writer's pure parts (the offline runs are in tests/smoke/test_wandb_writer_offline.py)."""

from __future__ import annotations

import inspect

import wandb

from mprl.wandb_writer import logger_cfg


def test_wandb_init_accepts_the_keywords_we_pass() -> None:
    params = set(inspect.signature(wandb.init).parameters)
    passed = {"id", "resume", "resume_from", "group", "job_type", "mode", "entity", "tags", "config"}
    assert passed <= params


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
