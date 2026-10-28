"""Smoke test: ResumableWandbWriter inside a real rsl_rl Logger, with real wandb in offline mode."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import wandb
from rsl_rl.utils.logger import Logger

from mprl.wandb_writer import ResumableWandbWriter, logger_cfg

pytestmark = [pytest.mark.smoke, pytest.mark.usefixtures("offline_wandb")]  # WANDB_DIR = tmp_path

RUN_ID = "run0abc"


def _logger(tmp_path: Path, cfg: dict) -> Logger:
    return Logger(
        log_dir=str(tmp_path / "logs"),
        cfg={"algorithm": {"rnd_cfg": None}, "logger": cfg},
        env_cfg={"decimation": 4},
        num_envs=2,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device="cpu",
    )


def test_logger_builds_writer_and_logs_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    logger = _logger(tmp_path, logger_cfg("mprl-tests", RUN_ID, "g", None, "offline"))
    try:
        logger.init_logging_writer()
        assert isinstance(logger.writer, ResumableWandbWriter)
        logger.writer.add_scalar("Loss/x", 1.0, 0)
        assert wandb.run is not None
        assert wandb.run.id == RUN_ID
        assert wandb.run.config["train_cfg"]["algorithm"] == {"rnd_cfg": None}
        assert wandb.run.config["env_cfg"] == {"decimation": 4}
    finally:
        logger.stop_logging_writer()
    run_dirs = list((tmp_path / "wandb").glob(f"offline-run-*-{RUN_ID}*"))
    assert len(run_dirs) == 1


def test_resume_from_iteration_uses_resume_from(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With an iteration given, the writer asks wandb to rewind to that step (beta resume_from)."""
    captured: dict[str, object] = {}
    real_init = wandb.init

    def spy(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        # offline mode cannot rewind a server-side run, so start the underlying run without resume_from
        kwargs.pop("resume_from")
        kwargs["id"] = RUN_ID
        return real_init(*args, **kwargs)

    monkeypatch.setattr(wandb, "init", spy)
    writer = ResumableWandbWriter(
        str(tmp_path / "logs"), "mprl-tests", RUN_ID, "g", mode="offline", resume_from_iteration=7
    )
    writer.stop()
    assert captured["resume_from"] == f"{RUN_ID}?_step=7"
    assert "id" not in captured
    assert "resume" not in captured


def test_store_config_accepts_real_mjlab_env_cfg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, build: Callable[..., tuple[object, Any]]
) -> None:
    """Real G1 env cfg (callables, class refs, nested configs) must survive the runner's init_logging_writer."""
    try:
        _, runner = build(logger=logger_cfg("mprl-tests", RUN_ID, "g", None, "offline"))
        runner.logger.init_logging_writer()  # what learn_for does at start, with the runner's real env cfg
        assert wandb.run is not None
        assert "scene" in wandb.run.config["env_cfg"]
        assert "algorithm" in wandb.run.config["train_cfg"]
    finally:
        wandb.finish()
