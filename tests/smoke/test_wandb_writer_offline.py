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


def test_every_iteration_metric_survives_rsl_rl_time_series(tmp_path: Path) -> None:
    """rsl_rl 5.4.2 writes ``Train/*/time`` at step=int(tot_time) after each iteration's metrics.

    Real wandb (offline) drops a step lower than the current one, so without the ``/time`` drop the summary
    would stay at iteration 0's value. The summary holds the last *committed* row: steps 0-3 (step 4 commits
    at finish), so it reads 3.
    """
    writer = ResumableWandbWriter(str(tmp_path / "logs"), "mprl-tests", RUN_ID, "g", mode="offline")
    try:
        for it in range(5):
            writer.add_scalar("Train/mean_reward", float(it), it)
            writer.add_scalar("Perf/train_clock_minutes", it * 0.5, it)
            writer.add_scalar("Train/mean_reward/time", float(it), 30)
        assert wandb.run is not None
        summary = dict(wandb.run.summary)
    finally:
        writer.stop()
    assert summary["Train/mean_reward"] == 3
    assert summary["_step"] == 3
    assert "Train/mean_reward/time" not in summary
