"""Shared fixtures for the smoke tier: the real G1 env + runner on CPU, and the few real trainings it shares.

Real training happens only where it is the behavior under test. Tests that only need a trained input
read one of two session artifacts, built once and never mutated (copy before changing anything):

* ``g1_checkpoint``: one 2-iteration, 4-env checkpoint (real weights) for evaluation, rendering and
  the results pipeline.
* ``completed_run_template``: one real ``train_run`` (2 s of training clock, checkpoint every 1 s,
  W&B offline) on a volume, copied per test by ``completed_run``.
"""

from __future__ import annotations

import copy
import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import wandb
from mjlab.envs import ManagerBasedRlEnv
from support import APP_ENV_PREFIXES, LONG_S, NUM_ENVS, CompletedRun, make_env_runner, run_spec

from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.clock import TrainingClock, cuda_sync_for
from mprl.config import RunSpec
from mprl.negative_space import expect
from mprl.train import TrainDeps, built_runner, train_run

Build = Callable[..., tuple[ManagerBasedRlEnv, Any]]

@pytest.fixture
def build(tmp_path: Path) -> Iterator[Build]:
    """Factory: ``build(name=..., **make_env_runner kwargs)``; closes every env it built."""
    envs: list[ManagerBasedRlEnv] = []

    def _factory(name: str = "logs", **kwargs: Any) -> tuple[ManagerBasedRlEnv, Any]:
        env, runner = make_env_runner(tmp_path / name, **kwargs)
        envs.append(env)
        return env, runner

    yield _factory
    for env in envs:
        env.close()


@pytest.fixture
def store(tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(tmp_path / "checkpoints")


@pytest.fixture
def offline_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """wandb reads WANDB_DIR once per service; restart it so the test's offline runs land in tmp_path."""
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    wandb.teardown()
    yield
    wandb.teardown()


@pytest.fixture(scope="session")
def g1_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> CheckpointRef:
    """A real 2-iteration, 4-env G1 checkpoint. Read only; copy it to build other run layouts."""
    root = tmp_path_factory.mktemp("g1")
    spec = RunSpec("fixture", NUM_ENVS, 0, LONG_S / 60.0, LONG_S / 60.0)
    store = CheckpointStore(root / "checkpoints")
    with built_runner(spec, "cpu", "tensorboard", root / "logs") as runner:
        runner.learn_for(
            TrainingClock(sync=cuda_sync_for("cpu")), LONG_S, LONG_S, store, fresh_start=True, max_iterations=2
        )
    return expect(store.latest(), "the fixture run saved no checkpoint")


@pytest.fixture(scope="session")
def completed_run_template(tmp_path_factory: pytest.TempPathFactory) -> CompletedRun:
    """One real ``train_run(run_spec(2.0))`` with W&B offline. Never mutated: tests use ``completed_run``."""
    base = tmp_path_factory.mktemp("completed")
    with pytest.MonkeyPatch.context() as mp:  # a session fixture runs before the per-test env cleanup
        for key in list(os.environ):
            if key.startswith(APP_ENV_PREFIXES):
                mp.delenv(key)
        mp.setenv("WANDB_DIR", str(base / "wandbdir"))
        wandb.teardown()
        try:
            state = train_run(run_spec(2.0), base / "volume", TrainDeps(publisher=None, wandb_mode="offline", device="cpu"))
        finally:
            wandb.teardown()
    return CompletedRun(base / "volume", base / "wandbdir", state)


@pytest.fixture
def completed_run(completed_run_template: CompletedRun, tmp_path: Path) -> CompletedRun:
    """Private copies of the completed run's volume and RunState (W&B dir not copied), safe to resume or mutate."""
    volume = Path(shutil.copytree(completed_run_template.volume, tmp_path / "volume"))
    return CompletedRun(volume, completed_run_template.wandb_dir, copy.deepcopy(completed_run_template.state))
