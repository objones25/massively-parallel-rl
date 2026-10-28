"""Shared fixtures for the smoke tests: a real G1 env + runner on CPU with 4 envs."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mjlab.tasks  # noqa: F401  (registers the tasks)
import pytest
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from mprl.checkpoints import CheckpointStore
from mprl.config import TASK_ID
from mprl.timed_runner import TimedOnPolicyRunner

NUM_ENVS = 4

Build = Callable[..., tuple[ManagerBasedRlEnv, Any]]


@pytest.fixture
def build(tmp_path: Path) -> Iterator[Build]:
    """Factory: (env, runner) for the shipped G1 config, seeded with its own seed; closes envs."""
    envs: list[ManagerBasedRlEnv] = []

    def _factory(
        runner_cls: type = TimedOnPolicyRunner, name: str = "logs", logger: object = "tensorboard"
    ) -> tuple[ManagerBasedRlEnv, Any]:
        env_cfg = load_env_cfg(TASK_ID)
        env_cfg.scene.num_envs = NUM_ENVS
        rl_cfg = load_rl_cfg(TASK_ID)
        env_cfg.seed = rl_cfg.seed  # as mjlab's train script does: one seed for env and agent
        env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
        envs.append(env)
        agent_cfg = asdict(rl_cfg)
        agent_cfg["logger"] = logger
        wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
        return env, runner_cls(wrapped, agent_cfg, str(tmp_path / name), "cpu")

    yield _factory
    for env in envs:
        env.close()


@pytest.fixture
def store(tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(tmp_path / "checkpoints")
