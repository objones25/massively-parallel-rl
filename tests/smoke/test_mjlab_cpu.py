"""Pinning tests for mjlab 1.6.0 / rsl_rl 5.4.2 behavior the rest of the project builds on.

They exercise the real libraries on CPU with 4 envs. They are characterization tests: they pass
on first run and fail loudly on any library upgrade that breaks an assumption.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mjlab.tasks  # noqa: F401  (registers the tasks)
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

pytestmark = pytest.mark.smoke

TASK = "Mjlab-Velocity-Flat-Unitree-G1"
NUM_ENVS = 4
STEPS_PER_ITERATION = 24  # num_steps_per_env in the shipped G1 PPO config


def _build(log_dir: Path, *, play: bool = False, num_envs: int = NUM_ENVS) -> tuple[ManagerBasedRlEnv, Any]:
    env_cfg = load_env_cfg(TASK, play=play)
    env_cfg.scene.num_envs = num_envs
    rl_cfg = load_rl_cfg(TASK)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode="rgb_array")
    agent_cfg = asdict(rl_cfg)
    agent_cfg["logger"] = "tensorboard"
    runner = load_runner_cls(TASK)(RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions), agent_cfg, str(log_dir), "cpu")
    return env, runner


@pytest.fixture
def build(tmp_path: Path) -> Callable[..., tuple[ManagerBasedRlEnv, Any]]:
    """Factory that builds (env, runner) and closes every env at teardown."""
    envs: list[ManagerBasedRlEnv] = []

    def _factory(log_dir: Path | None = None, **kwargs: Any) -> tuple[ManagerBasedRlEnv, Any]:
        env, runner = _build(log_dir if log_dir is not None else tmp_path / "logs", **kwargs)
        envs.append(env)
        return env, runner

    yield _factory
    for env in envs:
        env.close()


def test_g1_flat_env_steps_on_cpu(build) -> None:
    env, _ = build()
    env.reset()
    actions = torch.zeros(NUM_ENVS, env.action_manager.total_action_dim, device="cpu")

    for _ in range(24):
        obs, _rew, _terminated, _truncated, _extras = env.step(actions)

    assert all(torch.isfinite(t).all().item() for t in obs.values())
    assert env.common_step_counter == 24


def test_g1_runner_trains_two_iterations_on_cpu(build, tmp_path: Path) -> None:
    log_dir = tmp_path / "train"
    _, runner = build(log_dir)

    runner.learn(num_learning_iterations=2)

    assert (log_dir / "model_0.pt").is_file()
    assert (log_dir / "model_1.pt").is_file()


def test_fresh_runner_restores_iteration_and_step_counter(build, tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    env, runner = build(train_dir)
    runner.learn(num_learning_iterations=3)
    assert env.common_step_counter == 3 * STEPS_PER_ITERATION

    fresh_env, fresh_runner = build(tmp_path / "fresh")
    fresh_runner.load(str(train_dir / "model_2.pt"))

    assert fresh_runner.current_learning_iteration == 2
    assert fresh_env.common_step_counter == 72


def test_play_env_renders_rgb_frame(build) -> None:
    env, _ = build(play=True, num_envs=1)
    env.reset()

    frame = env.render()

    assert frame.shape == (240, 320, 3)
    assert frame.dtype == "uint8"
