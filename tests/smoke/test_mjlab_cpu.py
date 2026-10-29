"""Pinning tests for mjlab 1.6.0 behavior the rest of the project builds on, on CPU with 4 envs.

They are characterization tests: they pass on first run and fail loudly on any library upgrade that
breaks an assumption. The two termination tests pin what makes PPO bootstrap a time-limit end
(rsl_rl/algorithms/ppo.py ``process_env_step`` adds ``gamma * value * extras["time_outs"]`` to the reward)
and treat a fall as terminal. rsl_rl's own ``learn`` (2 iterations) and ``load`` (iteration and env step counter
restored) are pinned through our runner in tests/smoke/test_timed_runner.py
(``test_matches_rsl_rl_learn``, ``test_resume_in_fresh_runner_continues_iterations_and_clock``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from support import LONG_S, NUM_ENVS, STEPS_PER_ITERATION

from mprl.checkpoints import CheckpointStore
from mprl.clock import TrainingClock

pytestmark = pytest.mark.smoke

STEPS = 24


def test_g1_flat_env_steps_on_cpu(build: Callable[..., tuple[Any, Any]]) -> None:
    env, _ = build()
    env.reset()
    actions = torch.zeros(NUM_ENVS, env.action_manager.total_action_dim, device="cpu")

    for _ in range(STEPS):
        obs, _rew, _terminated, _truncated, _extras = env.step(actions)

    assert all(torch.isfinite(t).all().item() for t in obs.values())
    assert env.common_step_counter == STEPS


def test_play_env_renders_rgb_frame(build: Callable[..., tuple[Any, Any]]) -> None:
    env, _ = build(play=True, num_envs=1, render_mode="rgb_array")
    env.reset()

    frame = env.render()

    assert frame.shape == (240, 320, 3)
    assert frame.dtype == "uint8"


STEP_DT_S = 0.02  # decimation 4 x timestep 0.005 s
EPISODE_STEPS = 3  # a time limit every 3 control steps: 8 per env in one 24-step rollout


def _short_episodes(env_cfg: Any) -> None:
    env_cfg.episode_length_s = EPISODE_STEPS * STEP_DT_S


def _always_fallen(env_cfg: Any) -> None:
    env_cfg.terminations["fell_over"].params["limit_angle"] = -1.0  # |angle| > -1 rad: fires on every step


def test_time_limit_reaches_ppo_as_time_out_not_terminal(build: Callable[..., tuple[Any, Any]], tmp_path: Any) -> None:
    """Through learn_for's own rollout: every done is a time limit, flagged in extras["time_outs"] for PPO."""
    env, runner = build(configure_env=_short_episodes)
    seen: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    real_process = runner.alg.process_env_step

    def spy(obs: Any, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, Any]) -> None:
        time_outs = extras.get("time_outs")
        seen.append((dones.clone(), None if time_outs is None else time_outs.clone()))
        real_process(obs, rewards, dones, extras)

    runner.alg.process_env_step = spy

    runner.learn_for(
        TrainingClock(), LONG_S, LONG_S, CheckpointStore(tmp_path / "ckpt"), fresh_start=True, max_iterations=1
    )

    assert env.max_episode_length == EPISODE_STEPS
    assert len(seen) == STEPS_PER_ITERATION
    assert [time_outs is not None for _, time_outs in seen] == [True] * STEPS_PER_ITERATION
    dones = torch.stack([d for d, _ in seen])
    time_outs = torch.stack([t for _, t in seen if t is not None])
    assert int(time_outs.sum()) == NUM_ENVS * STEPS_PER_ITERATION // EPISODE_STEPS
    assert torch.equal(time_outs.long(), dones)  # nothing else ended an episode


def test_fall_is_terminal_not_a_time_out(build: Callable[..., tuple[Any, Any]]) -> None:
    _, runner = build(configure_env=_always_fallen)
    wrapped = runner.env

    _obs, _rew, dones, extras = wrapped.step(torch.zeros(NUM_ENVS, wrapped.num_actions))

    assert dones.tolist() == [1] * NUM_ENVS
    assert extras["time_outs"].tolist() == [False] * NUM_ENVS
