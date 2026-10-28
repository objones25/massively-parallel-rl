"""Pinning tests for mjlab 1.6.0 behavior the rest of the project builds on, on CPU with 4 envs.

They are characterization tests: they pass on first run and fail loudly on any library upgrade that
breaks an assumption. rsl_rl's own ``learn`` (2 iterations) and ``load`` (iteration and env step counter
restored) are pinned through our runner in tests/smoke/test_timed_runner.py
(``test_matches_rsl_rl_learn``, ``test_resume_in_fresh_runner_continues_iterations_and_clock``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from support import NUM_ENVS

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
