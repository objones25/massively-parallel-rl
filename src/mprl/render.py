"""Rollout frames for one checkpoint, from the evaluator's play env with the command pinned.

DRY with ``mprl.evaluator``: a one-env ``EvalConfig`` builds the env (``loaded_policy``), pins the
command (``pin_command``) and fixes the horizon (``horizon_steps``). Rendering is off the training clock.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mprl.evaluator import EvalConfig, command_tensor, horizon_steps, loaded_policy, pin_command, twist_term
from mprl.negative_space import check_shape, expect

if TYPE_CHECKING:
    from mprl.checkpoints import CheckpointRef


def render_rollout(
    ref: CheckpointRef,
    command: tuple[float, float, float],
    seconds: float,
    *,
    height: int = 360,
    width: int = 480,
    seed: int = 0,
    device: str = "cpu",
) -> np.ndarray:
    """``(T, height, width, 3)`` uint8: one robot following ``command`` for ``T = seconds / step_dt`` steps."""
    cfg = EvalConfig(commands=(command,), envs_per_command=1, seeds=(seed,), horizon_s=seconds)
    frames: list[np.ndarray] = []
    with loaded_policy(ref, cfg, device, render_size=(height, width)) as (env, policy):
        raw = env.unwrapped
        twist = twist_term(raw)
        pinned = command_tensor(cfg, raw.device)
        steps = horizon_steps(cfg, raw.step_dt)
        with torch.inference_mode():
            raw.reset(seed=seed)
            pin_command(twist, pinned)
            raw.observation_manager.compute(update_history=True)  # observations with the pinned command
            obs = env.get_observations()
            for _ in range(steps):
                obs, _, _, _ = env.step(policy(obs))
                pin_command(twist, pinned)  # a fallen robot is reset and its command resampled
                frame = expect(raw.render(), "env.render() returned None; was render_mode set?")
                frames.append(frame)
    video = np.stack(frames)
    check_shape(video, (steps, height, width, 3), name="rendered frames")
    return video
