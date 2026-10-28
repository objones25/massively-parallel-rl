"""Smoke tests for the fixed-command evaluator against real mjlab / rsl_rl on CPU.

Checkpoints come from real 2-iteration training runs at 4 envs (the shared session ``g1_checkpoint``) and
at 8 envs (here, for the one test whose variable is the training num_envs). The
horizon is kept short (0.3 s = 15 control steps at step_dt 0.02 s) and two envs per command, so each
evaluation steps 12 envs for 15 steps per seed.
"""

from __future__ import annotations

from dataclasses import astuple
from pathlib import Path

import pytest
import torch
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from support import LONG_S

from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.clock import TrainingClock, cuda_sync_for
from mprl.config import RunSpec
from mprl.evaluator import EvalConfig, command_tensor, eval_env_cfg, evaluate_checkpoint, loaded_policy, rollout
from mprl.negative_space import expect
from mprl.train import built_runner

pytestmark = pytest.mark.smoke

CFG = EvalConfig(envs_per_command=2, horizon_s=0.3)
FALL_CFG = EvalConfig(envs_per_command=2, horizon_s=2.5)
HORIZON_STEPS = 15  # 0.3 s / step_dt 0.02 s (decimation 4 x timestep 0.005 s)


def _train_two_iterations(root: Path, num_envs: int) -> CheckpointRef:
    spec = RunSpec("evaltest", num_envs, 0, LONG_S / 60.0, LONG_S / 60.0)
    store = CheckpointStore(root / "checkpoints")
    with built_runner(spec, "cpu", "tensorboard", root / "logs") as runner:
        runner.learn_for(
            TrainingClock(sync=cuda_sync_for("cpu")), LONG_S, LONG_S, store, fresh_start=True, max_iterations=2
        )
    return expect(store.latest(), "the 2-iteration run saved no checkpoint")


@pytest.fixture
def checkpoint_4(g1_checkpoint: CheckpointRef) -> CheckpointRef:
    return g1_checkpoint


@pytest.fixture(scope="session")
def checkpoint_8(tmp_path_factory: pytest.TempPathFactory) -> CheckpointRef:
    return _train_two_iterations(tmp_path_factory.mktemp("n8"), 8)


def test_commands_are_held_fixed(checkpoint_4: CheckpointRef) -> None:
    # Long enough that the 2-iteration policy falls (all 12 envs by ~1.6 s on average at 3 s), so the
    # auto-reset path, which resamples a fallen env's command, is exercised too.
    with loaded_policy(checkpoint_4, FALL_CFG, "cpu") as (env, policy):
        stats = rollout(env, policy, FALL_CFG, seed=0)
        twist = env.unwrapped.command_manager.get_command("twist")

    assert stats.no_fall_fraction < 1.0
    assert twist is not None
    assert env.num_envs == len(FALL_CFG.commands) * FALL_CFG.envs_per_command
    torch.testing.assert_close(twist, command_tensor(FALL_CFG, "cpu"), rtol=0.0, atol=0.0)


def test_rollout_steps_exactly_the_horizon(checkpoint_4: CheckpointRef) -> None:
    with loaded_policy(checkpoint_4, CFG, "cpu") as (env, policy):
        start = env.unwrapped.common_step_counter
        stats = rollout(env, policy, CFG, seed=0)
        stepped = env.unwrapped.common_step_counter - start

    assert stepped == HORIZON_STEPS
    assert 0.0 < stats.mean_time_to_fall_s <= CFG.horizon_s


def test_same_checkpoint_same_result(checkpoint_4: CheckpointRef) -> None:
    first = evaluate_checkpoint("run-a", checkpoint_4, CFG, "cpu")
    second = evaluate_checkpoint("run-a", checkpoint_4, CFG, "cpu")

    assert first == second


def test_result_metrics_finite_and_bounded(checkpoint_4: CheckpointRef) -> None:
    result = evaluate_checkpoint("run-a", checkpoint_4, CFG, "cpu")

    assert result.run_id == "run-a"
    assert result.iteration == checkpoint_4.iteration
    assert result.train_minutes == pytest.approx(checkpoint_4.train_seconds / 60.0)
    metrics = torch.tensor(astuple(result)[2:], dtype=torch.float64)  # every field after run_id, iteration
    assert bool(torch.isfinite(metrics).all()), result
    assert 0.0 <= result.no_fall_fraction <= 1.0
    assert 0.0 < result.mean_time_to_fall_s <= CFG.horizon_s
    assert 0.0 < result.mean_tracking_reward <= 1.0
    assert result.vel_error_xy >= 0.0
    assert result.vel_error_yaw >= 0.0


def _envs_and_commands_after_rollout(ref: CheckpointRef) -> tuple[int, torch.Tensor]:
    with loaded_policy(ref, CFG, "cpu") as (env, policy):
        rollout(env, policy, CFG, seed=1)
        twist = expect(env.unwrapped.command_manager.get_command("twist"), "no twist command")
        return env.num_envs, twist.clone()


def test_eval_does_not_depend_on_training_num_envs(checkpoint_4: CheckpointRef, checkpoint_8: CheckpointRef) -> None:
    envs_4, twist_4 = _envs_and_commands_after_rollout(checkpoint_4)
    envs_8, twist_8 = _envs_and_commands_after_rollout(checkpoint_8)

    assert envs_4 == envs_8 == len(CFG.commands) * CFG.envs_per_command
    torch.testing.assert_close(twist_4, twist_8, rtol=0.0, atol=0.0)
    torch.testing.assert_close(twist_4, command_tensor(CFG, "cpu"), rtol=0.0, atol=0.0)


def test_eval_env_cfg_disables_every_command_override() -> None:
    env_cfg = eval_env_cfg(CFG)
    twist = env_cfg.commands["twist"]

    assert isinstance(twist, UniformVelocityCommandCfg)
    assert env_cfg.scene.num_envs == len(CFG.commands) * CFG.envs_per_command
    assert not twist.heading_command
    assert twist.ranges.heading is None
    overrides = (
        twist.rel_standing_envs,
        twist.rel_heading_envs,
        twist.rel_world_envs,
        twist.rel_forward_envs,
        twist.init_velocity_prob,
    )
    assert overrides == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert min(twist.resampling_time_range) > CFG.horizon_s
    assert env_cfg.curriculum == {}
    assert set(env_cfg.terminations) == {"time_out", "fell_over"}
