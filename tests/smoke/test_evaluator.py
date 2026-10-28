"""Smoke tests for the fixed-command evaluator against real mjlab / rsl_rl on CPU.

Checkpoints come from real 2-iteration training runs at 4 envs (the shared session ``g1_checkpoint``) and
at 8 envs for 3 iterations (here, for the tests whose variable is the training run: a different num_envs
and a different restored env step counter, 72 vs 48). The
horizon is kept short (0.3 s = 15 control steps at step_dt 0.02 s) and two envs per command, so each
evaluation steps 12 envs for 15 steps per seed.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Any

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


def _train(root: Path, num_envs: int, iterations: int) -> CheckpointRef:
    spec = RunSpec("evaltest", num_envs, 0, LONG_S / 60.0, LONG_S / 60.0)
    store = CheckpointStore(root / "checkpoints")
    with built_runner(spec, "cpu", "tensorboard", root / "logs") as runner:
        runner.learn_for(
            TrainingClock(sync=cuda_sync_for("cpu")), LONG_S, LONG_S, store, fresh_start=True, max_iterations=iterations
        )
    return expect(store.latest(), f"the {iterations}-iteration run saved no checkpoint")


@pytest.fixture
def checkpoint_4(g1_checkpoint: CheckpointRef) -> CheckpointRef:
    return g1_checkpoint


@pytest.fixture(scope="session")
def checkpoint_8(tmp_path_factory: pytest.TempPathFactory) -> CheckpointRef:
    return _train(tmp_path_factory.mktemp("n8"), 8, 3)


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
    # No env falls within 0.3 s, so every env's time to fall is the horizon (15 steps x 0.02 s) exactly.
    assert stats.no_fall_fraction == pytest.approx(1.0)
    assert stats.mean_time_to_fall_s == pytest.approx(HORIZON_STEPS * 0.02)


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
    assert result.no_fall_fraction == pytest.approx(1.0)  # nothing falls within 0.3 s
    assert result.mean_time_to_fall_s == pytest.approx(CFG.horizon_s)
    assert 0.0 < result.mean_tracking_reward <= 1.0  # a mean of exp(-err^2 / std^2) terms
    assert result.vel_error_xy >= 0.0
    assert result.vel_error_yaw >= 0.0


@dataclass(frozen=True)
class EvalSetup:
    """What one checkpoint's evaluation started from and ended with, for every seed of SEEDS_CFG."""

    num_envs: int
    restored_step_counter: int  # the training env's counter, restored into the eval env by runner.load
    foot_friction: torch.Tensor  # geom_friction after the startup randomization (env seed)
    first_obs: list[torch.Tensor]  # per seed: the actor observation of the first policy call
    first_state: list[torch.Tensor]  # per seed, at that call: robot root pose (world) and joint positions
    final_command: torch.Tensor  # twist command after the last seed's rollout


SEEDS_CFG = EvalConfig(seeds=(0, 1), envs_per_command=2, horizon_s=0.3)


def _eval_setup(ref: CheckpointRef) -> EvalSetup:
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []  # (actor obs, robot state) at every policy call

    def recording(policy: Any, robot: Any) -> Any:
        def call(obs: Any) -> torch.Tensor:
            seen.append((obs["actor"].clone(), torch.cat([robot.data.root_link_pose_w, robot.data.joint_pos], -1)))
            return policy(obs)

        return call

    with loaded_policy(ref, SEEDS_CFG, "cpu") as (env, policy):
        raw = env.unwrapped
        counter = raw.common_step_counter
        friction = raw.sim.model.geom_friction.clone()
        firsts = []
        for seed in SEEDS_CFG.seeds:
            start = len(seen)
            rollout(env, recording(policy, raw.scene["robot"]), SEEDS_CFG, seed)
            firsts.append(seen[start])
        twist = expect(raw.command_manager.get_command("twist"), "no twist command")
        return EvalSetup(
            env.num_envs, counter, friction, [o for o, _ in firsts], [st for _, st in firsts], twist.clone()
        )


@pytest.fixture(scope="module")
def eval_setups(g1_checkpoint: CheckpointRef, checkpoint_8: CheckpointRef) -> tuple[EvalSetup, EvalSetup]:
    return _eval_setup(g1_checkpoint), _eval_setup(checkpoint_8)


def test_eval_commands_and_env_count_do_not_depend_on_the_training_run(
    eval_setups: tuple[EvalSetup, EvalSetup],
) -> None:
    setup_4, setup_8 = eval_setups

    assert setup_4.num_envs == setup_8.num_envs == len(SEEDS_CFG.commands) * SEEDS_CFG.envs_per_command
    torch.testing.assert_close(setup_4.final_command, setup_8.final_command, rtol=0.0, atol=0.0)
    torch.testing.assert_close(setup_4.final_command, command_tensor(SEEDS_CFG, "cpu"), rtol=0.0, atol=0.0)


def test_eval_initial_state_does_not_depend_on_the_training_run(eval_setups: tuple[EvalSetup, EvalSetup]) -> None:
    """Same startup randomization and, per seed, the same first observation, whatever N and progress trained it."""
    setup_4, setup_8 = eval_setups

    assert (setup_4.restored_step_counter, setup_8.restored_step_counter) == (48, 72)  # the runs really differ
    assert len(setup_4.first_obs) == len(setup_8.first_obs) == len(SEEDS_CFG.seeds)
    torch.testing.assert_close(setup_4.foot_friction, setup_8.foot_friction, rtol=0.0, atol=0.0)
    torch.testing.assert_close(setup_4.first_state, setup_8.first_state, rtol=0.0, atol=0.0)
    torch.testing.assert_close(setup_4.first_obs, setup_8.first_obs, rtol=0.0, atol=0.0)
    assert not torch.allclose(setup_4.first_state[0], setup_4.first_state[1])  # each seed resets differently


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
