"""Fixed-command checkpoint evaluation: the "walking quality" score every run is judged by.

Every checkpoint, whatever ``num_envs`` it trained at, is evaluated identically: the shipped play env
(``load_env_cfg(TASK_ID, play=True)``: no observation corruption, no pushes, no curriculum) with
``len(commands) * envs_per_command`` envs, the same per-env pinned commands, the same seeds and the same
horizon. Evaluation is off the training clock.

How a command is pinned (mjlab 1.6.0, ``mjlab/tasks/velocity/mdp/velocity_command.py`` and
``mjlab/managers/command_manager.py``):

* ``CommandTerm.compute`` resamples an env when its ``time_left`` (drawn from ``resampling_time_range``
  at reset) reaches 0, so ``resampling_time_range`` is set to ``(1e9, 1e9)`` s, far past any horizon.
* ``UniformVelocityCommand._update_command`` overwrites ``vel_command_b`` every step for heading envs
  (``heading_command`` / ``is_heading_env``), world-frame envs (``is_world_env``) and standing envs
  (``is_standing_env``, zeroed); ``_resample_command`` rewrites forward envs (``rel_forward_envs``) and
  ``reset`` may start an env moving (``init_velocity_prob``). The config turns all of these off
  (``heading_command=False``, ``ranges.heading=None``, every ``rel_*`` and ``init_velocity_prob`` 0), and
  ``pin_command`` also clears the per-env flags, because ``uniform(0, 1) <= 0.0`` can still draw True.
* After ``reset(seed=...)`` the command buffer ``vel_command_b`` (``.command``) is written with the
  pinned commands and the observations are recomputed, so the first action already sees them. The
  policy observes exactly the play-mode observation terms; none is altered.
* A fallen env is auto-reset by ``ManagerBasedRlEnv.step`` and its command resampled; it no longer
  counts toward any metric, and its command is re-pinned after every step so the buffer stays fixed.
  Every step ``rollout`` checks that each still-standing env's command was not changed by the step.

Metrics (per seed, then averaged over seeds; "standing" means ``fell_over`` has not fired yet, and the
step on which it fires is excluded because its post-step state is the reset state):

* ``vel_error_xy`` / ``vel_error_yaw``: mjlab's ``Metrics/twist/error_vel_xy`` / ``error_vel_yaw``
  definition, ``|cmd[:2] - root_link_lin_vel_b[:2]|`` and ``|cmd[2] - root_link_ang_vel_b[2]|``, as a
  plain mean over standing env-steps (mjlab instead divides an episode sum by the maximum steps per
  command).
* ``no_fall_fraction``: fraction of envs in which ``fell_over`` never fired within the horizon.
* ``mean_time_to_fall_s``: time of the step on which ``fell_over`` first fired, ``horizon_s`` if never.
* ``mean_tracking_reward``: the task's ``track_linear_velocity`` reward term evaluated with its own
  config (``exp(-err^2 / std^2)``, in (0, 1]; unweighted and not scaled by dt), mean over standing
  env-steps.

A metric averaged over standing env-steps is NaN for a seed in which no env stood for a single step.

Seeding: the env is built with ``seeds[0]``, which fixes the startup randomization (foot friction,
encoder bias, base CoM) for every checkpoint and seed alike; each seed then runs
``reset(seed=seed)``, which reseeds the RNGs before the reset randomization and the whole rollout.
"""

from __future__ import annotations

import csv
import math
import tempfile
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Final

import mjlab.tasks  # noqa: F401  # pyright: ignore[reportUnusedImport]  (registers the tasks)
import torch
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

from mprl.config import TASK_ID, RunSpec
from mprl.negative_space import check_finite, check_shape, expect, require
from mprl.train import built_runner

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from mjlab.rl import RslRlVecEnvWrapper
    from tensordict import TensorDict

    from mprl.checkpoints import CheckpointRef

EVAL_COMMANDS: Final[tuple[tuple[float, float, float], ...]] = (
    (0.5, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (-0.5, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.0, -0.5),
)
"""``(lin_x, lin_y, ang_z)``, all inside curriculum stage 0 (lin_vel_x +-1.0, ang_vel_z +-0.5)."""

_TWIST: Final = "twist"
_FELL_OVER: Final = "fell_over"
_TRACKING_TERM: Final = "track_linear_velocity"
_NEVER_S: Final = 1.0e9  # resampling time: no command is resampled within any horizon
_EVAL_SWEEP: Final = "eval"
_UNUSED_BUDGET_MIN: Final = 3.0  # RunSpec needs a budget >= its checkpoint interval; evaluation trains nothing

Policy = Callable[["TensorDict"], torch.Tensor]


@dataclass(frozen=True)
class EvalConfig:
    commands: tuple[tuple[float, float, float], ...] = EVAL_COMMANDS
    seeds: tuple[int, ...] = (0, 1)
    envs_per_command: int = 4
    horizon_s: float = 10.0

    def __post_init__(self) -> None:
        require(len(self.commands) > 0, "commands must not be empty")
        for command in self.commands:
            require(len(command) == 3, f"a command is (lin_x, lin_y, ang_z), got {command}")
            check_finite(torch.tensor(command), "command")
        require(len(self.seeds) > 0, "seeds must not be empty")
        require(self.envs_per_command >= 1, f"envs_per_command must be >= 1, got {self.envs_per_command}")
        check_finite(self.horizon_s, "horizon_s")
        require(0.0 < self.horizon_s < _NEVER_S, f"horizon_s must be in (0, {_NEVER_S}), got {self.horizon_s}")

    @property
    def num_envs(self) -> int:
        return len(self.commands) * self.envs_per_command


@dataclass(frozen=True)
class EvalResult:
    run_id: str
    iteration: int
    train_minutes: float
    vel_error_xy: float
    vel_error_yaw: float
    no_fall_fraction: float
    mean_time_to_fall_s: float
    mean_tracking_reward: float


@dataclass(frozen=True)
class SeedStats:
    """One seed's metrics (definitions in the module docstring)."""

    vel_error_xy: float
    vel_error_yaw: float
    no_fall_fraction: float
    mean_time_to_fall_s: float
    mean_tracking_reward: float


def command_tensor(cfg: EvalConfig, device: str | torch.device) -> torch.Tensor:
    """``(num_envs, 3)``: command ``i`` pinned on envs ``[i * envs_per_command, (i + 1) * envs_per_command)``."""
    commands = torch.tensor(cfg.commands, dtype=torch.float32, device=device)
    return commands.repeat_interleave(cfg.envs_per_command, dim=0)


def eval_env_cfg(cfg: EvalConfig) -> ManagerBasedRlEnvCfg:
    """The play env config with ``cfg.num_envs`` envs, seeded with ``seeds[0]``, every command override off."""
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seeds[0]
    require(cfg.horizon_s < env_cfg.episode_length_s, "the horizon must end before the play episode times out")
    require(_FELL_OVER in env_cfg.terminations, f"play env has no {_FELL_OVER} termination")
    for group in env_cfg.observations.values():
        require(not group.enable_corruption, "play observations must not be corrupted")
        for name, term in group.terms.items():
            # Recomputing observations after pinning is only side-effect free without these buffers.
            require(term.history_length == 0, f"observation {name} has history; re-check obs refresh")
            require(term.delay_max_lag == 0, f"observation {name} is delayed; re-check obs refresh")
    twist = env_cfg.commands[_TWIST]
    require(isinstance(twist, UniformVelocityCommandCfg), f"{_TWIST} is not a UniformVelocityCommandCfg")
    if isinstance(twist, UniformVelocityCommandCfg):
        twist.resampling_time_range = (_NEVER_S, _NEVER_S)
        twist.heading_command = False
        twist.ranges.heading = None
        twist.rel_standing_envs = 0.0
        twist.rel_heading_envs = 0.0
        twist.rel_world_envs = 0.0
        twist.rel_forward_envs = 0.0
        twist.init_velocity_prob = 0.0
    return env_cfg


@contextmanager
def loaded_policy(
    ref: CheckpointRef, cfg: EvalConfig, device: str, *, render_size: tuple[int, int] | None = None
) -> Generator[tuple[RslRlVecEnvWrapper, Policy]]:
    """The eval env plus ``ref``'s inference policy, loaded into a fresh runner; the env is closed on exit.

    With ``render_size=(height, width)`` the env is built with ``render_mode="rgb_array"`` at that viewer size.
    """
    require(ref.path.is_file(), f"checkpoint {ref.path} does not exist")
    spec = RunSpec(_EVAL_SWEEP, cfg.num_envs, cfg.seeds[0], _UNUSED_BUDGET_MIN)
    env_cfg = eval_env_cfg(cfg)
    if render_size is not None:
        require(min(render_size) >= 1, f"render_size must be positive, got {render_size}")
        env_cfg.viewer.height, env_cfg.viewer.width = render_size
    with (
        tempfile.TemporaryDirectory(prefix="mprl-eval-") as log_dir,
        built_runner(
            spec,
            device,
            "tensorboard",
            Path(log_dir),
            env_cfg=env_cfg,
            render_mode=None if render_size is None else "rgb_array",
        ) as runner,
    ):
        runner.load(str(ref.path), map_location=device)
        policy = runner.get_inference_policy(device)
        yield runner.env, policy  # pyright: ignore[reportReturnType]  rsl_rl types env as VecEnv


def twist_term(env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    term = env.command_manager.get_term(_TWIST)
    require(isinstance(term, UniformVelocityCommand), f"{_TWIST} is not a UniformVelocityCommand")
    return term  # pyright: ignore[reportReturnType]  narrowed by the require above


def pin_command(twist: UniformVelocityCommand, pinned: torch.Tensor) -> None:
    twist.vel_command_b[:] = pinned
    twist.vel_command_w[:] = pinned
    twist.is_heading_env[:] = False
    twist.is_standing_env[:] = False
    twist.is_world_env[:] = False
    twist.is_forward_env[:] = False


def horizon_steps(cfg: EvalConfig, step_dt: float) -> int:
    steps = round(cfg.horizon_s / step_dt)
    require(steps >= 1, f"horizon {cfg.horizon_s} s is shorter than one step of {step_dt} s")
    require(
        math.isclose(steps * step_dt, cfg.horizon_s, rel_tol=1e-6),
        f"horizon {cfg.horizon_s} s is not a whole number of {step_dt} s steps",
    )
    return steps


def rollout(env: RslRlVecEnvWrapper, policy: Policy, cfg: EvalConfig, seed: int) -> SeedStats:
    """Reset with ``seed``, pin the commands, step the horizon: one seed's metrics."""
    raw = env.unwrapped
    twist = twist_term(raw)
    pinned = command_tensor(cfg, raw.device)
    check_shape(twist.command, (raw.num_envs, 3), name="twist command")
    check_shape(pinned, (raw.num_envs, 3), name="pinned commands")
    steps = horizon_steps(cfg, raw.step_dt)
    robot = raw.scene["robot"]
    tracking = raw.reward_manager.get_term_cfg(_TRACKING_TERM)
    with torch.inference_mode():
        raw.reset(seed=seed)
        pin_command(twist, pinned)
        raw.observation_manager.compute(update_history=True)  # observations with the pinned command
        obs = env.get_observations()
        standing = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
        fall_step = torch.full((raw.num_envs,), steps, dtype=torch.long, device=raw.device)
        sum_xy = torch.zeros(raw.num_envs, device=raw.device)
        sum_yaw = torch.zeros_like(sum_xy)
        sum_reward = torch.zeros_like(sum_xy)
        counted = torch.zeros_like(sum_xy)
        for step in range(steps):
            obs, _, _, _ = env.step(policy(obs))
            fell = raw.termination_manager.get_term(_FELL_OVER)
            fall_step[standing & fell] = step + 1
            standing &= ~fell
            require(
                bool(torch.equal(twist.command[standing], pinned[standing])),
                f"step {step}: a standing env's command was changed by the env",
            )
            weight = standing.float()
            sum_xy += weight * torch.linalg.vector_norm(pinned[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=-1)
            sum_yaw += weight * torch.abs(pinned[:, 2] - robot.data.root_link_ang_vel_b[:, 2])
            sum_reward += weight * tracking.func(raw, **tracking.params)
            counted += weight
            pin_command(twist, pinned)  # fallen envs were reset and resampled; keep the buffer fixed
        require(bool(torch.all(twist.command_counter[standing] == 1)), "a standing env's command was resampled")
        total = counted.sum()
        return SeedStats(
            vel_error_xy=float(sum_xy.sum() / total),
            vel_error_yaw=float(sum_yaw.sum() / total),
            no_fall_fraction=float(standing.float().mean()),
            mean_time_to_fall_s=float(fall_step.float().mean() * raw.step_dt),
            mean_tracking_reward=float(sum_reward.sum() / total),
        )


def evaluate_checkpoint(run_id: str, ref: CheckpointRef, cfg: EvalConfig, device: str) -> EvalResult:
    """``ref`` evaluated once per seed in ``cfg.seeds``; each metric is the mean over seeds."""
    with loaded_policy(ref, cfg, device) as (env, policy):
        per_seed = [rollout(env, policy, cfg, seed) for seed in cfg.seeds]
    means = [sum(values) / len(values) for values in zip(*(astuple(s) for s in per_seed), strict=True)]
    result = EvalResult(run_id, ref.iteration, ref.train_seconds / 60.0, *means)
    require(0.0 <= result.no_fall_fraction <= 1.0, f"no_fall_fraction out of range: {result.no_fall_fraction}")
    require(
        0.0 < result.mean_time_to_fall_s <= cfg.horizon_s + 1e-6,
        f"mean_time_to_fall_s out of range: {result.mean_time_to_fall_s}",
    )
    return result


_CSV_TYPES: Final[dict[str, Callable[[str], object]]] = {"str": str, "int": int, "float": float}


def write_eval_csv(results: Sequence[EvalResult], path: Path) -> None:
    """One row per result, the ``EvalResult`` field names as header; floats in round-trip ``repr``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f.name for f in fields(EvalResult)])
        writer.writerows([repr(v) if isinstance(v, float) else str(v) for v in astuple(r)] for r in results)


def read_eval_csv(path: Path) -> list[EvalResult]:
    expected = [f.name for f in fields(EvalResult)]
    parsers = [expect(_CSV_TYPES.get(str(f.type)), f"no CSV parser for {f.type}") for f in fields(EvalResult)]
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        require(header == expected, f"{path}: header {header} != {expected}")
        rows = list(reader)
    results: list[EvalResult] = []
    for row in rows:
        require(len(row) == len(expected), f"{path}: row has {len(row)} fields, expected {len(expected)}")
        results.append(EvalResult(*(parse(value) for parse, value in zip(parsers, row, strict=True))))  # pyright: ignore[reportArgumentType]
    return results
