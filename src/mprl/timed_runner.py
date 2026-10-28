"""Time-budgeted PPO runner: the core of the equal-training-time comparison.

``TimedOnPolicyRunner.learn_for`` mirrors ``OnPolicyRunner.learn`` from **rsl-rl-lib 5.4.2**
(``rsl_rl/runners/on_policy_runner.py``) call for call, with three changes:

1. The rollout + ``compute_returns`` and the PPO ``update()`` run inside two separate
   ``TrainingClock.measure()`` blocks. Nothing else is on the clock.
2. It stops after the first iteration whose end puts the clock at or past ``budget_s`` (the
   overshoot is reported), or after ``max_iterations``, instead of after a fixed count.
3. Checkpoints are saved when the *training clock* crosses a multiple of ``checkpoint_every_s``
   (not every ``save_interval`` iterations) through ``CheckpointStore.save_atomic``, and the
   final iteration is always saved.

``tests/smoke/test_timed_runner.py::test_rsl_rl_version_is_the_one_learn_for_mirrors`` fails on any
other rsl-rl-lib version, so an upgrade forces a re-read of the upstream loop.

On the clock, per iteration, exactly what rsl_rl's ``learn`` times as collection + learning:

    measure #1 (under torch.inference_mode, as upstream):
        num_steps_per_env x [alg.act, env.step, check_nan (if cfg check_for_nan),
                             .to(device) of obs/rewards/dones, alg.process_env_step,
                             alg.intrinsic_rewards read (RND only), logger.process_env_step]
        alg.compute_returns(obs)
    measure #2:
        alg.update()

``logger.process_env_step`` is rsl_rl's per-step episode bookkeeping; it does a ``.cpu()`` sync per
step. It is part of the upstream rollout loop, so it stays there for parity; every run pays it
identically per env step, so the comparison stays fair.

Off the clock: the one-time setup before the loop (random episode lengths, ``get_observations``,
``train_mode``, writer init), ``on_before_update`` (between the two measures), ``logger.log``,
``on_iteration``, checkpoint I/O including ONNX export, ``on_checkpoint`` and writer shutdown.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner
from rsl_rl.utils import check_nan

from mprl.negative_space import check_finite, expect, require

if TYPE_CHECKING:
    from collections.abc import Callable

    from rsl_rl.env import VecEnv

    from mprl.checkpoints import CheckpointRef, CheckpointStore
    from mprl.clock import TrainingClock

_MAX_ITERATIONS = 10_000_000  # CheckpointStore names hold 7 iteration digits


def next_checkpoint_due(elapsed_s: float, every_s: float, last_saved_s: float) -> bool:
    """True once ``elapsed_s`` has reached the first multiple of ``every_s`` above ``last_saved_s``."""
    check_finite(elapsed_s, "elapsed_s")
    check_finite(every_s, "every_s")
    check_finite(last_saved_s, "last_saved_s")
    require(every_s > 0.0, f"every_s must be positive, got {every_s}")
    require(elapsed_s >= 0.0, f"elapsed_s must be non-negative, got {elapsed_s}")
    require(last_saved_s >= 0.0, f"last_saved_s must be non-negative, got {last_saved_s}")
    require(last_saved_s <= elapsed_s, f"last_saved_s {last_saved_s} is after elapsed_s {elapsed_s}")
    next_mark = (math.floor(last_saved_s / every_s) + 1) * every_s
    return elapsed_s >= next_mark


@dataclass(frozen=True)
class LearnResult:
    iterations_run: int
    final: CheckpointRef
    budget_exhausted: bool
    overshoot_s: float  # clock.elapsed_s - budget_s at return; negative when max_iterations stopped it


class TimedOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env: VecEnv, train_cfg: dict[str, Any], log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)
        self._clock: TrainingClock | None = None  # set by learn_for; save() records it

    def learn_for(
        self,
        clock: TrainingClock,
        budget_s: float,
        checkpoint_every_s: float,
        store: CheckpointStore,
        *,
        fresh_start: bool,
        max_iterations: int = _MAX_ITERATIONS,
        on_iteration: Callable[[int], None] | None = None,
        on_checkpoint: Callable[[CheckpointRef], None] | None = None,
        on_before_update: Callable[[int], None] | None = None,
    ) -> LearnResult:
        """Train until ``clock`` reaches ``budget_s`` (checked after each iteration) or ``max_iterations``.

        ``fresh_start`` randomizes initial episode lengths (upstream ``init_at_random_ep_len``); a
        resumed run must pass False so it does not re-randomize mid-training.
        """
        check_finite(budget_s, "budget_s")
        check_finite(checkpoint_every_s, "checkpoint_every_s")
        require(budget_s > 0.0, f"budget_s must be positive, got {budget_s}")
        require(checkpoint_every_s > 0.0, f"checkpoint_every_s must be positive, got {checkpoint_every_s}")
        require(
            0 < max_iterations <= _MAX_ITERATIONS,
            f"max_iterations must be in (0, {_MAX_ITERATIONS}], got {max_iterations}",
        )
        require(
            clock.elapsed_s < budget_s,
            f"budget already spent: clock at {clock.elapsed_s:.3f} s, budget {budget_s:.3f} s",
        )
        self._clock = clock

        # --- upstream setup (off the clock) ---
        if fresh_start:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + max_iterations
        last_saved_s = clock.elapsed_s  # a resumed run starts right after its checkpoint
        final: CheckpointRef | None = None
        iterations_run = 0
        budget_exhausted = False
        for it in range(start_it, total_it):
            # --- on the clock: rollout + compute_returns, exactly upstream's collection block ---
            # time.time() pairs feed logger.log's collect/learn times, as upstream; the budget
            # uses only the clock.
            start = time.time()
            with clock.measure(), torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    # Upstream bookkeeping (per-step .cpu()); charged equally to every run.
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)
            returns_time = time.time() - start

            # --- off the clock: hook between the measures (Task 5 snapshots weights here) ---
            if on_before_update is not None:
                on_before_update(it)

            # --- on the clock: PPO update ---
            start = time.time()
            with clock.measure():
                loss_dict = self.alg.update()
            stop = time.time()
            # Upstream's learn_time is compute_returns + update; the hook between them is excluded.
            learn_time = returns_time + (stop - start)
            self.current_learning_iteration = it
            iterations_run += 1

            # --- off the clock: logging, hooks, checkpoint I/O ---
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=(
                    expect(self.alg.rnd, "rnd_cfg is set but PPO has no RND module").weight
                    if self.cfg["algorithm"]["rnd_cfg"]
                    else None
                ),
            )
            if on_iteration is not None:
                on_iteration(it)

            budget_exhausted = clock.elapsed_s >= budget_s
            is_last = budget_exhausted or it == total_it - 1
            if is_last or next_checkpoint_due(clock.elapsed_s, checkpoint_every_s, last_saved_s):
                final = store.save_atomic(lambda path: self.save(str(path)), it, clock.elapsed_s)
                last_saved_s = clock.elapsed_s
                if on_checkpoint is not None:
                    on_checkpoint(final)
            if budget_exhausted:
                break

        if self.logger.writer is not None:
            self.logger.stop_logging_writer()
        final = expect(final, "learn_for ran no iteration, so it saved no final checkpoint")
        require(final.iteration == self.current_learning_iteration, "final checkpoint is not the last iteration")
        return LearnResult(
            iterations_run=iterations_run,
            final=final,
            budget_exhausted=budget_exhausted,
            overshoot_s=clock.elapsed_s - budget_s,
        )

    def save(self, path: str, infos: dict[str, Any] | None = None) -> None:
        """Upstream save plus ``infos["clock"]``, the training-clock state resume reads back."""
        clock = expect(self._clock, "save() outside learn_for: no training clock to record")
        super().save(path, {**(infos or {}), "clock": clock.state_dict()})
