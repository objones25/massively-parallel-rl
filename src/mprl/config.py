"""Run and sweep configuration, batch arithmetic, and the config hash."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Final

from mprl.negative_space import require

TASK_ID: Final = "Mjlab-Velocity-Flat-Unitree-G1"
CHECKPOINT_EVERY_MINUTES: Final = 3.0  # the default checkpoint interval of a run and a sweep


@dataclass(frozen=True)
class RunSpec:
    """One training run. Only `num_envs` and `seed` vary within a sweep."""

    sweep: str
    num_envs: int
    seed: int
    budget_minutes: float
    checkpoint_every_minutes: float = CHECKPOINT_EVERY_MINUTES
    task_id: str = TASK_ID

    def __post_init__(self) -> None:
        require(self.num_envs > 0, f"num_envs must be positive, got {self.num_envs}")
        require(self.budget_minutes > 0, f"budget_minutes must be positive, got {self.budget_minutes}")
        require(
            0 < self.checkpoint_every_minutes <= self.budget_minutes,
            f"checkpoint_every_minutes must be in (0, budget_minutes], "
            f"got {self.checkpoint_every_minutes} with budget {self.budget_minutes}",
        )

    @property
    def run_id(self) -> str:
        return f"{self.sweep}-n{self.num_envs:05d}-s{self.seed}"

    def config_hash(self) -> str:
        """sha256 of the canonical JSON of every field except `budget_minutes`,
        so raising a budget is not a config change."""
        fields = asdict(self)
        del fields["budget_minutes"]
        canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class SweepSpec:
    name: str
    env_counts: tuple[int, ...] = (64, 256, 1024, 4096, 16384)
    seeds: tuple[int, ...] = (0, 1, 2)
    budget_minutes: float = 45.0
    checkpoint_every_minutes: float = CHECKPOINT_EVERY_MINUTES
    eval_every_minutes: float = 3.0
    long_run_envs: int = 4096
    long_run_minutes: float = 180.0

    def runs(self) -> list[RunSpec]:
        """Every (count, seed) run, count-major and seed-minor."""
        return [
            RunSpec(
                self.name,
                num_envs,
                seed,
                self.budget_minutes,
                self.checkpoint_every_minutes,
            )
            for num_envs in self.env_counts
            for seed in self.seeds
        ]

    def long_run(self) -> RunSpec:
        return RunSpec(
            f"{self.name}-long",
            self.long_run_envs,
            0,
            self.long_run_minutes,
            self.checkpoint_every_minutes,
        )


@dataclass(frozen=True)
class BatchArithmetic:
    num_envs: int
    steps_per_env: int
    num_mini_batches: int

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.steps_per_env

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_mini_batches


def batch_arithmetic(num_envs: int, steps_per_env: int = 24, num_mini_batches: int = 4) -> BatchArithmetic:
    """Batch = envs x steps; it must split evenly into minibatches."""
    if num_envs <= 0 or steps_per_env <= 0 or num_mini_batches <= 0:
        raise ValueError(
            f"num_envs, steps_per_env, num_mini_batches must be positive, "
            f"got {num_envs}, {steps_per_env}, {num_mini_batches}"
        )
    batch = num_envs * steps_per_env
    if batch % num_mini_batches != 0:
        raise ValueError(
            f"batch size {batch} ({num_envs} envs x {steps_per_env} steps) is not "
            f"divisible by {num_mini_batches} minibatches"
        )
    return BatchArithmetic(num_envs, steps_per_env, num_mini_batches)
