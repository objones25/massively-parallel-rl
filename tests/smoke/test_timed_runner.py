"""Smoke tests for TimedOnPolicyRunner against real mjlab / rsl_rl on CPU with 4 envs.

On this Mac one G1 flat iteration at 4 envs takes about 0.5 s of training clock, so the budgets
below (2-4 s) run a handful of iterations each.
"""

from __future__ import annotations

import importlib.metadata
import itertools
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mjlab.tasks  # noqa: F401  (registers the tasks)
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_runner_cls
from rsl_rl.utils.log_writer import LogWriter

from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.clock import TrainingClock, cuda_sync_for
from mprl.config import TASK_ID
from mprl.negative_space import expect

pytestmark = pytest.mark.smoke

STEPS_PER_ITERATION = 24  # num_steps_per_env in the shipped G1 PPO config
LONG_S = 3600.0  # a budget / checkpoint interval no smoke test reaches

Build = Callable[..., tuple[ManagerBasedRlEnv, Any]]


class RecordingWriter(LogWriter):
    """A real rsl_rl LogWriter that records every model upload request."""

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self.saved_models: list[tuple[str, int]] = []

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        pass

    def save_model(self, model_path: str, it: int) -> None:
        self.saved_models.append((model_path, it))


def _busy_wait(seconds: float) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pass


def _actor_params(runner: Any) -> list[torch.Tensor]:
    return [p.detach().clone() for p in runner.alg.get_policy().parameters()]


def test_rsl_rl_version_is_the_one_learn_for_mirrors() -> None:
    assert importlib.metadata.version("rsl-rl-lib") == "5.4.2", (
        "TimedOnPolicyRunner.learn_for mirrors OnPolicyRunner.learn from rsl-rl-lib 5.4.2 call for call; "
        "re-check it against the new rsl_rl/runners/on_policy_runner.py before bumping this pin"
    )


def test_learn_for_stops_when_budget_reached(build: Build, store: CheckpointStore) -> None:
    _, runner = build()
    clock = TrainingClock(sync=cuda_sync_for("cpu"))

    result = runner.learn_for(clock, 3.0, LONG_S, store, fresh_start=True)

    assert result.budget_exhausted
    assert clock.elapsed_s >= 3.0
    assert result.overshoot_s == pytest.approx(clock.elapsed_s - 3.0)
    assert 0.0 <= result.overshoot_s < 2.0  # stops after the iteration that crossed the budget
    assert result.iterations_run >= 2
    assert store.list() == [result.final]


def test_checkpoints_follow_training_clock(build: Build, store: CheckpointStore) -> None:
    _, runner = build()
    clock = TrainingClock(sync=cuda_sync_for("cpu"))
    saved_at: list[tuple[CheckpointRef, float]] = []

    result = runner.learn_for(
        clock, 4.0, 1.0, store, fresh_start=True, on_checkpoint=lambda ref: saved_at.append((ref, clock.elapsed_s))
    )

    times = [t for _, t in saved_at]
    windows = [math.floor(t) for t in times]
    gaps = [b - a for a, b in itertools.pairwise(times)]
    assert [ref for ref, _ in saved_at] == store.list()
    assert saved_at[-1][0] == result.final
    assert len(saved_at) >= 3
    assert times[0] >= 1.0  # nothing saved before the first 1 s mark
    assert windows == sorted(set(windows))  # one checkpoint per crossed 1 s mark, never two
    assert all(0.0 < g < 2.0 for g in gaps), gaps  # about 1 s apart on the training clock
    assert times[-1] >= 4.0
    assert [ref.train_seconds for ref, _ in saved_at] == [float(round(t)) for t in times]


def test_io_is_off_the_clock(build: Build, store: CheckpointStore, monkeypatch: pytest.MonkeyPatch) -> None:
    _, runner = build()
    clock = TrainingClock(sync=cuda_sync_for("cpu"))
    real_save_atomic = store.save_atomic
    busy: list[float] = []

    def slow_save_atomic(write: Callable[[Path], None], iteration: int, train_seconds: float) -> CheckpointRef:
        def slow_write(path: Path) -> None:
            _busy_wait(1.0)
            busy.append(1.0)
            write(path)

        return real_save_atomic(slow_write, iteration, train_seconds)

    monkeypatch.setattr(store, "save_atomic", slow_save_atomic)
    wall_start = time.perf_counter()
    runner.learn_for(clock, 2.0, 1.0, store, fresh_start=True)
    wall_s = time.perf_counter() - wall_start

    assert len(busy) >= 2
    # On the clock it would hold the busy-waits too, i.e. exceed wall time minus the busy-waits.
    assert clock.elapsed_s <= wall_s - sum(busy)


def test_matches_rsl_rl_learn(build: Build, store: CheckpointStore) -> None:
    _, reference = build(load_runner_cls(TASK_ID), name="reference")
    reference.learn(num_learning_iterations=2, init_at_random_ep_len=True)
    expected = _actor_params(reference)

    _, timed = build(name="timed")
    result = timed.learn_for(TrainingClock(), LONG_S, LONG_S, store, fresh_start=True, max_iterations=2)

    assert result.iterations_run == 2
    assert not result.budget_exhausted
    assert timed.current_learning_iteration == reference.current_learning_iteration
    actual = _actor_params(timed)
    assert len(actual) == len(expected)
    assert all(torch.equal(a, e) for a, e in zip(actual, expected, strict=True))


def test_checkpoint_payload_has_clock_and_step_counter(build: Build, store: CheckpointStore) -> None:
    env, runner = build()
    clock = TrainingClock(sync=cuda_sync_for("cpu"))

    result = runner.learn_for(clock, LONG_S, LONG_S, store, fresh_start=True, max_iterations=3)

    payload = torch.load(result.final.path, weights_only=False)
    assert payload["infos"]["clock"] == {"elapsed_s": clock.elapsed_s}
    assert payload["infos"]["env_state"]["common_step_counter"] == 3 * STEPS_PER_ITERATION
    assert env.common_step_counter == 3 * STEPS_PER_ITERATION
    assert payload["iter"] == result.final.iteration == 2
    assert result.final.train_seconds == float(round(clock.elapsed_s))


def test_hooks_run_in_order_off_the_clock(build: Build, store: CheckpointStore) -> None:
    _, runner = build()
    clock = TrainingClock(sync=cuda_sync_for("cpu"))
    events: list[tuple[str, int, float]] = []

    def hook(name: str) -> Callable[[int], None]:
        def _record(it: int) -> None:
            events.append((name, it, clock.elapsed_s))
            _busy_wait(0.3)

        return _record

    wall_start = time.perf_counter()
    runner.learn_for(
        clock,
        LONG_S,
        LONG_S,
        store,
        fresh_start=True,
        max_iterations=2,
        on_before_update=hook("before_update"),
        on_iteration=hook("iteration"),
    )
    wall_s = time.perf_counter() - wall_start

    assert [(name, it) for name, it, _ in events] == [
        ("before_update", 0),
        ("iteration", 0),
        ("before_update", 1),
        ("iteration", 1),
    ]
    # Between consecutive hooks an update or a rollout ran, so the clock moved.
    assert all(b[2] > a[2] for a, b in itertools.pairwise(events))
    # On the clock it would hold the 4 x 0.3 s of hook time too.
    assert clock.elapsed_s <= wall_s - 4 * 0.3


def test_store_holds_only_checkpoints_and_one_final_onnx(build: Build, store: CheckpointStore, tmp_path: Path) -> None:
    _, runner = build(logger={"class_name": RecordingWriter})
    runner.cfg["upload_model"] = True  # the shipped default; learn_for must still not upload

    result = runner.learn_for(TrainingClock(), 3.0, 1.0, store, fresh_start=True)

    store_dir = tmp_path / "checkpoints"
    checkpoints = {ref.path.name for ref in store.list()}
    assert len(checkpoints) >= 2
    assert {p.name for p in store_dir.iterdir()} == checkpoints | {"checkpoints.onnx"}
    assert result.final.path.parent / "checkpoints.onnx" == store_dir / "checkpoints.onnx"
    writer = runner.logger.writer
    assert isinstance(writer, RecordingWriter)
    assert not any(path.endswith(".tmp") for path, _ in writer.saved_models), writer.saved_models


def test_resume_in_fresh_runner_continues_iterations_and_clock(build: Build, store: CheckpointStore) -> None:
    _, first = build(name="first")
    first.learn_for(TrainingClock(), 2.0, 1.0, store, fresh_start=True)
    loaded = expect(store.latest(), "first run saved no checkpoint")
    before = {ref.path.name for ref in store.list()}

    _, resumed = build(name="resumed")
    infos = resumed.load(str(loaded.path))
    clock = TrainingClock.from_state_dict(infos["clock"])
    restored_s = clock.elapsed_s
    result = resumed.learn_for(clock, restored_s + 1.5, 1.0, store, fresh_start=False)

    new_refs = [ref for ref in store.list() if ref.path.name not in before]
    iterations = [ref.iteration for ref in store.list()]
    assert restored_s >= 2.0
    assert new_refs
    assert min(ref.iteration for ref in new_refs) > loaded.iteration
    assert len(iterations) == len(set(iterations))
    assert resumed.current_learning_iteration == loaded.iteration + result.iterations_run
    assert result.final.iteration == resumed.current_learning_iteration
    assert clock.elapsed_s >= restored_s + 1.5
    assert torch.load(result.final.path, weights_only=False)["infos"]["clock"] == {"elapsed_s": clock.elapsed_s}
