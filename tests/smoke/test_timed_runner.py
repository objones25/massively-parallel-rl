"""Smoke tests for TimedOnPolicyRunner against real mjlab / rsl_rl on CPU with 4 envs.

On this Mac one G1 flat iteration at 4 envs takes about 0.5 s of training clock, so the budgets
below (2-4 s) run a handful of iterations each. The training loop is the behavior here, so every run is
real; tests that only read a run's outcome share one module-scoped run (``recorded_run``, ``hooked_run``).
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_runner_cls
from rsl_rl.utils.log_writer import LogWriter
from support import LONG_S, STEPS_PER_ITERATION, make_env_runner

from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.clock import TrainingClock, cuda_sync_for
from mprl.config import TASK_ID
from mprl.negative_space import expect
from mprl.timed_runner import LearnResult

pytestmark = pytest.mark.smoke

Build = Callable[..., tuple[ManagerBasedRlEnv, Any]]
HOOK_BUSY_S = 0.1  # off-clock work per hook call
SAVE_BUSY_S = 0.3  # off-clock work per checkpoint write


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


@dataclass(frozen=True)
class RecordedRun:
    """One real ``learn_for(budget 4 s, checkpoint every 1 s)`` with ``upload_model`` on, as plain data."""

    result: LearnResult
    elapsed_s: float
    saved_at: list[tuple[CheckpointRef, float]]  # each on_checkpoint ref with the clock at that moment
    stored: list[CheckpointRef]
    store_files: set[str]
    store_dir: Path
    uploads_requested: list[tuple[str, int]]


@pytest.fixture(scope="module")
def recorded_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[RecordedRun]:
    root = tmp_path_factory.mktemp("recorded")
    env, runner = make_env_runner(root / "logs", logger={"class_name": RecordingWriter})
    try:
        runner.cfg["upload_model"] = True  # the shipped default; learn_for must still not upload
        store = CheckpointStore(root / "checkpoints")
        clock = TrainingClock(sync=cuda_sync_for("cpu"))
        saved_at: list[tuple[CheckpointRef, float]] = []
        result = runner.learn_for(
            clock, 4.0, 1.0, store, fresh_start=True, on_checkpoint=lambda ref: saved_at.append((ref, clock.elapsed_s))
        )
        writer = runner.logger.writer
        assert isinstance(writer, RecordingWriter)
        yield RecordedRun(
            result,
            clock.elapsed_s,
            saved_at,
            store.list(),
            {p.name for p in (root / "checkpoints").iterdir()},
            root / "checkpoints",
            list(writer.saved_models),
        )
    finally:
        env.close()


def test_learn_for_stops_when_budget_reached(recorded_run: RecordedRun) -> None:
    result = recorded_run.result

    assert result.budget_exhausted
    assert recorded_run.elapsed_s >= 4.0
    assert result.overshoot_s == pytest.approx(recorded_run.elapsed_s - 4.0)
    assert 0.0 <= result.overshoot_s < 2.0  # stops after the iteration that crossed the budget
    assert result.iterations_run >= 2
    assert recorded_run.stored[-1] == result.final


def test_budget_stop_saves_the_final_checkpoint_off_cadence(build: Build, store: CheckpointStore) -> None:
    """With no checkpoint mark ever due, the iteration that crosses the budget still saves the one checkpoint."""
    _, runner = build()

    result = runner.learn_for(TrainingClock(sync=cuda_sync_for("cpu")), 3.0, LONG_S, store, fresh_start=True)

    assert result.budget_exhausted
    assert store.list() == [result.final]
    assert result.final.iteration == runner.current_learning_iteration


def test_checkpoints_follow_training_clock(recorded_run: RecordedRun) -> None:
    saved_at = recorded_run.saved_at

    times = [t for _, t in saved_at]
    windows = [math.floor(t) for t in times]
    gaps = [b - a for a, b in itertools.pairwise(times)]
    assert [ref for ref, _ in saved_at] == recorded_run.stored
    assert saved_at[-1][0] == recorded_run.result.final
    assert len(saved_at) >= 3
    assert times[0] >= 1.0  # nothing saved before the first 1 s mark
    assert windows == sorted(set(windows))  # one checkpoint per crossed 1 s mark, never two
    assert all(0.0 < g < 2.0 for g in gaps), gaps  # about 1 s apart on the training clock
    assert times[-1] >= 4.0
    assert [ref.train_seconds for ref, _ in saved_at] == [float(round(t)) for t in times]


def test_store_holds_only_checkpoints_and_one_final_onnx(recorded_run: RecordedRun) -> None:
    checkpoints = {ref.path.name for ref in recorded_run.stored}

    assert len(checkpoints) >= 2
    assert recorded_run.store_files == checkpoints | {"checkpoints.onnx"}
    assert recorded_run.result.final.path.parent / "checkpoints.onnx" == recorded_run.store_dir / "checkpoints.onnx"
    assert not any(path.endswith(".tmp") for path, _ in recorded_run.uploads_requested), recorded_run.uploads_requested


def test_io_is_off_the_clock(build: Build, store: CheckpointStore, monkeypatch: pytest.MonkeyPatch) -> None:
    _, runner = build()
    clock = TrainingClock(sync=cuda_sync_for("cpu"))
    real_save_atomic = store.save_atomic
    busy: list[float] = []

    def slow_save_atomic(write: Callable[[Path], None], iteration: int, train_seconds: float) -> CheckpointRef:
        def slow_write(path: Path) -> None:
            _busy_wait(SAVE_BUSY_S)
            busy.append(SAVE_BUSY_S)
            write(path)

        return real_save_atomic(slow_write, iteration, train_seconds)

    monkeypatch.setattr(store, "save_atomic", slow_save_atomic)
    wall_start = time.perf_counter()
    runner.learn_for(clock, 2.0, 1.0, store, fresh_start=True)
    wall_s = time.perf_counter() - wall_start

    assert len(busy) >= 2
    # On the clock it would hold the busy-waits too, i.e. exceed wall time minus the busy-waits.
    assert clock.elapsed_s <= wall_s - sum(busy)


@pytest.mark.usefixtures("torch_determinism")  # the assertion is bitwise parity of two runs
def test_matches_rsl_rl_learn(build: Build, store: CheckpointStore) -> None:
    _, reference = build(name="reference", runner_cls=load_runner_cls(TASK_ID))
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


@dataclass(frozen=True)
class HookedRun:
    """One real 3-iteration ``learn_for`` whose two hooks record (name, iteration, clock) and busy-wait."""

    result: LearnResult
    elapsed_s: float
    wall_s: float
    events: list[tuple[str, int, float]]
    env_step_counter: int
    payload: dict[str, Any]


HOOKED_ITERATIONS = 3


@pytest.fixture(scope="module")
def hooked_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[HookedRun]:
    root = tmp_path_factory.mktemp("hooked")
    env, runner = make_env_runner(root / "logs")
    try:
        clock = TrainingClock(sync=cuda_sync_for("cpu"))
        events: list[tuple[str, int, float]] = []

        def hook(name: str) -> Callable[[int], None]:
            def _record(it: int) -> None:
                events.append((name, it, clock.elapsed_s))
                _busy_wait(HOOK_BUSY_S)

            return _record

        wall_start = time.perf_counter()
        result = runner.learn_for(
            clock,
            LONG_S,
            LONG_S,
            CheckpointStore(root / "checkpoints"),
            fresh_start=True,
            max_iterations=HOOKED_ITERATIONS,
            on_before_update=hook("before_update"),
            on_iteration=hook("iteration"),
        )
        wall_s = time.perf_counter() - wall_start
        payload = torch.load(result.final.path, weights_only=False)
        yield HookedRun(result, clock.elapsed_s, wall_s, events, env.common_step_counter, payload)
    finally:
        env.close()


def test_checkpoint_payload_has_clock_and_step_counter(hooked_run: HookedRun) -> None:
    payload, final = hooked_run.payload, hooked_run.result.final

    assert payload["infos"]["clock"] == {"elapsed_s": hooked_run.elapsed_s}
    assert payload["infos"]["env_state"]["common_step_counter"] == HOOKED_ITERATIONS * STEPS_PER_ITERATION
    assert hooked_run.env_step_counter == HOOKED_ITERATIONS * STEPS_PER_ITERATION
    assert payload["iter"] == final.iteration == HOOKED_ITERATIONS - 1
    assert final.train_seconds == float(round(hooked_run.elapsed_s))


def test_hooks_run_in_order_off_the_clock(hooked_run: HookedRun) -> None:
    events = hooked_run.events

    assert [(name, it) for name, it, _ in events] == [
        (name, it) for it in range(HOOKED_ITERATIONS) for name in ("before_update", "iteration")
    ]
    # Between consecutive hooks an update or a rollout ran, so the clock moved.
    assert all(b[2] > a[2] for a, b in itertools.pairwise(events))
    # On the clock it would hold the hooks' busy-waits too.
    assert hooked_run.elapsed_s <= hooked_run.wall_s - len(events) * HOOK_BUSY_S


def test_resume_in_fresh_runner_continues_iterations_and_clock(build: Build, store: CheckpointStore) -> None:
    _, first = build(name="first")
    first.learn_for(TrainingClock(), 2.0, 1.0, store, fresh_start=True)
    loaded = expect(store.latest(), "first run saved no checkpoint")
    before = {ref.path.name for ref in store.list()}

    resumed_env, resumed = build(name="resumed")
    infos = resumed.load(str(loaded.path))
    restored_steps = resumed_env.common_step_counter
    clock = TrainingClock.from_state_dict(infos["clock"])
    restored_s = clock.elapsed_s
    result = resumed.learn_for(clock, restored_s + 1.5, 1.0, store, fresh_start=False)

    new_refs = [ref for ref in store.list() if ref.path.name not in before]
    iterations = [ref.iteration for ref in store.list()]
    assert restored_s >= 2.0
    assert restored_steps == (loaded.iteration + 1) * STEPS_PER_ITERATION  # env_state restored by load
    assert new_refs
    assert min(ref.iteration for ref in new_refs) > loaded.iteration
    assert len(iterations) == len(set(iterations))
    assert resumed.current_learning_iteration == loaded.iteration + result.iterations_run
    assert result.final.iteration == resumed.current_learning_iteration
    assert clock.elapsed_s >= restored_s + 1.5
    assert torch.load(result.final.path, weights_only=False)["infos"]["clock"] == {"elapsed_s": clock.elapsed_s}
