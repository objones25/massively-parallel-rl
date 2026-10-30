"""Smoke tests for TimedOnPolicyRunner against real mjlab / rsl_rl on CPU with 4 envs.

Tests that assert a cadence (how many iterations or checkpoints a budget yields) run the training clock on a
fake time source (support.Ticker, or env.step / alg.update / checkpoint writes advancing a counter), so they
hold on any machine; the training itself is always real. The training loop is the behavior here, so every run is
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
from support import ITERATION_S, LONG_S, STEPS_PER_ITERATION, TICK_S, Ticker, make_env_runner

from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.clock import TrainingClock, cuda_sync_for
from mprl.config import TASK_ID
from mprl.negative_space import expect
from mprl.timed_runner import LearnResult

pytestmark = pytest.mark.smoke

Build = Callable[..., tuple[ManagerBasedRlEnv, Any]]
HOOK_BUSY_S = 0.1  # off-clock work per hook call


class RecordingWriter(LogWriter):
    """A real rsl_rl LogWriter that records every scalar's (tag, step) and every model upload request."""

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self.saved_models: list[tuple[str, int]] = []
        self.scalars: list[tuple[str, int]] = []

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:  # noqa: ARG002  LogWriter API
        self.scalars.append((tag, global_step))

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
    """One real ``learn_for(budget 4 s, checkpoint every 1 s)`` on a Ticker clock, ``upload_model`` on, as data."""

    result: LearnResult
    elapsed_s: float
    saved_at: list[tuple[CheckpointRef, float]]  # each on_checkpoint ref with the clock at that moment
    stored: list[CheckpointRef]
    store_files: set[str]
    store_dir: Path
    uploads_requested: list[tuple[str, int]]
    iteration_ends: list[float]  # the clock at the end of every iteration (on_iteration)
    scalars: list[tuple[str, int]]  # every (tag, step) rsl_rl's logger wrote


BUDGET_S = 4.0
EVERY_S = 1.0


def _expected_saves(iteration_ends: list[float]) -> list[float]:
    """Clock values at which a checkpoint is due: the first iteration end past each whole EVERY_S mark, and the last."""
    saves: list[float] = []
    last = 0.0
    for t in iteration_ends:
        if math.floor(t / EVERY_S) > math.floor(last / EVERY_S) or t == iteration_ends[-1]:
            saves.append(t)
            last = t
    return saves


@pytest.fixture(scope="module")
def recorded_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[RecordedRun]:
    root = tmp_path_factory.mktemp("recorded")
    env, runner = make_env_runner(root / "logs", logger={"class_name": RecordingWriter})
    try:
        runner.cfg["upload_model"] = True  # the shipped default; learn_for must still not upload
        store = CheckpointStore(root / "checkpoints")
        clock = TrainingClock(now=Ticker(TICK_S), sync=cuda_sync_for("cpu"))  # 8 iterations of ITERATION_S
        saved_at: list[tuple[CheckpointRef, float]] = []
        ends: list[float] = []
        result = runner.learn_for(
            clock, BUDGET_S, EVERY_S, store, fresh_start=True,
            on_checkpoint=lambda ref: saved_at.append((ref, clock.elapsed_s)),
            on_iteration=lambda _it: ends.append(clock.elapsed_s),
        )  # fmt: skip
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
            ends,
            list(writer.scalars),
        )
    finally:
        env.close()


def test_learn_for_stops_when_budget_reached(recorded_run: RecordedRun) -> None:
    result = recorded_run.result

    ends = recorded_run.iteration_ends

    assert result.budget_exhausted
    assert result.iterations_run == len(ends) == round(BUDGET_S / ITERATION_S)
    assert ends[-1] == recorded_run.elapsed_s
    assert ends[-2] < BUDGET_S <= ends[-1]  # stops after exactly the iteration that crossed the budget
    assert result.overshoot_s == pytest.approx(recorded_run.elapsed_s - BUDGET_S)
    assert recorded_run.stored[-1] == result.final


def test_budget_stop_saves_the_final_checkpoint_off_cadence(build: Build, store: CheckpointStore) -> None:
    """With no checkpoint mark ever due, the iteration that crosses the budget still saves the one checkpoint."""
    _, runner = build()

    result = runner.learn_for(TrainingClock(now=Ticker(TICK_S)), 3.0, LONG_S, store, fresh_start=True)

    assert result.budget_exhausted
    assert store.list() == [result.final]
    assert result.final.iteration == runner.current_learning_iteration


def test_checkpoints_follow_training_clock(recorded_run: RecordedRun) -> None:
    saved_at = recorded_run.saved_at

    times = [t for _, t in saved_at]
    assert [ref for ref, _ in saved_at] == recorded_run.stored
    assert saved_at[-1][0] == recorded_run.result.final
    # Exactly the iteration ends that first cross each 1 s mark of the training clock, plus the last one.
    assert times == _expected_saves(recorded_run.iteration_ends)
    assert times[0] >= EVERY_S  # nothing saved before the first mark
    assert [ref.train_seconds for ref, _ in saved_at] == [float(round(t)) for t in times]


def test_store_holds_only_checkpoints_and_one_final_onnx(recorded_run: RecordedRun) -> None:
    checkpoints = {ref.path.name for ref in recorded_run.stored}

    assert len(checkpoints) == 4  # the 1, 2 and 3 s marks, and the budget at 4 s
    assert recorded_run.store_files == checkpoints | {"checkpoints.onnx"}
    assert recorded_run.result.final.path.parent / "checkpoints.onnx" == recorded_run.store_dir / "checkpoints.onnx"
    assert not any(path.endswith(".tmp") for path, _ in recorded_run.uploads_requested), recorded_run.uploads_requested


def test_io_is_off_the_clock(build: Build, store: CheckpointStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Checkpoint writes advance the time source too, yet the clock holds only the rollouts and updates."""
    _, runner = build()
    fake_now = _advance_on_step_and_update(runner)
    real_save_atomic = store.save_atomic
    saves: list[int] = []

    def slow_save_atomic(write: Callable[[Path], None], iteration: int, train_seconds: float) -> CheckpointRef:
        def slow_write(path: Path) -> None:
            fake_now[0] += SAVE_COST_S
            saves.append(iteration)
            write(path)

        return real_save_atomic(slow_write, iteration, train_seconds)

    monkeypatch.setattr(store, "save_atomic", slow_save_atomic)
    clock = TrainingClock(now=lambda: fake_now[0])
    runner.learn_for(clock, 2.5 * FAKE_ITERATION_S, FAKE_ITERATION_S, store, fresh_start=True)

    assert saves == [0, 1, 2]  # one per mark crossed, the last at the budget
    assert clock.elapsed_s == pytest.approx(3 * FAKE_ITERATION_S)  # on the clock it would add 3 * SAVE_COST_S


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
    first.learn_for(TrainingClock(now=Ticker(TICK_S)), 2.0, 1.0, store, fresh_start=True)
    loaded = expect(store.latest(), "first run saved no checkpoint")
    before = {ref.path.name for ref in store.list()}

    resumed_env, resumed = build(name="resumed")
    infos = resumed.load(str(loaded.path))
    restored_steps = resumed_env.common_step_counter
    clock = TrainingClock.from_state_dict(infos["clock"], now=Ticker(TICK_S))
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


def test_resume_in_fresh_runner_randomizes_episode_lengths(build: Build, store: CheckpointStore) -> None:
    # A checkpoint does not hold episode_length_buf, so a resumed env starts every episode at 0. Without
    # mjlab's init_at_random_ep_len on resume, all envs time out on the same iteration from then on.
    _, first = build(name="first")
    first.learn_for(TrainingClock(), LONG_S, LONG_S, store, fresh_start=True, max_iterations=1)
    loaded = expect(store.latest(), "first run saved no checkpoint")
    resumed_env, resumed = build(name="resumed")
    resumed.load(str(loaded.path))
    lengths_after_first_rollout: list[torch.Tensor] = []

    resumed.learn_for(
        TrainingClock(), LONG_S, LONG_S, store, fresh_start=False, max_iterations=1,
        on_before_update=lambda _: lengths_after_first_rollout.append(resumed_env.episode_length_buf.clone()),
    )

    # Unrandomized, no env could be past the first rollout's STEPS_PER_ITERATION steps.
    assert int(lengths_after_first_rollout[0].max()) > STEPS_PER_ITERATION


STEP_COST_S = 1.0  # fake seconds each env.step takes
UPDATE_COST_S = 1000.0  # fake seconds each alg.update takes
SAVE_COST_S = 100000.0  # fake seconds each checkpoint write takes (test_io_is_off_the_clock)
FAKE_ITERATION_S = STEPS_PER_ITERATION * STEP_COST_S + UPDATE_COST_S
SPLIT_ITERATIONS = 2


def _advance_on_step_and_update(runner: Any) -> list[float]:
    """Wrap the runner's env.step and alg.update so each advances a fake time; returns the [now] cell."""
    fake_now = [0.0]
    real_step, real_update = runner.env.step, runner.alg.update

    def step(actions: torch.Tensor) -> Any:
        fake_now[0] += STEP_COST_S
        return real_step(actions)

    def update() -> Any:
        fake_now[0] += UPDATE_COST_S
        return real_update()

    runner.env.step, runner.alg.update = step, update
    return fake_now


@pytest.fixture(scope="module")
def split_clock(tmp_path_factory: pytest.TempPathFactory) -> Iterator[list[tuple[str, float]]]:
    """One 2-iteration learn_for on a fake time source that only env.step and alg.update advance.

    Yields the clock at each hook: ``before_update`` (after the rollout measure) and ``iteration`` (after the
    update measure), so the rollout's and the update's share of the clock can be read separately and exactly.
    """
    root = tmp_path_factory.mktemp("split")
    env, runner = make_env_runner(root / "logs")
    try:
        fake_now = _advance_on_step_and_update(runner)
        clock = TrainingClock(now=lambda: fake_now[0])
        marks: list[tuple[str, float]] = []
        runner.learn_for(
            clock, 1e9, 1e9, CheckpointStore(root / "checkpoints"), fresh_start=True, max_iterations=SPLIT_ITERATIONS,
            on_before_update=lambda _it: marks.append(("before_update", clock.elapsed_s)),
            on_iteration=lambda _it: marks.append(("iteration", clock.elapsed_s)),
        )  # fmt: skip
        yield marks
    finally:
        env.close()


def test_clock_counts_every_rollout_env_step(split_clock: list[tuple[str, float]]) -> None:
    previous = [0.0, *[t for name, t in split_clock if name == "iteration"][:-1]]
    after_rollout = [t for name, t in split_clock if name == "before_update"]

    rollout_s = [b - a for a, b in zip(previous, after_rollout, strict=True)]

    assert rollout_s == [STEPS_PER_ITERATION * STEP_COST_S] * SPLIT_ITERATIONS


def test_clock_counts_every_ppo_update(split_clock: list[tuple[str, float]]) -> None:
    after_rollout = [t for name, t in split_clock if name == "before_update"]
    after_update = [t for name, t in split_clock if name == "iteration"]

    update_s = [b - a for a, b in zip(after_rollout, after_update, strict=True)]

    assert update_s == [UPDATE_COST_S] * SPLIT_ITERATIONS


RSL_RL_METRICS = (
    "Loss/value",
    "Loss/surrogate",
    "Loss/entropy",
    "Loss/learning_rate",
    "Policy/mean_std",
    "Perf/total_fps",
    "Perf/collection_time",
    "Perf/learning_time",
)


def test_rsl_rl_metrics_reach_the_writer_every_iteration(recorded_run: RecordedRun) -> None:
    """learn_for's logger.log writes rsl_rl's losses, learning rate, policy std and throughput at every step."""
    expected = {(tag, it) for tag in RSL_RL_METRICS for it in range(recorded_run.result.iterations_run)}

    assert expected - set(recorded_run.scalars) == set()
