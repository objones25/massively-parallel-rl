"""Unit tests for mprl.sweep with an autospecced ``train_run`` (its real behavior is pinned by
tests/smoke/test_train_run.py) and an autospecced ``PodTerminator``."""

from __future__ import annotations

import json
import weakref
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest

import mprl.sweep as sweep_mod
import mprl.train
from mprl.budget import CostEstimate, Measurements, estimate
from mprl.config import RunSpec, SweepSpec
from mprl.errors import EnvDoesNotFit, InvalidInput, MeasurementMissing, TerminateFailed
from mprl.negative_space import CheckFailed
from mprl.paths import RunPaths, sweep_results_dir
from mprl.run_state import RunState, save_state
from mprl.runpod import PodTerminator
from mprl.sweep import configured_sweep, pilot_spec, run_sweep
from mprl.train import TrainDeps

SHA = "a" * 40
DEPS = TrainDeps(publisher=None, wandb_mode="disabled", device="cpu")
SMALL = SweepSpec("main", env_counts=(64, 4096), seeds=(0, 1))


def _m(io: float = 0.0) -> Measurements:
    return Measurements(0.0, io, {64: True, 4096: True}, {64: 900.0, 4096: 250_000.0})


def _full_m(io: float = 0.0) -> Measurements:
    counts = SweepSpec("main").env_counts
    return Measurements(0.0, io, dict.fromkeys(counts, True), {n: 100.0 * n for n in counts})


def _state(spec: RunSpec, *, train_seconds: float | None = None, sha: str = SHA, **overrides: Any) -> RunState:
    seconds = spec.budget_minutes * 60.0 if train_seconds is None else train_seconds
    fields: dict[str, Any] = {
        "run_id": spec.run_id,
        "config_hash": spec.config_hash(),
        "wandb_run_id": f"{spec.run_id}-w",
        "status": "complete",
        "train_seconds": seconds,
        "budget_minutes": spec.budget_minutes,
        "git_sha": sha,
        "uploads": {},
    } | overrides
    return RunState(**fields)


@pytest.fixture
def train_run(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    double = create_autospec(mprl.train.train_run)
    double.side_effect = lambda spec, *_: _state(spec)
    monkeypatch.setattr(sweep_mod, "train_run", double)
    monkeypatch.setattr(sweep_mod, "git_sha", lambda: SHA)
    return double


@pytest.fixture
def terminator() -> MagicMock:
    return create_autospec(PodTerminator, instance=True)


def _trained(train_run: MagicMock) -> list[str]:
    return [call.args[0].run_id for call in train_run.call_args_list]


def _yes(est: CostEstimate, cuts: tuple[str, ...]) -> bool:
    return True


def test_sweep_runs_every_run_then_the_long_run_and_terminates(
    tmp_path: Path, train_run: MagicMock, terminator: MagicMock
) -> None:
    states = run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=terminator, reuse_pilot=False)

    expected = [r.run_id for r in SMALL.runs()] + [SMALL.long_run().run_id]
    assert _trained(train_run) == expected
    assert [s.run_id for s in states] == expected
    assert all(call.args[1:] == (tmp_path, DEPS) for call in train_run.call_args_list)
    terminator.terminate.assert_called_once_with()


def test_sweep_excludes_counts_that_do_not_fit(tmp_path: Path, train_run: MagicMock) -> None:
    sweep = SweepSpec("main")
    m = Measurements(0.0, 0.0, {64: True, 256: True, 1024: True, 4096: True, 16384: False}, {64: 900.0, 4096: 2.5e5})

    run_sweep(sweep, tmp_path, DEPS, m, confirm=_yes, terminator=None, reuse_pilot=False)

    trained = [call.args[0] for call in train_run.call_args_list]
    assert {s.num_envs for s in trained} == {64, 256, 1024, 4096}
    assert len(trained) == 4 * 3 + 1
    manifest = json.loads((sweep_results_dir(tmp_path, "main") / "sweep_manifest.json").read_text())
    reason = manifest["excluded"]["16384"]
    assert "16384" in reason
    assert "out of memory" in reason
    assert "4096: 250000" in reason  # the measured steps/s that do fit are cited
    assert manifest["planned"]["env_counts"] == [64, 256, 1024, 4096]
    assert manifest["git_sha"] == SHA


def test_sweep_refuses_when_64_does_not_fit(tmp_path: Path, train_run: MagicMock) -> None:
    with pytest.raises(EnvDoesNotFit, match="64"):
        run_sweep(
            SMALL,
            tmp_path,
            DEPS,
            Measurements(0.0, 0.0, {64: False, 4096: True}, {4096: 250_000.0}),
            confirm=_yes,
            terminator=None,
            reuse_pilot=False,
        )
    train_run.assert_not_called()


def test_sweep_requires_confirmation_when_cuts_applied(
    tmp_path: Path, train_run: MagicMock, terminator: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[tuple[CostEstimate, tuple[str, ...]]] = []

    def refuse(est: CostEstimate, cuts: tuple[str, ...]) -> bool:
        seen.append((est, cuts))
        return False

    states = run_sweep(
        SweepSpec("main"), tmp_path, DEPS, _full_m(io=0.4), confirm=refuse, terminator=terminator, reuse_pilot=False
    )

    assert states == []
    train_run.assert_not_called()
    terminator.terminate.assert_not_called()
    assert len(seen) == 1
    assert seen[0][1] == ("eval_every_6min", "long_run_half")
    assert seen[0][0].total_usd <= 40.0
    assert "$" in capsys.readouterr().out  # the estimate is printed


def test_sweep_does_not_ask_without_cuts(tmp_path: Path, train_run: MagicMock) -> None:
    def never(est: CostEstimate, cuts: tuple[str, ...]) -> bool:
        raise AssertionError("confirm called although no cut applies")

    states = run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=never, terminator=None, reuse_pilot=False)

    assert len(states) == len(SMALL.runs()) + 1


def test_sweep_terminates_pod_on_crash(tmp_path: Path, train_run: MagicMock, terminator: MagicMock) -> None:
    train_run.side_effect = [_state(SMALL.runs()[0]), RuntimeError("simulator exploded")]

    with pytest.raises(RuntimeError, match="simulator exploded"):
        run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=terminator, reuse_pilot=False)

    assert train_run.call_count == 2
    terminator.terminate.assert_called_once_with()


def _write_pilot(root: Path, state: RunState) -> Path:
    paths = RunPaths(root, pilot_spec(SMALL))
    save_state(paths.state_file, state)
    return paths.run_dir


def _reused_id() -> str:
    return RunSpec("main", 4096, 0, 45.0).run_id


@pytest.mark.parametrize(
    ("label", "pilot_state", "reused"),
    [
        ("match", _state(pilot_spec(SMALL)), True),
        ("other code", _state(pilot_spec(SMALL), sha="b" * 40), False),
        ("other config", _state(replace(pilot_spec(SMALL), checkpoint_every_minutes=1.5)), False),
        ("unfinished", _state(pilot_spec(SMALL), status="running"), False),
        ("short", _state(pilot_spec(SMALL), train_seconds=44.0 * 60), False),
    ],
)
def test_pilot_reused_only_when_hash_and_sha_match(
    tmp_path: Path, train_run: MagicMock, label: str, pilot_state: RunState, reused: bool
) -> None:
    _write_pilot(tmp_path, pilot_state)

    run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=None, reuse_pilot=True)

    assert (_reused_id() not in _trained(train_run)) is reused, label
    assert (sweep_results_dir(tmp_path, "main") / "pilot_reuse.json").is_file() is reused, label
    assert _trained(train_run)[-1] == SMALL.long_run().run_id  # the long run resumes the pilot either way


def test_pilot_reuse_manifest_names_source_and_cap(tmp_path: Path, train_run: MagicMock) -> None:
    pilot_dir = _write_pilot(tmp_path, _state(pilot_spec(SMALL)))

    run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=None, reuse_pilot=True)

    manifest = json.loads((sweep_results_dir(tmp_path, "main") / "pilot_reuse.json").read_text())
    assert manifest == {
        _reused_id(): {
            "source_run_dir": str(pilot_dir),
            "max_minutes": pytest.approx(45.0),
            "git_sha": SHA,
            "config_hash": pilot_spec(SMALL).config_hash(),
        }
    }


def test_pilot_not_reused_without_flag(tmp_path: Path, train_run: MagicMock) -> None:
    _write_pilot(tmp_path, _state(pilot_spec(SMALL)))

    run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=None, reuse_pilot=False)

    assert _reused_id() in _trained(train_run)
    assert not (sweep_results_dir(tmp_path, "main") / "pilot_reuse.json").exists()


def test_pilot_not_reused_when_head_unknown(
    tmp_path: Path, train_run: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_pilot(tmp_path, _state(pilot_spec(SMALL), sha="unknown"))
    monkeypatch.setattr(sweep_mod, "git_sha", lambda: "unknown")

    run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=None, reuse_pilot=True)

    assert _reused_id() in _trained(train_run)


def test_pilot_not_reused_when_sweep_started_on_other_code(tmp_path: Path, train_run: MagicMock) -> None:
    _write_pilot(tmp_path, _state(pilot_spec(SMALL)))
    manifest = sweep_results_dir(tmp_path, "main") / "sweep_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"git_sha": "c" * 40}))

    run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=None, reuse_pilot=True)

    assert _reused_id() in _trained(train_run)


def test_recorded_pilot_reuse_survives_long_run_resuming(tmp_path: Path, train_run: MagicMock) -> None:
    """After a crash during the long run the pilot is 'running' past 45 min; the recorded reuse still holds."""
    _write_pilot(tmp_path, _state(pilot_spec(SMALL)))
    run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=None, reuse_pilot=True)
    _write_pilot(
        tmp_path, _state(replace(pilot_spec(SMALL), budget_minutes=180.0), status="running", train_seconds=6000.0)
    )
    train_run.reset_mock()

    run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=None, reuse_pilot=True)

    assert _reused_id() not in _trained(train_run)


def test_pilot_spec_is_the_first_45_minutes_of_the_long_run() -> None:
    spec = pilot_spec(SMALL)

    assert spec.run_id == SMALL.long_run().run_id
    assert spec.budget_minutes == pytest.approx(45.0)
    assert spec.config_hash() == SMALL.long_run().config_hash()
    assert replace(spec, sweep="main").config_hash() == RunSpec("main", 4096, 0, 45.0).config_hash()


def test_run_pilot_trains_the_pilot_spec_then_terminates(
    tmp_path: Path, train_run: MagicMock, terminator: MagicMock
) -> None:
    state = sweep_mod.run_pilot(SMALL, tmp_path, DEPS, terminator=terminator)

    train_run.assert_called_once_with(pilot_spec(SMALL), tmp_path, DEPS)
    assert state.run_id == SMALL.long_run().run_id
    terminator.terminate.assert_called_once_with()


def test_stale_pilot_reuse_is_removed_without_the_flag(tmp_path: Path, train_run: MagicMock) -> None:
    _write_pilot(tmp_path, _state(pilot_spec(SMALL)))
    run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=None, reuse_pilot=True)
    reuse_file = sweep_results_dir(tmp_path, "main") / "pilot_reuse.json"
    assert reuse_file.is_file()
    train_run.reset_mock()

    run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=None, reuse_pilot=False)

    assert _reused_id() in _trained(train_run)
    assert not reuse_file.exists()


def test_recorded_pilot_reuse_survives_a_code_change_after_the_sweep_started(
    tmp_path: Path, train_run: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2026-10-29: the long run failed after the sweep; its fix changes HEAD, and the rerun must not retrain
    the reused run (as with every other complete run, finished work is not redone on new code). The long run's
    resume also restamps the pilot's state with the new sha."""
    _write_pilot(tmp_path, _state(pilot_spec(SMALL)))
    run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=None, reuse_pilot=True)
    reuse_file = sweep_results_dir(tmp_path, "main") / "pilot_reuse.json"
    recorded = reuse_file.read_text()
    head = "d" * 40
    monkeypatch.setattr(sweep_mod, "git_sha", lambda: head)
    _write_pilot(
        tmp_path,
        _state(replace(pilot_spec(SMALL), budget_minutes=90.0), status="failed", train_seconds=2700.0, sha=head),
    )
    train_run.reset_mock()

    run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=None, reuse_pilot=True)

    assert _trained(train_run) == [r.run_id for r in SMALL.runs() if r.run_id != _reused_id()] + [
        SMALL.long_run().run_id
    ]
    assert reuse_file.read_text() == recorded


def test_sweep_refuses_unmeasured_counts(tmp_path: Path, train_run: MagicMock, terminator: MagicMock) -> None:
    sweep = SweepSpec("main", env_counts=(64, 256, 4096, 16384))

    with pytest.raises(MeasurementMissing, match=r"\[256, 16384\]"):
        run_sweep(sweep, tmp_path, DEPS, _m(), confirm=_yes, terminator=terminator, reuse_pilot=False)

    train_run.assert_not_called()
    terminator.terminate.assert_not_called()


def test_keyboard_interrupt_during_the_sweep_still_terminates(
    tmp_path: Path, train_run: MagicMock, terminator: MagicMock
) -> None:
    """The sweep runs unattended, so Ctrl-C (or a KeyboardInterrupt from anywhere) does not keep the pod."""
    train_run.side_effect = KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=terminator, reuse_pilot=False)

    terminator.terminate.assert_called_once_with()


def test_sweep_refuses_a_long_run_count_outside_the_planned_counts(
    tmp_path: Path, train_run: MagicMock, terminator: MagicMock
) -> None:
    """Every sweep run uses a planned count; the long run's 4096 is not among (64,), so nothing trains."""
    sweep = SweepSpec("main", env_counts=(64,), seeds=(0,))

    with pytest.raises(CheckFailed, match=r"^a run uses an excluded or cut env count$"):
        run_sweep(sweep, tmp_path, DEPS, _m(), confirm=_yes, terminator=terminator, reuse_pilot=False)

    train_run.assert_not_called()


@pytest.mark.parametrize(("reuse", "done_minutes"), [(True, 45.0), (False, 0.0)])
def test_sweep_estimate_subtracts_the_reused_pilot_from_the_long_run(
    tmp_path: Path, train_run: MagicMock, reuse: bool, done_minutes: float
) -> None:
    """A reused pilot is both the sweep's n=4096 s=0 run and the long run's first 45 min: counted once."""
    _write_pilot(tmp_path, _state(pilot_spec(SMALL)))
    m = Measurements(36.0, 0.5, {64: True, 4096: True}, {64: 900.0, 4096: 250_000.0})

    run_sweep(SMALL, tmp_path, DEPS, m, confirm=_yes, terminator=None, reuse_pilot=reuse)

    manifest = json.loads((sweep_results_dir(tmp_path, "main") / "sweep_manifest.json").read_text())
    expected = estimate(SMALL, m, long_run_done_minutes=done_minutes)
    assert manifest["estimate"]["hours"] == pytest.approx(expected.total_hours)
    assert manifest["estimate"]["usd"] == pytest.approx(expected.total_usd)


def test_failed_termination_does_not_mask_the_training_error(
    tmp_path: Path, train_run: MagicMock, terminator: MagicMock
) -> None:
    train_run.side_effect = RuntimeError("boom")
    terminator.terminate.side_effect = TerminateFailed("DELETE failed with HTTP 500")

    with pytest.raises(RuntimeError, match="boom") as info:
        run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=terminator, reuse_pilot=False)

    assert any("DELETE failed with HTTP 500" in note for note in info.value.__notes__)


def test_failed_pilot_termination_does_not_mask_the_training_error(
    tmp_path: Path, train_run: MagicMock, terminator: MagicMock
) -> None:
    train_run.side_effect = RuntimeError("boom")
    terminator.terminate.side_effect = TerminateFailed("DELETE failed with HTTP 500")

    with pytest.raises(RuntimeError, match="boom") as info:
        sweep_mod.run_pilot(SMALL, tmp_path, DEPS, terminator=terminator)

    assert any("TerminateFailed: DELETE failed with HTTP 500" in note for note in info.value.__notes__)


def test_configured_sweep_applies_the_overrides() -> None:
    sweep = configured_sweep("main", seeds=(0, 2), budget_minutes=60.0, long_run_minutes=90.0)

    assert sweep == SweepSpec("main", seeds=(0, 2), budget_minutes=60.0, long_run_minutes=90.0)


def test_configured_sweep_defaults_are_the_sweep_spec() -> None:
    defaults = SweepSpec("main")

    sweep = configured_sweep(
        "main", seeds=defaults.seeds, budget_minutes=defaults.budget_minutes, long_run_minutes=defaults.long_run_minutes
    )

    assert sweep == defaults


@pytest.mark.parametrize(
    ("seeds", "budget", "long_run", "match"),
    [
        ((), 45.0, 180.0, r"^--seeds must name at least one seed"),
        ((0, 0), 45.0, 180.0, r"^--seeds must not repeat a seed"),
        ((-1,), 45.0, 180.0, r"^--seeds must be non-negative"),
        ((0,), 2.0, 180.0, r"^--budget-minutes must be a finite number of at least 3"),
        ((0,), float("nan"), 180.0, r"^--budget-minutes must be a finite number of at least 3"),
        ((0,), 45.0, 30.0, r"^--long-run-minutes must be a finite number of at least 45"),
        ((0,), 45.0, float("inf"), r"^--long-run-minutes must be a finite number of at least 45"),
    ],
)
def test_configured_sweep_rejects_bad_overrides(
    seeds: tuple[int, ...], budget: float, long_run: float, match: str
) -> None:
    with pytest.raises(InvalidInput, match=match):
        configured_sweep("main", seeds=seeds, budget_minutes=budget, long_run_minutes=long_run)


def test_pilot_reuse_unaffected_by_price_seed_and_long_run_options(tmp_path: Path, train_run: MagicMock) -> None:
    """Only run selection and cost change; the n=4096 s=0 run's hash (budget excluded) still equals the pilot's."""
    adapted = replace(SMALL, seeds=(0, 2), long_run_minutes=90.0)
    _write_pilot(tmp_path, _state(pilot_spec(SMALL)))

    run_sweep(adapted, tmp_path, DEPS, _m(), confirm=_yes, terminator=None, reuse_pilot=True, price_per_hour=2.49)

    assert _reused_id() not in _trained(train_run)
    assert (sweep_results_dir(tmp_path, "main") / "pilot_reuse.json").is_file()
    assert _trained(train_run) == [
        RunSpec("main", 64, 0, 45.0).run_id,
        RunSpec("main", 64, 2, 45.0).run_id,
        RunSpec("main", 4096, 2, 45.0).run_id,
        SMALL.long_run().run_id,
    ]


def test_pilot_not_reused_when_the_sweep_budget_exceeds_the_pilot(tmp_path: Path, train_run: MagicMock) -> None:
    _write_pilot(tmp_path, _state(pilot_spec(SMALL)))

    run_sweep(replace(SMALL, budget_minutes=60.0), tmp_path, DEPS, _m(), confirm=_yes, terminator=None, reuse_pilot=True)

    assert RunSpec("main", 4096, 0, 60.0).run_id in _trained(train_run)
    assert not (sweep_results_dir(tmp_path, "main") / "pilot_reuse.json").exists()


def test_sweep_manifest_records_the_price(tmp_path: Path, train_run: MagicMock) -> None:
    run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=None, reuse_pilot=False, price_per_hour=2.5)

    manifest = json.loads((sweep_results_dir(tmp_path, "main") / "sweep_manifest.json").read_text())
    assert manifest["price_per_hour"] == pytest.approx(2.5)
    assert manifest["estimate"]["usd"] == pytest.approx(manifest["estimate"]["hours"] * 2.5)


class _Runner:
    """Stands in for a finished run's runner: a reference cycle (mjlab managers point back at the env)."""

    def __init__(self) -> None:
        self.cycle = self


def test_device_memory_is_released_after_every_run_in_order(
    tmp_path: Path, train_run: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, str]] = []
    train_run.side_effect = lambda spec, *_: events.append(("train", spec.run_id)) or _state(spec)
    release = create_autospec(mprl.train.release_device_memory, side_effect=lambda d: events.append(("release", d)))
    monkeypatch.setattr(sweep_mod, "release_device_memory", release)

    run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=None, reuse_pilot=False)

    trained = [run_id for kind, run_id in events if kind == "train"]
    assert len(trained) == len(SMALL.runs()) + 1
    assert events == [e for run_id in trained for e in (("train", run_id), ("release", "cpu"))]


def test_finished_runner_has_no_live_reference_when_memory_is_released(
    tmp_path: Path, train_run: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs: list[weakref.ref[_Runner]] = []
    alive_after_collect: list[bool] = []

    def train(spec: RunSpec, *_args: object) -> RunState:
        refs.append(weakref.ref(_Runner()))  # only gc can free it, as the real runner/env cycle
        return _state(spec)

    def release(device: str) -> None:
        mprl.train.release_device_memory(device)
        alive_after_collect.append(refs[-1]() is not None)

    train_run.side_effect = train
    monkeypatch.setattr(sweep_mod, "release_device_memory", release)

    run_sweep(SMALL, tmp_path, DEPS, _m(), confirm=_yes, terminator=None, reuse_pilot=False)

    assert alive_after_collect == [False] * (len(SMALL.runs()) + 1)


def test_pilot_releases_device_memory_after_training(
    tmp_path: Path, train_run: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = create_autospec(mprl.train.release_device_memory)
    monkeypatch.setattr(sweep_mod, "release_device_memory", release)

    sweep_mod.run_pilot(SMALL, tmp_path, DEPS, terminator=None)

    release.assert_called_once_with("cpu")
