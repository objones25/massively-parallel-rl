"""Unit tests for the ``mprl`` CLI's control flow: fail fast before training, and when the pod is terminated.

Training, the sweep and the results builder are autospecced (their behavior is pinned by their own tests);
the terminator is an autospecced ``PodTerminator``. ``main`` runs in-process with an empty ``.env``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest

import mprl.cli as cli
import mprl.runpod as runpod_mod
from mprl.budget import CostEstimate, Measurements, save_measurements
from mprl.errors import BudgetExceeded, OwnPodUnknown
from mprl.results import build_results
from mprl.run_state import RunState
from mprl.runpod import PodTerminator
from mprl.sweep import run_sweep
from mprl.train import train_run

EST = CostEstimate((("sweep runs", 10.0),), 10.0, 17.9)
STATE = RunState("dbg-n00064-s1", "h" * 64, "dbg-n00064-s1-w", "complete", 180.0, 3.0, "a" * 40, {})


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    volume = tmp_path / "volume"
    monkeypatch.setenv("MPRL_VOLUME_ROOT", str(volume))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("HF_REPO_ID", "user/repo")
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    return volume


@pytest.fixture
def terminator(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    double = create_autospec(PodTerminator, instance=True)
    factory = create_autospec(cli.own_terminator, return_value=double)
    monkeypatch.setattr(cli, "own_terminator", factory)
    return double


def _main(tmp_path: Path, *argv: str) -> int:
    return cli.main(list(argv), env_file=tmp_path / "no.env")


def _measured(root: Path) -> None:
    save_measurements(root / "measurements.json", Measurements(0.0, 0.0, {64: True}, {64: 900.0}))


def _sweep_double(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> MagicMock:
    double = create_autospec(run_sweep, **kwargs)
    monkeypatch.setattr(cli, "run_sweep", double)
    return double


def test_sweep_budget_exceeded_unattended_terminates(
    tmp_path: Path, root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _measured(root)
    _sweep_double(monkeypatch, side_effect=BudgetExceeded("over the $40 cap"))

    assert _main(tmp_path, "sweep", "--name", "main") == 1

    terminator.terminate.assert_called_once_with()
    assert "over the $40 cap" in capsys.readouterr().err


def test_sweep_budget_exceeded_attended_leaves_pod(
    tmp_path: Path, root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _measured(root)
    _sweep_double(monkeypatch, side_effect=BudgetExceeded("over"))
    monkeypatch.setattr(cli, "_interactive", lambda: True)

    assert _main(tmp_path, "sweep", "--name", "main") == 1

    terminator.terminate.assert_not_called()


def test_sweep_without_measurements_terminates_before_training(
    tmp_path: Path, root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sweep = _sweep_double(monkeypatch)

    assert _main(tmp_path, "sweep", "--name", "main") == 1

    sweep.assert_not_called()
    terminator.terminate.assert_called_once_with()
    assert "mprl throughput" in capsys.readouterr().err


def test_sweep_no_terminate_never_builds_a_terminator(
    tmp_path: Path, root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _measured(root)
    sweep = _sweep_double(monkeypatch, side_effect=BudgetExceeded("over"))

    assert _main(tmp_path, "sweep", "--name", "main", "--no-terminate") == 1

    cli.own_terminator.assert_not_called()  # type: ignore[attr-defined]
    assert sweep.call_args.kwargs["terminator"] is None


def _confirming(cuts: tuple[str, ...]) -> Any:
    def side_effect(*args: Any, confirm: Any, **kwargs: Any) -> list[RunState]:
        return [] if not confirm(EST, cuts) else [STATE]

    return side_effect


def test_interactive_refusal_does_not_terminate(
    tmp_path: Path, root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _measured(root)
    _sweep_double(monkeypatch, side_effect=_confirming(("drop_16384",)))
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    prompts: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "n")

    assert _main(tmp_path, "sweep", "--name", "main") == 1

    terminator.terminate.assert_not_called()
    out = capsys.readouterr().out
    assert "drop_16384" in out
    assert EST.describe() in out
    assert prompts == ["Train with these cuts? [y/N] "]


def test_unattended_without_yes_refuses_and_terminates(
    tmp_path: Path, root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _measured(root)
    _sweep_double(monkeypatch, side_effect=_confirming(("drop_16384",)))

    assert _main(tmp_path, "sweep", "--name", "main") == 1

    terminator.terminate.assert_called_once_with()
    assert "--yes" in capsys.readouterr().out


def test_yes_confirms_after_printing_the_plan(
    tmp_path: Path, root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _measured(root)
    sweep = _sweep_double(monkeypatch, side_effect=_confirming(("two_seeds",)))

    assert _main(tmp_path, "sweep", "--name", "main", "--yes", "--reuse-pilot") == 0

    out = capsys.readouterr().out
    assert "two_seeds" in out
    assert EST.describe() in out
    assert sweep.call_args.kwargs["reuse_pilot"] is True
    assert sweep.call_args.kwargs["terminator"] is terminator
    terminator.terminate.assert_not_called()  # run_sweep itself terminates after training


def test_train_run_refuses_before_training_without_pod_id(
    tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_key")
    monkeypatch.setattr(runpod_mod, "own_pod_id", create_autospec(runpod_mod.own_pod_id, side_effect=OwnPodUnknown("no id")))
    train = create_autospec(train_run)
    monkeypatch.setattr(cli, "train_run", train)

    code = _main(tmp_path, "train-run", "--sweep", "dbg", "--num-envs", "64", "--seed", "0", "--minutes", "3")

    assert code == 1
    train.assert_not_called()
    assert "--no-terminate" in capsys.readouterr().err


def test_train_run_refuses_without_api_key(
    tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RUNPOD_POD_ID", "abc123")
    train = create_autospec(train_run)
    monkeypatch.setattr(cli, "train_run", train)

    assert _main(tmp_path, "train-run", "--sweep", "dbg", "--num-envs", "64", "--seed", "0", "--minutes", "3") == 1

    train.assert_not_called()
    err = capsys.readouterr().err
    assert "RUNPOD_API_KEY" in err
    assert "--no-terminate" in err


def test_train_run_terminates_after_the_run(
    tmp_path: Path, root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = create_autospec(train_run, return_value=STATE)
    monkeypatch.setattr(cli, "train_run", train)

    assert _main(tmp_path, "train-run", "--sweep", "dbg", "--num-envs", "64", "--seed", "1", "--minutes", "3") == 0

    spec = train.call_args.args[0]
    assert (spec.sweep, spec.num_envs, spec.seed, spec.budget_minutes) == ("dbg", 64, 1, 3.0)
    assert train.call_args.args[1] == root
    terminator.terminate.assert_called_once_with()


def test_results_leaves_the_pod_by_default(
    tmp_path: Path, root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = create_autospec(build_results, return_value=root / "out")
    monkeypatch.setattr(cli, "build_results", results)

    assert _main(tmp_path, "results", "--name", "main", "--primary-metric", "vel_error_xy") == 0

    assert results.call_args.kwargs["primary_metric"] == "vel_error_xy"
    cli.own_terminator.assert_not_called()  # type: ignore[attr-defined]


def test_results_terminate_flag_terminates(
    tmp_path: Path, root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "build_results", create_autospec(build_results, return_value=root / "out"))

    assert _main(tmp_path, "results", "--name", "main", "--primary-metric", "vel_error_xy", "--terminate") == 0

    terminator.terminate.assert_called_once_with()


def test_cli_metric_choices_match_the_curves_metrics() -> None:
    from mprl.curves import METRICS

    assert set(cli.METRIC_CHOICES) == set(METRICS)
