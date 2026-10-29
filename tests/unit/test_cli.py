"""Unit tests for the ``mprl`` CLI's control flow: fail fast before training, and when the pod is terminated.

Training, the sweep and the results builder are autospecced (their behavior is pinned by their own tests);
the terminator is an autospecced ``PodTerminator``. ``main`` runs in-process with an empty ``.env``.
"""

from __future__ import annotations

import importlib.metadata
import os
import signal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest

import mprl.cli as cli
import mprl.runpod as runpod_mod
from mprl.budget import PRICE_PER_HOUR, CostEstimate, Measurements, save_measurements
from mprl.config import SweepSpec
from mprl.errors import BudgetExceeded, OwnPodUnknown, Terminated, TerminateFailed
from mprl.results import build_results
from mprl.run_state import RunState
from mprl.runpod import PodTerminator
from mprl.sweep import run_pilot, run_sweep
from mprl.train import train_run

EST = CostEstimate((("sweep runs", 10.0),), 10.0, 17.9)
STATE = RunState("dbg-n00064-s1", "h" * 64, "dbg-n00064-s1-w", "complete", 180.0, 3.0, "a" * 40, {})


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    volume = tmp_path / "volume"
    monkeypatch.setenv("MPRL_VOLUME_ROOT", str(volume))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("HF_REPO_ID", "user/repo")
    monkeypatch.setenv("MPRL_ENV_FILE", str(tmp_path / "no.env"))
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    return volume


@pytest.fixture
def terminator(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    double = create_autospec(PodTerminator, instance=True)
    factory = create_autospec(cli.own_terminator, return_value=double)
    monkeypatch.setattr(cli, "own_terminator", factory)
    return double


def _main(*argv: str) -> int:
    return cli.main(list(argv))


def _measured(root: Path) -> None:
    save_measurements(root / "measurements.json", Measurements(0.0, 0.0, {64: True}, {64: 900.0}))


def _sweep_double(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> MagicMock:
    double = create_autospec(run_sweep, **kwargs)
    monkeypatch.setattr(cli, "run_sweep", double)
    return double


def test_sweep_budget_exceeded_unattended_terminates(
    root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _measured(root)
    _sweep_double(monkeypatch, side_effect=BudgetExceeded("over the $40 cap"))

    assert _main("sweep", "--name", "main") == 1

    terminator.terminate.assert_called_once_with()
    assert "over the $40 cap" in capsys.readouterr().err


def test_sweep_budget_exceeded_attended_still_terminates(
    root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _measured(root)
    _sweep_double(monkeypatch, side_effect=BudgetExceeded("over"))
    monkeypatch.setattr(cli, "_interactive", lambda: True)

    assert _main("sweep", "--name", "main") == 1

    terminator.terminate.assert_called_once_with()


def test_sweep_corrupt_measurements_terminates(
    root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root.mkdir(parents=True)
    (root / "measurements.json").write_text("{not json")
    sweep = _sweep_double(monkeypatch)

    assert _main("sweep", "--name", "main") == 1

    sweep.assert_not_called()
    terminator.terminate.assert_called_once_with()
    assert f"mprl: MeasurementsCorrupt: {root / 'measurements.json'} is corrupt (" in capsys.readouterr().err


TRAINING_ARGV = {
    "train-run": ("train-run", "--sweep", "dbg", "--num-envs", "64", "--seed", "0", "--minutes", "3"),
    "pilot": ("pilot",),
    "sweep": ("sweep", "--name", "main"),
}


@pytest.mark.parametrize("command", list(TRAINING_ARGV))
def test_missing_hf_key_terminates(
    command: str,
    root: Path,
    terminator: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("HF_TOKEN")
    _measured(root)

    assert _main(*TRAINING_ARGV[command]) == 1

    terminator.terminate.assert_called_once_with()
    assert "HF_TOKEN" in capsys.readouterr().err


TRAINERS = {"train-run": ("train_run", train_run), "pilot": ("run_pilot", run_pilot), "sweep": ("run_sweep", run_sweep)}


@pytest.mark.parametrize("command", list(TRAINING_ARGV))
def test_training_error_terminates_then_propagates(
    command: str, root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    name, trainer = TRAINERS[command]
    monkeypatch.setattr(cli, name, create_autospec(trainer, side_effect=RuntimeError("boom")))
    _measured(root)

    with pytest.raises(RuntimeError, match=r"^boom$"):
        _main(*TRAINING_ARGV[command])

    terminator.terminate.assert_called_once_with()


def test_pilot_terminates_after_the_pilot(root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    pilot = create_autospec(run_pilot, return_value=STATE)
    monkeypatch.setattr(cli, "run_pilot", pilot)

    assert _main(*TRAINING_ARGV["pilot"]) == 0

    assert pilot.call_args.args[0].name == "main"
    assert pilot.call_args.kwargs["terminator"] is None  # the CLI's guard terminates, once
    terminator.terminate.assert_called_once_with()


def test_failed_termination_keeps_the_original_error(
    root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "train_run", create_autospec(train_run, side_effect=RuntimeError("boom")))
    terminator.terminate.side_effect = TerminateFailed("DELETE of pod p failed with HTTP 500")

    with pytest.raises(RuntimeError, match="boom") as info:
        _main(*TRAINING_ARGV["train-run"])

    assert info.value.__notes__ == [
        "pod termination also failed: TerminateFailed: DELETE of pod p failed with HTTP 500"
    ]


def test_main_prints_the_termination_failure_with_an_operating_error(
    root: Path,
    terminator: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("HF_REPO_ID")
    terminator.terminate.side_effect = TerminateFailed("DELETE of pod p failed with HTTP 500")

    assert _main(*TRAINING_ARGV["pilot"]) == 1

    err = capsys.readouterr().err
    assert "MissingSecret: HF_REPO_ID is not set" in err
    assert "pod termination also failed: TerminateFailed: DELETE of pod p failed with HTTP 500" in err


@pytest.mark.parametrize(("tty", "terminated"), [(True, False), (False, True)])
def test_keyboard_interrupt_terminates_only_without_a_terminal(
    tty: bool, terminated: bool, root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "train_run", create_autospec(train_run, side_effect=KeyboardInterrupt()))
    monkeypatch.setattr(cli, "_interactive", lambda: tty)

    with pytest.raises(KeyboardInterrupt):
        _main(*TRAINING_ARGV["train-run"])

    assert terminator.terminate.called is terminated


def test_sweep_without_measurements_terminates_before_training(
    root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sweep = _sweep_double(monkeypatch)

    assert _main("sweep", "--name", "main") == 1

    sweep.assert_not_called()
    terminator.terminate.assert_called_once_with()
    assert "mprl throughput" in capsys.readouterr().err


def test_sweep_no_terminate_never_builds_a_terminator(
    root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _measured(root)
    sweep = _sweep_double(monkeypatch, side_effect=BudgetExceeded("over"))

    assert _main("sweep", "--name", "main", "--no-terminate") == 1

    cli.own_terminator.assert_not_called()  # type: ignore[attr-defined]
    assert sweep.call_args.kwargs["terminator"] is None


def _confirming(cuts: tuple[str, ...]) -> Any:
    def side_effect(*args: Any, confirm: Any, **kwargs: Any) -> list[RunState]:
        return [] if not confirm(EST, cuts) else [STATE]

    return side_effect


def test_interactive_refusal_does_not_terminate(
    root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _measured(root)
    _sweep_double(monkeypatch, side_effect=_confirming(("drop_16384",)))
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    prompts: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "n")

    assert _main("sweep", "--name", "main") == 1

    terminator.terminate.assert_not_called()
    out = capsys.readouterr().out
    assert "drop_16384" in out
    assert EST.describe() in out
    assert prompts == ["Train with these cuts? [y/N] "]


def test_unattended_without_yes_refuses_and_terminates(
    root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _measured(root)
    _sweep_double(monkeypatch, side_effect=_confirming(("drop_16384",)))

    assert _main("sweep", "--name", "main") == 1

    terminator.terminate.assert_called_once_with()
    assert "--yes" in capsys.readouterr().out


def test_yes_confirms_after_printing_the_plan(
    root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _measured(root)
    sweep = _sweep_double(monkeypatch, side_effect=_confirming(("two_seeds",)))

    assert _main("sweep", "--name", "main", "--yes", "--reuse-pilot") == 0

    out = capsys.readouterr().out
    assert "two_seeds" in out
    assert EST.describe() in out
    assert sweep.call_args.kwargs["reuse_pilot"] is True
    assert sweep.call_args.kwargs["terminator"] is None  # the CLI's guard owns termination
    terminator.terminate.assert_called_once_with()


def test_train_run_refuses_before_training_without_pod_id(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_key")
    monkeypatch.setattr(
        runpod_mod, "own_pod_id", create_autospec(runpod_mod.own_pod_id, side_effect=OwnPodUnknown("no id"))
    )
    train = create_autospec(train_run)
    monkeypatch.setattr(cli, "train_run", train)

    code = _main("train-run", "--sweep", "dbg", "--num-envs", "64", "--seed", "0", "--minutes", "3")

    assert code == 1
    train.assert_not_called()
    assert "--no-terminate" in capsys.readouterr().err
    assert not root.exists()  # refused before anything touched the volume


def test_train_run_refuses_without_api_key(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RUNPOD_POD_ID", "abc123")
    train = create_autospec(train_run)
    monkeypatch.setattr(cli, "train_run", train)

    assert _main("train-run", "--sweep", "dbg", "--num-envs", "64", "--seed", "0", "--minutes", "3") == 1

    train.assert_not_called()
    err = capsys.readouterr().err
    assert "RUNPOD_API_KEY" in err
    assert "--no-terminate" in err


def test_train_run_terminates_after_the_run(root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    train = create_autospec(train_run, return_value=STATE)
    monkeypatch.setattr(cli, "train_run", train)

    assert _main("train-run", "--sweep", "dbg", "--num-envs", "64", "--seed", "1", "--minutes", "3") == 0

    spec = train.call_args.args[0]
    assert (spec.sweep, spec.num_envs, spec.seed, spec.budget_minutes) == ("dbg", 64, 1, 3.0)
    assert train.call_args.args[1] == root
    terminator.terminate.assert_called_once_with()


def test_results_leaves_the_pod_by_default(
    root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    results = create_autospec(build_results, return_value=root / "out")
    monkeypatch.setattr(cli, "build_results", results)

    assert _main("results", "--name", "main", "--primary-metric", "vel_error_xy") == 0

    assert results.call_args.kwargs["primary_metric"] == "vel_error_xy"
    assert capsys.readouterr().out.splitlines()[-1] == f"results in {root / 'out'}"
    cli.own_terminator.assert_not_called()  # type: ignore[attr-defined]


def test_results_terminate_flag_terminates(root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "build_results", create_autospec(build_results, return_value=root / "out"))

    assert _main("results", "--name", "main", "--primary-metric", "vel_error_xy", "--terminate") == 0

    terminator.terminate.assert_called_once_with()


def test_cli_metric_choices_match_the_curves_metrics() -> None:
    from mprl.curves import METRICS

    assert set(cli.METRIC_CHOICES) == set(METRICS)


SUBCOMMANDS = ("setup", "preflight", "throughput", "train-run", "pilot", "sweep", "results")


def test_help_lists_every_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match=r"^0$"):
        cli.main(["--help"])

    out = capsys.readouterr().out
    assert [name for name in SUBCOMMANDS if name not in out] == []
    assert tuple(cli.COMMANDS) == SUBCOMMANDS


def test_console_script_is_cli_main() -> None:
    (script,) = importlib.metadata.entry_points(group="console_scripts", name="mprl")

    assert script.value == "mprl.cli:main"


def test_preflight_exits_nonzero_on_a_failed_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MPRL_VOLUME_ROOT", str(tmp_path))
    monkeypatch.setenv("MPRL_ENV_FILE", str(tmp_path / "no.env"))

    code = cli.main(["preflight", "--cpu", "--only", "wandb"])

    assert code == 1
    assert "FAIL wandb: MissingSecret: WANDB_API_KEY is not set" in capsys.readouterr().out


def test_env_file_override_is_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text(f"MPRL_VOLUME_ROOT={tmp_path / 'from-env-file'}\n")
    monkeypatch.setenv("MPRL_ENV_FILE", str(env_file))
    # load_dotenv writes os.environ directly; registering the key with monkeypatch makes teardown remove it.
    monkeypatch.setenv("MPRL_VOLUME_ROOT", "unset-below")
    monkeypatch.delenv("MPRL_VOLUME_ROOT")

    code = cli.main(["preflight", "--cpu", "--only", "volume"])

    out = capsys.readouterr().out
    assert code == 0, out
    assert f"{tmp_path / 'from-env-file'} is writable" in out


# --- SIGTERM (kill, docker stop, RunPod stop) is a failure exit of every guarded command ---------------


def _sigterm_self(*_args: object, **_kwargs: object) -> None:
    os.kill(os.getpid(), signal.SIGTERM)


GUARDED_ARGV = {
    **TRAINING_ARGV,
    "results-terminate": ("results", "--name", "main", "--primary-metric", "vel_error_xy", "--terminate"),
}
GUARDED_BODIES = {**TRAINERS, "results-terminate": ("build_results", build_results)}


@pytest.mark.parametrize("command", list(GUARDED_ARGV))
def test_sigterm_mid_command_terminates_the_pod_and_exits_nonzero(
    command: str,
    root: Path,
    terminator: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sigterm_sentinel: list[int],
) -> None:
    name, body = GUARDED_BODIES[command]
    monkeypatch.setattr(cli, name, create_autospec(body, side_effect=_sigterm_self))
    _measured(root)

    assert _main(*GUARDED_ARGV[command]) == 1

    terminator.terminate.assert_called_once_with()
    assert "mprl: Terminated: received SIGTERM" in capsys.readouterr().err
    assert sigterm_sentinel == []


@pytest.mark.parametrize("command", list(GUARDED_ARGV))
def test_guarded_command_restores_the_previous_sigterm_handler(
    command: str, root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch, sigterm_sentinel: list[int]
) -> None:
    name, body = GUARDED_BODIES[command]
    monkeypatch.setattr(cli, name, create_autospec(body, side_effect=_sigterm_self))
    _measured(root)
    _main(*GUARDED_ARGV[command])

    os.kill(os.getpid(), signal.SIGTERM)

    assert sigterm_sentinel == [signal.SIGTERM]


def test_sigterm_at_a_terminal_still_terminates(
    root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch, sigterm_sentinel: list[int]
) -> None:
    """Unlike Ctrl-C, SIGTERM never comes from the user at the keyboard, so it does not keep the pod."""
    monkeypatch.setattr(cli, "train_run", create_autospec(train_run, side_effect=_sigterm_self))
    monkeypatch.setattr(cli, "_interactive", lambda: True)

    assert _main(*TRAINING_ARGV["train-run"]) == 1

    terminator.terminate.assert_called_once_with()


def test_second_sigterm_during_termination_is_ignored(
    root: Path,
    terminator: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sigterm_sentinel: list[int],
) -> None:
    """``docker stop`` may repeat SIGTERM; the cleanup it interrupts (state save, the DELETE) must still finish."""
    monkeypatch.setattr(cli, "train_run", create_autospec(train_run, side_effect=_sigterm_self))
    terminator.terminate.side_effect = _sigterm_self

    assert _main(*TRAINING_ARGV["train-run"]) == 1

    terminator.terminate.assert_called_once_with()
    err = capsys.readouterr().err
    assert err.count("mprl: Terminated: received SIGTERM") == 1
    assert "pod termination also failed" not in err
    assert sigterm_sentinel == []


def test_sigterm_after_a_successful_body_is_ignored_until_the_pod_is_terminated(
    root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch, sigterm_sentinel: list[int]
) -> None:
    monkeypatch.setattr(cli, "train_run", create_autospec(train_run, return_value=STATE))
    terminator.terminate.side_effect = _sigterm_self

    assert _main(*TRAINING_ARGV["train-run"]) == 0

    terminator.terminate.assert_called_once_with()
    assert sigterm_sentinel == []


def _sigterm_swallowed(result: object) -> Any:
    """A body whose library catches Exception around the frame SIGTERM landed in, then returns ``result``."""

    def body(*_args: object, **_kwargs: object) -> object:
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Terminated:
            pass  # as logging.Handler.handleError does around emit()
        return result

    return body


GUARDED_RESULTS: dict[str, object] = {"train-run": STATE, "pilot": STATE, "sweep": [STATE], "results-terminate": "out"}


@pytest.mark.parametrize("command", list(GUARDED_ARGV))
def test_swallowed_sigterm_is_still_a_failure_exit(
    command: str,
    root: Path,
    terminator: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sigterm_sentinel: list[int],
) -> None:
    name, body = GUARDED_BODIES[command]
    monkeypatch.setattr(cli, name, create_autospec(body, side_effect=_sigterm_swallowed(GUARDED_RESULTS[command])))
    _measured(root)

    assert _main(*GUARDED_ARGV[command]) == 1

    terminator.terminate.assert_called_once_with()
    assert "mprl: Terminated: received SIGTERM" in capsys.readouterr().err
    assert sigterm_sentinel == []


def test_sweep_options_reach_the_sweep(root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    _measured(root)
    sweep = _sweep_double(monkeypatch, return_value=[STATE])

    argv = ("--price-per-hour", "2.49", "--seeds", "0", "1", "--budget-minutes", "60", "--long-run-minutes", "90")
    assert _main("sweep", "--name", "main", *argv) == 0

    assert sweep.call_args.args[0] == SweepSpec("main", seeds=(0, 1), budget_minutes=60.0, long_run_minutes=90.0)
    assert sweep.call_args.kwargs["price_per_hour"] == pytest.approx(2.49)


def test_sweep_options_default_to_the_spec(root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    _measured(root)
    sweep = _sweep_double(monkeypatch, return_value=[STATE])

    assert _main("sweep", "--name", "main") == 0

    assert sweep.call_args.args[0] == SweepSpec("main")
    assert sweep.call_args.kwargs["price_per_hour"] == pytest.approx(PRICE_PER_HOUR)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("sweep", "--name", "main", "--price-per-hour", "0"), "--price-per-hour must be a positive finite number"),
        (("sweep", "--name", "main", "--seeds"), "--seeds must name at least one seed"),
        (("sweep", "--name", "main", "--budget-minutes", "1"), "--budget-minutes must be a finite number of at least 3"),
        (("pilot", "--price-per-hour", "-1"), "--price-per-hour must be a positive finite number"),
    ],
)
def test_bad_option_refuses_before_the_pod_guard(
    root: Path,
    terminator: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: tuple[str, ...],
    message: str,
) -> None:
    """A typo at the prompt is an operating error: exit 1, nothing trains, and the pod stays up for the retry."""
    _measured(root)
    sweep = _sweep_double(monkeypatch, return_value=[STATE])
    pilot = create_autospec(run_pilot, return_value=STATE)
    monkeypatch.setattr(cli, "run_pilot", pilot)

    assert _main(*argv) == 1

    assert f"mprl: InvalidInput: {message}" in capsys.readouterr().err
    sweep.assert_not_called()
    pilot.assert_not_called()
    terminator.terminate.assert_not_called()


def test_pilot_prints_its_cost_at_the_given_price(
    root: Path, terminator: MagicMock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "run_pilot", create_autospec(run_pilot, return_value=STATE))

    assert _main("pilot", "--price-per-hour", "2.00") == 0

    assert "pilot: 45 training minutes at 4096 envs, $1.50 at $2.00/h before off-clock overhead" in capsys.readouterr().out
