"""Unit tests for mprl.wandb_writer's pure parts (the offline runs are in tests/smoke/test_wandb_writer_offline.py)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, create_autospec

import pytest
import wandb
from wandb.errors import AuthenticationError, CommError, UsageError

from mprl.wandb_writer import CLOCK_METRIC, ResumableWandbWriter, logger_cfg


@pytest.mark.parametrize(
    ("resume_from_iteration", "run_keys"),
    [(None, {"id", "resume"}), (3, {"id", "resume"})],
    ids=["fresh", "resumed"],
)
def test_wandb_init_accepts_the_keywords_we_pass(
    resume_from_iteration: int | None, run_keys: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The keywords the writer really passes (captured from its call), checked against wandb.init's signature."""
    init = create_autospec(wandb.init, return_value=_run_double())  # an autospec double rejects a keyword wandb.init does not take
    monkeypatch.setattr(wandb, "init", init)

    ResumableWandbWriter("logs", "p", "rid", "grp", resume_from_iteration=resume_from_iteration, mode="disabled")

    passed = set(init.call_args.kwargs)
    common = {"project", "entity", "group", "job_type", "mode", "tags", "config"}
    assert passed == common | run_keys
    assert passed <= set(inspect.signature(wandb.init).parameters)


@pytest.mark.parametrize("resume_from_iteration", [None, 2534], ids=["fresh", "resumed"])
def test_init_continues_the_run_by_id_without_rewind(
    resume_from_iteration: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``resume_from`` (W&B Rewind) is a private preview: the server answers 400, which killed the 2026-10-29 long run."""
    init = create_autospec(wandb.init, return_value=_run_double())
    monkeypatch.setattr(wandb, "init", init)

    ResumableWandbWriter("logs", "p", "rid", "grp", resume_from_iteration=resume_from_iteration, mode="online")

    kwargs = init.call_args.kwargs
    assert (kwargs["id"], kwargs["resume"], "resume_from" in kwargs) == ("rid", "allow", False)


def test_offline_retry_tears_down_the_failed_online_run_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed online init keeps its run id in wandb's service; an offline retry with that id raises 'in use'."""
    manager = MagicMock()
    init = create_autospec(wandb.init, side_effect=[CommError("entity not found"), _run_double()])
    teardown = create_autospec(wandb.teardown)
    manager.attach_mock(init, "init")
    manager.attach_mock(teardown, "teardown")
    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.setattr(wandb, "teardown", teardown)

    ResumableWandbWriter("logs", "p", "rid", "grp", mode="online")

    assert [name for name, _, _ in manager.mock_calls] == ["init", "teardown", "init"]


def test_logger_cfg_shape() -> None:
    cfg = logger_cfg("p", "rid", "grp", 3, "offline")
    assert cfg == {
        "class_name": "mprl.wandb_writer:ResumableWandbWriter",
        "project": "p",
        "run_id": "rid",
        "group": "grp",
        "resume_from_iteration": 3,
        "mode": "offline",
        "tags": [],
    }


def _run_double() -> MagicMock:
    return create_autospec(wandb.Run, instance=True)


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """``wandb.init`` as an autospec double returning an autospecced ``wandb.Run``, which is returned."""
    double = _run_double()
    monkeypatch.setattr(wandb, "init", create_autospec(wandb.init, return_value=double))
    return double


def test_add_scalar_drops_the_tensorboard_wall_time_series(run: MagicMock) -> None:
    """rsl_rl writes ``Train/*/time`` at step=int(tot_time) for any non-W&B writer; W&B would drop later steps."""
    writer = ResumableWandbWriter("logs", "p", "rid", "grp", mode="disabled")

    writer.add_scalar("Train/mean_reward/time", 1.0, 30)
    writer.add_scalar("Train/mean_episode_length/time", 2.0, 30)

    run.log.assert_not_called()


def test_add_scalar_logs_other_tags_at_their_step(run: MagicMock) -> None:
    writer = ResumableWandbWriter("logs", "p", "rid", "grp", mode="disabled")

    writer.add_scalar("Train/mean_reward", 1.5, 3)

    run.log.assert_called_once_with({"Train/mean_reward": 1.5}, step=3)


def test_rsl_rl_log_sequence_reaches_wandb_with_monotonic_steps(run: MagicMock) -> None:
    """rsl_rl 5.4.2's per-iteration order (logger.py:221-233): metrics at step=it, then /time at step=30."""
    writer = ResumableWandbWriter("logs", "p", "rid", "grp", mode="disabled")

    for it in range(5):
        writer.add_scalar("Train/mean_reward", float(it), it)
        writer.add_scalar("Train/mean_reward/time", float(it), 30)

    steps = [c.kwargs["step"] for c in run.log.call_args_list]
    assert steps == [0, 1, 2, 3, 4]


def test_training_clock_is_defined_as_the_train_metrics_x_axis(run: MagicMock) -> None:
    ResumableWandbWriter("logs", "p", "rid", "grp", mode="disabled")

    calls = [(c.args, c.kwargs) for c in run.define_metric.call_args_list]
    assert calls == [
        ((CLOCK_METRIC,), {}),
        (("Train/*",), {"step_metric": CLOCK_METRIC}),
    ]
    assert CLOCK_METRIC == "Perf/train_clock_minutes"


@pytest.mark.parametrize(
    "error",
    [
        CommError("Network error (ConnectTimeout)"),
        AuthenticationError("API key invalid"),
        TimeoutError("init timed out"),
        wandb.errors.Error("wandb-core error tagged UNKNOWN"),
    ],
    ids=["network", "auth", "timeout", "unknown"],
)
def test_failed_online_init_falls_back_to_offline_for_that_run(
    error: Exception, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    init = create_autospec(wandb.init, side_effect=[error, _run_double()])
    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.setattr(wandb, "teardown", create_autospec(wandb.teardown))

    writer = ResumableWandbWriter("logs", "p", "rid", "grp", resume_from_iteration=7, mode="online")

    retry = init.call_args_list[1].kwargs
    assert (retry["mode"], retry["id"], retry["resume"]) == ("offline", "rid", "allow")
    assert writer.offline_fallback == f"{type(error).__name__}: {error}"
    assert "falling back to offline" in caplog.text


def test_successful_init_records_no_fallback(run: MagicMock) -> None:
    writer = ResumableWandbWriter("logs", "p", "rid", "grp", mode="online")

    assert writer.offline_fallback is None


def test_usage_error_at_init_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad argument is a programmer error, not the outside world: it must stop the run."""
    monkeypatch.setattr(wandb, "init", create_autospec(wandb.init, side_effect=UsageError("bad resume_from")))

    with pytest.raises(UsageError, match=r"^bad resume_from$"):
        ResumableWandbWriter("logs", "p", "rid", "grp", mode="online")


def test_failed_offline_init_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    init = create_autospec(wandb.init, side_effect=CommError("disk"))
    monkeypatch.setattr(wandb, "init", init)

    with pytest.raises(CommError, match=r"^disk$"):
        ResumableWandbWriter("logs", "p", "rid", "grp", mode="offline")

    assert init.call_count == 1


def test_logger_cfg_carries_tags() -> None:
    cfg = logger_cfg("p", "rid", "grp", None, "offline", tags=["n00064", "seed1"])

    assert cfg["tags"] == ["n00064", "seed1"]
