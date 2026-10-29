"""Unit tests for train_run's decisions on hand-built run directories (no env, no training).

``_train`` (env build + training, pinned end to end by tests/smoke/test_train_run.py) is replaced by an
autospec double returning a hand-written checkpoint, so everything around it runs for real: the skip
and upload-retry paths, the W&B id choice, state.json, the upload staging and the pod DELETE ordering.
"""

from __future__ import annotations

import os
import signal
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest
import wandb
from huggingface_hub import HfApi
from support import runpod_terminator
from wandb.errors import CommError

import mprl.publish as publish_mod
import mprl.train as train_mod
from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.clock import TrainingClock
from mprl.config import RunSpec
from mprl.errors import ConfigMismatch, Terminated
from mprl.hub import HubPublisher
from mprl.negative_space import expect
from mprl.paths import RunPaths
from mprl.publish import PodGuard, finish_and_terminate, sigterm_raises
from mprl.run_state import RunState, Status, load_state, save_state
from mprl.train import TrainDeps, failed_after_final_checkpoint, train_run, wandb_run_id_for
from mprl.wandb_writer import CLOCK_METRIC, ResumableWandbWriter

EVERY_MIN = 1.0 / 60.0
SPEC = RunSpec("main", 4, 0, 2.0 / 60.0, EVERY_MIN)
DEPS = TrainDeps(publisher=None, wandb_mode="disabled", device="cpu")
REPO = "someone/mprl-test"
PATH_IN_REPO = f"runs/main/{SPEC.run_id}"
UPLOADED = {"model_it0000003_s000002.pt", "checkpoints.onnx", "env.yaml", "agent.yaml", "state.json"}


def _ref(iteration: int, seconds: float = 0.0) -> CheckpointRef:
    return CheckpointRef(iteration, seconds, Path(f"model_it{iteration:07d}_s{round(seconds):06d}.pt"))


def _state(status: Status, *, wandb_run_id: str = "main-n00004-s0-w0", uploads: dict[str, str] | None = None) -> RunState:
    return RunState(
        SPEC.run_id, SPEC.config_hash(), wandb_run_id, status, 2.0, SPEC.budget_minutes, "sha", uploads or {}
    )


def _finished_run(root: Path) -> CheckpointRef:
    """A run directory as training leaves it: one final checkpoint, its ONNX and the params."""
    paths = RunPaths(root, SPEC)
    final = CheckpointStore(paths.checkpoints_dir).save_atomic(lambda p: p.write_bytes(b"weights"), 3, 2.0)
    (paths.checkpoints_dir / "checkpoints.onnx").write_bytes(b"onnx")
    (paths.run_dir / "params").mkdir(parents=True)
    for name in ("env.yaml", "agent.yaml"):
        (paths.run_dir / "params" / name).write_text("{}\n")
    return final


def _logger_cfg(_spec: RunSpec, wandb_run_id: str, *_args: object, **_kwargs: object) -> dict[str, Any]:
    return {"run_id": wandb_run_id}


def _trains(root: Path) -> Callable[..., tuple[CheckpointRef, float, dict[str, str]]]:
    """A ``_train`` side effect: training happened, leaving the finished run's files; 2 s on the clock."""

    def fake(*_args: object) -> tuple[CheckpointRef, float, dict[str, str]]:
        return _finished_run(root), 2.0, {}

    return fake


@pytest.fixture
def train(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """``_train`` as an autospec double; tests set its return value or assert it was never called."""
    double = create_autospec(train_mod._train)
    monkeypatch.setattr(train_mod, "_train", double)
    monkeypatch.setattr(train_mod, "git_sha", lambda: "c" * 40)
    # The real logger config reads the mjlab task's PPO config (pinned in the smoke tier); keep its run id.
    logger_cfg = create_autospec(train_mod.wandb_logger_cfg, side_effect=_logger_cfg)
    monkeypatch.setattr(train_mod, "wandb_logger_cfg", logger_cfg)
    return double


def _publisher(upload: Callable[..., Any] | None = None) -> tuple[HubPublisher, MagicMock]:
    api = create_autospec(HfApi, instance=True)
    if upload is None:
        api.upload_folder.return_value.commit_url = f"https://huggingface.co/{REPO}/commit/abc"
    else:
        api.upload_folder.side_effect = upload
    return HubPublisher(api, REPO, max_attempts=2, sleep=lambda _s: None), api


# --- pure decisions ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stored_id", "latest", "expected"),
    [
        (None, None, "main-n00004-s0-minted"),  # fresh run
        ("crashed0", None, "main-n00004-s0-minted"),  # crashed before its first checkpoint: steps restart at 0
        (None, _ref(3), "main-n00004-s0-minted"),  # a checkpoint without state.json is not a resumable run
        ("stored0", _ref(3), "stored0"),  # resume: the W&B run continues
    ],
)
def test_wandb_run_id_reused_only_with_state_and_checkpoint(
    stored_id: str | None, latest: CheckpointRef | None, expected: str
) -> None:
    state = None if stored_id is None else _state("running", wandb_run_id=stored_id)

    assert wandb_run_id_for(state, latest, SPEC.run_id, mint=lambda: "minted") == expected


@pytest.mark.parametrize(
    ("saved", "elapsed_s", "current_iteration", "expected"),
    [
        ((), 5.0, 3, False),  # nothing saved: a training failure
        ((_ref(2),), 1.0, 2, False),  # below the budget: the final checkpoint does not exist yet
        ((_ref(1), _ref(3)), 2.0, 3, True),  # budget reached and the last save is the current iteration
        ((_ref(1), _ref(3)), 2.5, 4, False),  # an iteration ran after the last save
    ],
)
def test_failed_after_final_checkpoint(
    saved: tuple[CheckpointRef, ...], elapsed_s: float, current_iteration: int, expected: bool
) -> None:
    assert failed_after_final_checkpoint(list(saved), elapsed_s, 2.0, current_iteration) is expected


# --- train_run around a doubled _train -----------------------------------------------------------


def test_resume_refuses_config_mismatch(tmp_path: Path, train: MagicMock) -> None:
    changed = RunSpec("main", 4, 0, SPEC.budget_minutes, 2 * EVERY_MIN)  # same run id, different config
    save_state(RunPaths(tmp_path, SPEC).state_file, _state("running"))

    with pytest.raises(ConfigMismatch, match=r"^run main-n00004-s0: stored config hash"):
        train_run(changed, tmp_path, DEPS)

    train.assert_not_called()


def test_completed_run_is_skipped(tmp_path: Path, train: MagicMock) -> None:
    _finished_run(tmp_path)
    paths = RunPaths(tmp_path, SPEC)
    save_state(paths.state_file, _state("complete"))
    before = CheckpointStore(paths.checkpoints_dir).list()

    state = train_run(SPEC, tmp_path, DEPS)

    train.assert_not_called()
    assert state == _state("complete")
    assert load_state(paths.state_file) == state
    assert CheckpointStore(paths.checkpoints_dir).list() == before


def test_run_without_checkpoint_mints_new_wandb_id(tmp_path: Path, train: MagicMock) -> None:
    paths = RunPaths(tmp_path, SPEC)
    save_state(paths.state_file, _state("running", wandb_run_id="crashed0"))
    train.side_effect = _trains(tmp_path)

    state = train_run(SPEC, tmp_path, DEPS)

    assert train.call_args.args[4] is None  # trained from scratch
    assert state.status == "complete"
    assert state.wandb_run_id.startswith(f"{SPEC.run_id}-")
    assert state.wandb_run_id != "crashed0"
    assert train.call_args.args[3]["run_id"] == state.wandb_run_id


def test_failed_upload_is_retried_without_training(tmp_path: Path, train: MagicMock) -> None:
    _finished_run(tmp_path)
    paths = RunPaths(tmp_path, SPEC)
    first = _state("complete", uploads={PATH_IN_REPO: "UploadFailed: 503"})
    save_state(paths.state_file, first)
    before = CheckpointStore(paths.checkpoints_dir).list()
    publisher, api = _publisher()

    second = train_run(SPEC, tmp_path, TrainDeps(publisher=publisher, wandb_mode="disabled", device="cpu"))

    train.assert_not_called()
    assert api.upload_folder.call_count == 1
    assert api.upload_folder.call_args.kwargs["path_in_repo"] == PATH_IN_REPO
    assert second.uploads[PATH_IN_REPO] == "ok"
    assert load_state(paths.state_file) == second
    assert second.train_seconds == first.train_seconds
    assert CheckpointStore(paths.checkpoints_dir).list() == before
    assert not (paths.run_dir / "upload").exists()


@pytest.mark.parametrize(
    ("status", "recorded", "attempts"),
    [(503, "UploadFailed", 2), (403, "HfHubHTTPError", 1)],  # 5xx is retried (max_attempts=2), 403 is not
)
def test_terminate_runs_after_failed_upload(
    tmp_path: Path, train: MagicMock, hub_http_error: Callable[[int], Exception], status: int, recorded: str, attempts: int
) -> None:
    paths = RunPaths(tmp_path, SPEC)
    train.side_effect = _trains(tmp_path)
    events: list[tuple[str, object]] = []

    def failing_upload(**kwargs: Any) -> None:
        events.append(("upload", sorted(p.name for p in Path(kwargs["folder_path"]).rglob("*") if p.is_file())))
        raise hub_http_error(status)

    publisher, api = _publisher(failing_upload)

    state = train_run(SPEC, tmp_path, TrainDeps(publisher=publisher, wandb_mode="disabled", device="cpu"))
    finish_and_terminate([state], runpod_terminator(paths, events))

    upload_events = [files for name, files in events if name == "upload"]
    assert upload_events == [sorted(UPLOADED)] * attempts
    assert [name for name, _ in events][-1:] == ["DELETE"]
    assert [name for name, _ in events].count("DELETE") == 1
    assert state.uploads[PATH_IN_REPO].startswith(f"{recorded}:")
    assert str(status) in state.uploads[PATH_IN_REPO]
    assert events[-1][1] == state.uploads  # the failure was durable on disk before the DELETE
    assert api.upload_folder.call_args.kwargs["path_in_repo"] == PATH_IN_REPO
    assert not (paths.run_dir / "upload").exists()  # staging removed after the failed attempt


def test_terminate_runs_after_staging_error(tmp_path: Path, train: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """A full volume while staging the upload is recorded in state.json, and the DELETE still follows."""
    paths = RunPaths(tmp_path, SPEC)
    train.side_effect = _trains(tmp_path)
    events: list[tuple[str, object]] = []
    publisher, api = _publisher()

    def no_space(*args: object, **kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(publish_mod.shutil, "copy2", no_space)
    state = train_run(SPEC, tmp_path, TrainDeps(publisher=publisher, wandb_mode="disabled", device="cpu"))
    finish_and_terminate([state], runpod_terminator(paths, events))

    recorded = state.uploads[PATH_IN_REPO]
    assert recorded == "OSError: [Errno 28] No space left on device"
    assert [name for name, _ in events] == ["DELETE"]
    assert events[0][1] == state.uploads  # durable on disk before the DELETE
    assert api.upload_folder.call_count == 0
    assert not (paths.run_dir / "upload").exists()


def test_training_error_marks_the_run_failed_and_propagates(tmp_path: Path, train: MagicMock) -> None:
    train.side_effect = RuntimeError("env exploded")

    with pytest.raises(RuntimeError, match=r"^env exploded$"):
        train_run(SPEC, tmp_path, DEPS)

    stored = expect(load_state(RunPaths(tmp_path, SPEC).state_file), "no state.json after the failure")
    assert stored.status == "failed"


def test_crash_mid_training_is_durable_before_the_pod_guard_terminates(tmp_path: Path, train: MagicMock) -> None:
    """A crash after a checkpoint: state.json says failed and the checkpoint is on disk when the DELETE goes out."""
    paths = RunPaths(tmp_path, SPEC)
    events: list[tuple[str, object]] = []

    def crash_after_a_checkpoint(*_args: object) -> None:
        CheckpointStore(paths.checkpoints_dir).save_atomic(lambda p: p.write_bytes(b"weights"), 1, 1.0)
        raise RuntimeError("env exploded")

    def durable(state: RunState) -> object:
        return state.status, [ref.path.name for ref in CheckpointStore(paths.checkpoints_dir).list()]

    train.side_effect = crash_after_a_checkpoint

    with (
        pytest.raises(RuntimeError, match=r"^env exploded$"),
        PodGuard(runpod_terminator(paths, events, durable), interactive=False),
    ):
        train_run(SPEC, tmp_path, DEPS)

    assert events == [("DELETE", ("failed", ["model_it0000001_s000001.pt"]))]


@pytest.mark.usefixtures("sigterm_sentinel")
def test_sigterm_mid_training_leaves_the_run_failed_before_the_pod_guard_terminates(
    tmp_path: Path, train: MagicMock
) -> None:
    """SIGTERM after a checkpoint, under the CLI's handler: state.json is failed when the DELETE goes out."""
    paths = RunPaths(tmp_path, SPEC)
    events: list[tuple[str, object]] = []

    def sigterm_after_a_checkpoint(*_args: object) -> None:
        CheckpointStore(paths.checkpoints_dir).save_atomic(lambda p: p.write_bytes(b"weights"), 1, 1.0)
        os.kill(os.getpid(), signal.SIGTERM)

    def durable(state: RunState) -> object:
        return state.status, [ref.path.name for ref in CheckpointStore(paths.checkpoints_dir).list()]

    train.side_effect = sigterm_after_a_checkpoint

    with (
        pytest.raises(Terminated, match=r"^received SIGTERM"),
        sigterm_raises(),
        PodGuard(runpod_terminator(paths, events, durable), interactive=False),
    ):
        train_run(SPEC, tmp_path, DEPS)

    assert events == [("DELETE", ("failed", ["model_it0000001_s000001.pt"]))]


@pytest.mark.usefixtures("sigterm_sentinel")
def test_swallowed_sigterm_in_training_still_leaves_the_run_failed(tmp_path: Path, train: MagicMock) -> None:
    """A library inside training caught the Terminated; train_run must still fail the run before the DELETE."""
    paths = RunPaths(tmp_path, SPEC)
    events: list[tuple[str, object]] = []

    def swallowing_training(*_args: object) -> tuple[CheckpointRef, float, dict[str, str]]:
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Terminated:
            pass
        return _finished_run(tmp_path), 2.0, {}

    train.side_effect = swallowing_training

    with (
        pytest.raises(Terminated, match=r"^received SIGTERM"),
        sigterm_raises(),
        PodGuard(runpod_terminator(paths, events, lambda state: state.status), interactive=False),
    ):
        train_run(SPEC, tmp_path, DEPS)

    assert events == [("DELETE", "failed")]


class _RunnerSwallowingSigterm:
    """Duck-types ``TimedOnPolicyRunner``: SIGTERM is swallowed during iteration 0, then the loop would go on."""

    current_learning_iteration = 0

    def __init__(self) -> None:
        self.iterations_finished: list[int] = []

    def learn_for(self, *_args: object, on_iteration: Callable[[int], None], **_kwargs: object) -> None:
        for it in range(3):
            if it == 0:
                try:
                    os.kill(os.getpid(), signal.SIGTERM)
                except Terminated:
                    pass
            on_iteration(it)
            self.iterations_finished.append(it)


@pytest.mark.usefixtures("sigterm_sentinel")
def test_swallowed_sigterm_stops_training_at_the_next_iteration_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        train_mod.diagnostics, "for_runner", lambda *_args: (_StubDiagnostics(), lambda _it: None, lambda _it: None)
    )
    runner = _RunnerSwallowingSigterm()
    budget_s = SPEC.budget_minutes * 60.0

    with pytest.raises(Terminated, match=r"^received SIGTERM"), sigterm_raises():
        train_mod._learn(  # type: ignore[arg-type]  duck-typed runner
            SPEC, RunPaths(tmp_path, SPEC), DEPS, runner, TrainingClock(), budget_s, fresh_start=True
        )

    assert runner.iterations_finished == []


class _StubDiagnostics:
    def attach(self) -> None: ...
    def detach(self) -> None: ...


class _RunnerTerminatedAfterTheFinalCheckpoint:
    """Duck-types what ``train._learn`` uses of ``TimedOnPolicyRunner``: the final checkpoint, then SIGTERM."""

    current_learning_iteration = 3

    def learn_for(self, *_args: object, on_checkpoint: Callable[[CheckpointRef], None], **_kwargs: object) -> None:
        on_checkpoint(_ref(3, 2.0))
        raise Terminated("received SIGTERM; stopping")

    def export_policy_to_onnx(self, path: str, filename: str) -> None:
        raise AssertionError(f"exported {path}/{filename}: a terminated run must not continue post-training work")


def test_sigterm_after_the_final_checkpoint_is_a_failure_not_a_post_training_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        train_mod.diagnostics, "for_runner", lambda *_args: (_StubDiagnostics(), lambda _it: None, lambda _it: None)
    )
    runner = _RunnerTerminatedAfterTheFinalCheckpoint()
    budget_s = SPEC.budget_minutes * 60.0

    with pytest.raises(Terminated, match=r"^received SIGTERM"):
        train_mod._learn(  # type: ignore[arg-type]  duck-typed runner
            SPEC, RunPaths(tmp_path, SPEC), DEPS, runner, TrainingClock(budget_s), budget_s, fresh_start=True
        )


class _RunnerTerminatedDuringExport:
    current_learning_iteration = 3

    def export_policy_to_onnx(self, path: str, filename: str) -> None:
        raise Terminated(f"received SIGTERM; stopping (exporting {path}/{filename})")


def test_sigterm_during_onnx_export_propagates() -> None:
    with pytest.raises(Terminated, match=r"^received SIGTERM"):
        publish_mod.export_onnx(_RunnerTerminatedDuringExport(), _ref(3, 2.0))  # type: ignore[arg-type]  duck-typed


SECRETS = {"WANDB_API_KEY": "SENTINEL-wandb-3f9a1c7e5b", "HF_TOKEN": "hf_SENTINEL-hf-8d2e6a4c0f", "RUNPOD_API_KEY": "rpa_SENTINEL-rp"}


def test_run_files_and_upload_never_contain_secrets(
    tmp_path: Path, train: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    train.side_effect = _trains(tmp_path)
    staged: list[str] = []

    def record_upload(**kwargs: Any) -> MagicMock:
        staged.extend(p.read_text(errors="replace") for p in Path(kwargs["folder_path"]).rglob("*") if p.is_file())
        staged.append(repr(kwargs))
        return MagicMock(commit_url=f"https://huggingface.co/{REPO}/commit/abc")

    publisher, _ = _publisher(record_upload)

    train_run(SPEC, tmp_path, TrainDeps(publisher=publisher, wandb_mode="disabled", device="cpu"))

    on_disk = [p.read_text(errors="replace") for p in tmp_path.rglob("*") if p.is_file()]
    assert len(staged) == len(UPLOADED) + 1  # control: the upload ran
    assert [name for name, value in SECRETS.items() if value in "\n".join(on_disk + staged)] == []


# --- per-iteration W&B: training clock and the offline fallback ---------------------------------------


class _Logger:
    def __init__(self, writer: ResumableWandbWriter) -> None:
        self.writer = writer


class _RunnerOnTheClock:
    """Duck-types what ``train._learn`` uses: two iterations, 15 s on the clock each, then the final checkpoint."""

    current_learning_iteration = 1

    def __init__(self, writer: ResumableWandbWriter) -> None:
        self.logger = _Logger(writer)

    def learn_for(
        self,
        clock: TrainingClock,
        *_args: object,
        on_iteration: Callable[[int], None],
        on_checkpoint: Callable[[CheckpointRef], None],
        **_kwargs: object,
    ) -> SimpleNamespace:
        for it in range(2):
            with clock.measure():
                pass
            on_iteration(it)
        final = _ref(1, clock.elapsed_s)
        on_checkpoint(final)
        return SimpleNamespace(final=final)


def _fake_now(step_s: float) -> Callable[[], float]:
    """A time source that advances ``step_s`` per read: each ``measure()`` (a start and an end read) adds ``step_s``."""
    ticks = iter(range(1_000))
    return lambda: next(ticks) * step_s


def _learn_with(writer: ResumableWandbWriter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    monkeypatch.setattr(
        train_mod.diagnostics, "for_runner", lambda *_args: (_StubDiagnostics(), lambda _it: None, lambda _it: None)
    )
    _, _, errors = train_mod._learn(  # type: ignore[arg-type]  duck-typed runner
        SPEC,
        RunPaths(tmp_path, SPEC),
        DEPS,
        _RunnerOnTheClock(writer),
        TrainingClock(now=_fake_now(15.0)),
        3600.0,
        fresh_start=True,
    )
    return errors


def test_training_clock_is_logged_every_iteration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = create_autospec(wandb.Run, instance=True)
    monkeypatch.setattr(wandb, "init", create_autospec(wandb.init, return_value=run))
    writer = ResumableWandbWriter("logs", "p", "rid", "grp", mode="disabled")

    errors = _learn_with(writer, tmp_path, monkeypatch)

    logged = [(c.args[0], c.kwargs["step"]) for c in run.log.call_args_list]
    assert logged == [({CLOCK_METRIC: pytest.approx(0.25)}, 0), ({CLOCK_METRIC: pytest.approx(0.5)}, 1)]
    assert errors == {}


def test_wandb_offline_fallback_is_recorded_for_the_run_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = create_autospec(wandb.Run, instance=True)
    monkeypatch.setattr(wandb, "init", create_autospec(wandb.init, side_effect=[CommError("503"), run]))
    writer = ResumableWandbWriter("logs", "p", "rid", "grp", mode="online")

    errors = _learn_with(writer, tmp_path, monkeypatch)

    assert errors == {"wandb": "offline fallback after CommError: 503"}


def test_wandb_offline_fallback_lands_in_state_json(tmp_path: Path, train: MagicMock) -> None:
    final = _finished_run(tmp_path)
    train.return_value = (final, 2.0, {"wandb": "offline fallback after CommError: 503"})

    state = train_run(SPEC, tmp_path, DEPS)

    assert state.uploads == {"wandb": "offline fallback after CommError: 503"}
    assert expect(load_state(RunPaths(tmp_path, SPEC).state_file), "state").uploads == state.uploads


def test_wandb_tags_name_the_env_count_and_seed() -> None:
    assert train_mod.wandb_tags(RunSpec("main", 64, 2, 45.0)) == ["n00064", "seed2"]
