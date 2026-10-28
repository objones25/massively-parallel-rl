"""Smoke tests for train_run / finish_and_terminate against real mjlab on CPU with 4 envs and W&B offline.

One G1 flat iteration at 4 envs is about 0.5 s of training clock on this Mac, so budgets of 2-4 s run a
handful of iterations. Checkpoints every 1 s of training clock (1/60 min).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import httpx
import httpx2
import pytest
import wandb
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import hf_raise_for_status

import mprl.train as train_mod
from mprl.checkpoints import CheckpointStore
from mprl.config import RunSpec
from mprl.errors import ConfigMismatch
from mprl.hub import HubPublisher
from mprl.negative_space import expect
from mprl.paths import RunPaths
from mprl.run_state import RunState, load_state, save_state
from mprl.runpod import PodTerminator
from mprl.timed_runner import TimedOnPolicyRunner
from mprl.train import TrainDeps, finish_and_terminate, resolve_cfgs, train_run, wandb_logger_cfg

pytestmark = pytest.mark.smoke

EVERY_S = 1.0 / 60.0  # checkpoint every 1 s of training clock
DEPS = TrainDeps(publisher=None, wandb_mode="offline", device="cpu")
REPO = "someone/mprl-test"
POD = "abc123podid"
SRC_DIR = Path(train_mod.__file__).resolve().parents[1]
KILL_DEADLINE_S = 180.0  # mjlab import + env build + first checkpoint, generously


def _spec(budget_s: float, *, every_minutes: float = EVERY_S, num_envs: int = 4, seed: int = 0) -> RunSpec:
    return RunSpec("main", num_envs, seed, budget_s / 60.0, every_minutes)


@pytest.fixture(autouse=True)
def _offline_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """wandb reads WANDB_DIR once per service; restart it so each test sees its own tmp_path."""
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    wandb.teardown()
    yield
    wandb.teardown()


def _http_error(status: int) -> HfHubHTTPError:
    request = httpx2.Request("POST", f"https://huggingface.co/api/models/{REPO}/commit/main")
    try:
        hf_raise_for_status(httpx2.Response(status, request=request))
    except HfHubHTTPError as error:
        return error
    raise AssertionError(f"status {status} did not raise")


def _wait_for_first_checkpoint(child: subprocess.Popen[bytes], store: CheckpointStore) -> None:
    """Poll the volume until the child saved a checkpoint; fail if the child exits first."""
    deadline = time.monotonic() + KILL_DEADLINE_S
    while store.latest() is None and time.monotonic() < deadline:
        try:
            code = child.wait(timeout=0.2)  # the bounded wait between polls
        except subprocess.TimeoutExpired:
            continue
        raise AssertionError(f"train_run child exited with {code} before its first checkpoint")


def _runpod_handler(paths: RunPaths, events: list[tuple[str, object]]) -> Callable[[httpx.Request], httpx.Response]:
    """A RunPod pod that is RUNNING until DELETE (which records the on-disk uploads), then 404."""

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            on_disk = expect(load_state(paths.state_file), "no state.json at DELETE time")
            events.append(("DELETE", on_disk.uploads))
            return httpx.Response(204)
        if any(name == "DELETE" for name, _ in events):
            return httpx.Response(404)
        return httpx.Response(200, json={"id": POD, "status": "RUNNING"})

    return handle


def _terminator(paths: RunPaths, events: list[tuple[str, object]]) -> PodTerminator:
    transport = httpx.MockTransport(_runpod_handler(paths, events))
    return PodTerminator(httpx.Client(transport=transport), "key", POD, base_url="https://api.test/v2")


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for key, item in value.items():
            out |= _flatten(item, f"{prefix}.{key}" if prefix else str(key))
        return out
    if isinstance(value, list | tuple):
        out = {}
        for i, item in enumerate(value):
            out |= _flatten(item, f"{prefix}[{i}]")
        return out
    return {prefix: value}


def _diff(a: dict[str, Any], b: dict[str, Any]) -> set[str]:
    fa, fb = _flatten(a), _flatten(b)
    return {key for key in fa.keys() | fb.keys() if fa.get(key, "<missing>") != fb.get(key, "<missing>")}


def test_fresh_run_completes_and_writes_state(tmp_path: Path) -> None:
    spec = _spec(2.0)
    paths = RunPaths(tmp_path, spec)

    state = train_run(spec, tmp_path, DEPS)

    assert state.status == "complete"
    assert state.run_id == spec.run_id
    assert state.config_hash == spec.config_hash()
    assert state.train_seconds >= 2.0
    assert state.budget_minutes == spec.budget_minutes
    assert state.uploads == {}
    assert state.git_sha != ""
    assert load_state(paths.state_file) == state
    refs = CheckpointStore(paths.checkpoints_dir).list()
    assert len(refs) >= 2
    assert (paths.checkpoints_dir / "checkpoints.onnx").is_file()
    assert {p.name for p in (paths.run_dir / "params").iterdir()} == {"env.yaml", "agent.yaml"}
    assert len(list((tmp_path / "wandb").glob(f"offline-run-*-{state.wandb_run_id}"))) == 1


def test_completed_run_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec(2.0)
    first = train_run(spec, tmp_path, DEPS)
    before = CheckpointStore(RunPaths(tmp_path, spec).checkpoints_dir).list()

    def no_env(*args: object, **kwargs: object) -> None:
        raise AssertionError("a completed run built an environment")

    monkeypatch.setattr(train_mod, "ManagerBasedRlEnv", no_env)
    started = time.monotonic()
    second = train_run(spec, tmp_path, DEPS)

    assert time.monotonic() - started < 1.0
    assert second == first
    assert CheckpointStore(RunPaths(tmp_path, spec).checkpoints_dir).list() == before


def test_raised_budget_continues(tmp_path: Path) -> None:
    first = train_run(_spec(2.0), tmp_path, DEPS)
    store = CheckpointStore(RunPaths(tmp_path, _spec(2.0)).checkpoints_dir)
    first_refs = store.list()

    second = train_run(_spec(4.0), tmp_path, DEPS)

    refs = store.list()
    assert second.status == "complete"
    assert second.train_seconds >= 4.0
    assert second.budget_minutes == pytest.approx(4.0 / 60.0)
    assert second.wandb_run_id == first.wandb_run_id
    assert first_refs[0] in refs
    assert set(first_refs) < set(refs)
    assert min(r.iteration for r in refs if r not in first_refs) > first_refs[-1].iteration
    iterations = [r.iteration for r in refs]
    assert len(iterations) == len(set(iterations))


def test_resume_after_kill_continues_clock(tmp_path: Path) -> None:
    long_spec = _spec(600.0)  # never finishes on its own; killed after the first checkpoint
    paths = RunPaths(tmp_path, long_spec)
    store = CheckpointStore(paths.checkpoints_dir)
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from mprl.config import RunSpec\n"
        "from mprl.train import TrainDeps, train_run\n"
        f"train_run(RunSpec('main', 4, 0, {long_spec.budget_minutes!r}, {EVERY_S!r}), Path(sys.argv[1]),"
        " TrainDeps(None, 'offline', 'cpu'))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC_DIR), "WANDB_DIR": str(tmp_path)}
    with (tmp_path / "child.log").open("w") as log:
        child = subprocess.Popen(  # noqa: S603  fixed interpreter and code
            [sys.executable, "-c", code, str(tmp_path)], env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            _wait_for_first_checkpoint(child, store)
        finally:
            child.send_signal(signal.SIGKILL)
            child.wait(timeout=30)
    killed_at = expect(store.latest(), f"no checkpoint within {KILL_DEADLINE_S} s; see {tmp_path / 'child.log'}")
    stored = expect(load_state(paths.state_file), "the child wrote no state.json")
    assert stored.status == "running"
    partial = paths.checkpoints_dir / ".model_it9999999_s999999.pt.tmp"
    partial.write_bytes(b"truncated torch payload")

    resumed_spec = _spec(killed_at.train_seconds + 2.0)
    state = train_run(resumed_spec, tmp_path, DEPS)

    refs = store.list()
    assert state.status == "complete"
    assert state.train_seconds >= killed_at.train_seconds + 2.0
    assert state.wandb_run_id == stored.wandb_run_id
    assert partial.exists()
    assert all(r.path != partial for r in refs)
    new = [r for r in refs if r.train_seconds > killed_at.train_seconds]
    assert new
    assert all(r.iteration > killed_at.iteration for r in new)  # continued after the loaded iteration
    assert len({r.iteration for r in refs}) == len(refs)


def test_resume_refuses_config_mismatch(tmp_path: Path) -> None:
    started = _spec(2.0, every_minutes=EVERY_S)
    changed = _spec(2.0, every_minutes=2 * EVERY_S)  # same run id, different config
    save_state(
        RunPaths(tmp_path, started).state_file,
        RunState(started.run_id, started.config_hash(), "wid", "running", 1.0, started.budget_minutes, "sha", {}),
    )

    with pytest.raises(ConfigMismatch, match="main-n00004-s0"):
        train_run(changed, tmp_path, DEPS)


def test_run_without_checkpoint_mints_new_wandb_id(tmp_path: Path) -> None:
    """A crash before the first checkpoint must not continue the crashed W&B run (its steps restart at 0)."""
    spec = _spec(2.0)
    save_state(
        RunPaths(tmp_path, spec).state_file,
        RunState(spec.run_id, spec.config_hash(), "crashed0", "running", 0.0, spec.budget_minutes, "sha", {}),
    )

    state = train_run(spec, tmp_path, DEPS)

    assert state.status == "complete"
    assert state.wandb_run_id != "crashed0"


def test_crash_after_final_checkpoint_completes_without_training(tmp_path: Path) -> None:
    spec = _spec(2.0)
    paths = RunPaths(tmp_path, spec)
    first = train_run(spec, tmp_path, DEPS)
    refs = CheckpointStore(paths.checkpoints_dir).list()
    save_state(paths.state_file, RunState(**{**asdict(first), "status": "running"}))  # died before "complete"

    state = train_run(spec, tmp_path, DEPS)

    assert state.status == "complete"
    assert state.train_seconds == first.train_seconds
    assert CheckpointStore(paths.checkpoints_dir).list() == refs


def test_onnx_export_failure_is_recorded_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_export(self: TimedOnPolicyRunner, path: str, filename: str = "policy.onnx") -> None:
        raise RuntimeError("onnx exporter exploded")

    monkeypatch.setattr(TimedOnPolicyRunner, "export_policy_to_onnx", broken_export)
    spec = _spec(2.0)

    state = train_run(spec, tmp_path, DEPS)

    assert state.status == "complete"
    assert state.uploads["onnx"] == "RuntimeError: onnx exporter exploded"
    assert CheckpointStore(RunPaths(tmp_path, spec).checkpoints_dir).latest() is not None
    assert load_state(RunPaths(tmp_path, spec).state_file) == state


@pytest.mark.parametrize(
    ("status", "recorded"),
    [(503, "UploadFailed"), (403, "HfHubHTTPError")],
)
def test_terminate_runs_after_failed_upload(tmp_path: Path, status: int, recorded: str) -> None:
    spec = _spec(2.0)
    paths = RunPaths(tmp_path, spec)
    events: list[tuple[str, object]] = []
    api = create_autospec(HfApi, instance=True)

    def failing_upload(**kwargs: Any) -> None:
        events.append(("upload", sorted(p.name for p in Path(kwargs["folder_path"]).rglob("*") if p.is_file())))
        raise _http_error(status)

    api.upload_folder.side_effect = failing_upload
    publisher = HubPublisher(api, REPO, max_attempts=2, sleep=lambda _s: None)

    terminator = _terminator(paths, events)

    state = train_run(spec, tmp_path, TrainDeps(publisher=publisher, wandb_mode="offline", device="cpu"))
    finish_and_terminate([state], terminator)

    path_in_repo = f"runs/main/{spec.run_id}"
    final = expect(CheckpointStore(paths.checkpoints_dir).latest(), "no checkpoint")
    uploaded = {final.path.name, "checkpoints.onnx", "env.yaml", "agent.yaml", "state.json"}
    upload_events = [files for name, files in events if name == "upload"]
    assert upload_events
    assert all(set(files) == uploaded for files in upload_events)  # type: ignore[arg-type]
    assert events[-1][0] == "DELETE"
    assert [name for name, _ in events].count("DELETE") == 1
    assert state.uploads[path_in_repo].startswith(f"{recorded}:")
    assert str(status) in state.uploads[path_in_repo]
    assert events[-1][1] == state.uploads  # the failure was durable on disk before the DELETE
    assert final.path.is_file()
    assert api.upload_folder.call_args.kwargs["path_in_repo"] == path_in_repo


def test_only_num_envs_and_seed_differ_between_sweep_runs() -> None:
    small, large = RunSpec("main", 64, 0, 45.0), RunSpec("main", 4096, 1, 45.0)

    def resolved(spec: RunSpec) -> dict[str, Any]:
        logger = wandb_logger_cfg(spec, f"{spec.run_id}-id", None, "offline", git_sha="sha", environ={})
        env_cfg, agent_cfg = resolve_cfgs(spec, logger)
        return {"env": asdict(env_cfg), "agent": agent_cfg}

    a, b = resolved(small), resolved(large)

    assert a["env"]["scene"]["num_envs"] == 64
    assert b["env"]["scene"]["num_envs"] == 4096
    assert (a["env"]["seed"], a["agent"]["seed"]) == (0, 0)
    assert (b["env"]["seed"], b["agent"]["seed"]) == (1, 1)
    # The W&B run config restates num_envs, seed and the batch arithmetic derived from num_envs.
    derived = {
        "agent.logger.extra_config.run.num_envs",
        "agent.logger.extra_config.run.seed",
        "agent.logger.extra_config.run_id",
        "agent.logger.extra_config.batch.num_envs",
        "agent.logger.extra_config.batch.batch_size",
        "agent.logger.extra_config.batch.minibatch_size",
    }
    assert _diff(a, b) == {"env.scene.num_envs", "env.seed", "agent.seed", "agent.logger.run_id"} | derived
    assert json.dumps(a["agent"]["logger"]["extra_config"])  # serializable for wandb.config
