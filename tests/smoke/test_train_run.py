"""Smoke tests for train_run against real mjlab on CPU with 4 envs and W&B offline.

The training clock runs on a fake time source (support.Ticker): one iteration is exactly ITERATION_S = 0.5 s
of training clock on any machine, so budgets of 2-4 s run 4-8 iterations. Checkpoints every 1 s of training
clock (1/60 min). The SIGKILL test's child process keeps real time.

Real training runs only where it is the behavior: the fresh run (the session ``completed_run_template``),
a raised budget, the SIGKILL resume and the two post-training failures. Every other path starts from a
copy of the completed run (``completed_run``). The skip, upload-retry, W&B id, staging-error and
config-mismatch decisions are unit-tested on hand-built run directories (tests/unit/test_train_run.py).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest
import wandb
from huggingface_hub import HfApi
from support import EVERY_MIN, ITERATION_S, TICK_S, CompletedRun, Ticker, run_spec, runpod_terminator

import mprl.train as train_mod
from mprl.checkpoints import CheckpointStore
from mprl.config import RunSpec
from mprl.hub import HubPublisher
from mprl.negative_space import expect
from mprl.paths import RunPaths
from mprl.publish import finish_and_terminate
from mprl.run_state import RunState, load_state, save_state
from mprl.timed_runner import TimedOnPolicyRunner
from mprl.train import TrainDeps, resolve_cfgs, train_run, wandb_logger_cfg
from mprl.wandb_writer import ResumableWandbWriter

pytestmark = [pytest.mark.smoke, pytest.mark.usefixtures("offline_wandb")]

DEPS = TrainDeps(publisher=None, wandb_mode="offline", device="cpu", now=Ticker(TICK_S))
REPO = "someone/mprl-test"
SRC_DIR = Path(train_mod.__file__).resolve().parents[1]
KILL_DEADLINE_S = 180.0  # mjlab import + env build + first checkpoint, generously
POLL_S = 0.2


def _wait_for_first_checkpoint(child: Any, store: CheckpointStore) -> None:
    """Poll the volume until the child saved a checkpoint; fail if the child exits first (bounded)."""
    for _ in range(int(KILL_DEADLINE_S / POLL_S)):
        if store.latest() is not None:
            return
        try:
            code = child.wait(timeout=POLL_S)  # the bounded wait between polls
        except subprocess.TimeoutExpired:  # still training
            continue
        raise AssertionError(f"train_run child exited with {code} before its first checkpoint")


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


def test_fresh_run_completes_and_writes_state(completed_run_template: CompletedRun) -> None:
    spec = run_spec(2.0)
    paths = RunPaths(completed_run_template.volume, spec)
    state = completed_run_template.state

    assert state.status == "complete"
    assert state.run_id == spec.run_id
    assert state.config_hash == spec.config_hash()
    assert state.train_seconds == pytest.approx(4 * ITERATION_S)  # the 4th iteration reaches the 2 s budget
    assert state.budget_minutes == spec.budget_minutes
    assert state.uploads == {}
    assert state.git_sha == train_mod.git_sha()
    assert load_state(paths.state_file) == state
    refs = CheckpointStore(paths.checkpoints_dir).list()
    # Due at the 1 s mark (iteration 1 ends at 2 * ITERATION_S) and at the budget (iteration 3).
    assert [r.iteration for r in refs] == [1, 3]
    assert [r.train_seconds for r in refs] == pytest.approx([2 * ITERATION_S, 4 * ITERATION_S])
    assert (paths.checkpoints_dir / "checkpoints.onnx").is_file()
    assert {p.name for p in (paths.run_dir / "params").iterdir()} == {"env.yaml", "agent.yaml"}
    offline_runs = list((completed_run_template.wandb_dir / "wandb").glob(f"offline-run-*-{state.wandb_run_id}"))
    assert len(offline_runs) == 1


def test_raised_budget_continues(completed_run: CompletedRun) -> None:
    store = CheckpointStore(RunPaths(completed_run.volume, run_spec(2.0)).checkpoints_dir)
    first_refs = store.list()

    second = train_run(run_spec(4.0), completed_run.volume, DEPS)

    refs = store.list()
    assert second.status == "complete"
    assert second.train_seconds >= 4.0
    assert second.budget_minutes == pytest.approx(4.0 / 60.0)
    assert second.wandb_run_id == completed_run.state.wandb_run_id
    assert set(first_refs) < set(refs)
    assert min(r.iteration for r in refs if r not in first_refs) > first_refs[-1].iteration
    iterations = [r.iteration for r in refs]
    assert len(iterations) == len(set(iterations))


def test_crash_after_final_checkpoint_completes_without_training(completed_run: CompletedRun) -> None:
    spec = run_spec(2.0)
    paths = RunPaths(completed_run.volume, spec)
    first = completed_run.state
    refs = CheckpointStore(paths.checkpoints_dir).list()
    (paths.checkpoints_dir / "checkpoints.onnx").unlink()
    save_state(paths.state_file, RunState(**{**asdict(first), "status": "running"}))  # died before "complete"

    state = train_run(spec, completed_run.volume, DEPS)

    assert state.status == "complete"
    assert state.train_seconds == first.train_seconds
    assert state.uploads == {}
    assert CheckpointStore(paths.checkpoints_dir).list() == refs
    assert (paths.checkpoints_dir / "checkpoints.onnx").is_file()  # re-exported from the loaded checkpoint


def test_resume_after_kill_continues_clock(tmp_path: Path, subprocesses: Any) -> None:
    long_spec = run_spec(600.0)  # never finishes on its own; killed after the first checkpoint
    paths = RunPaths(tmp_path, long_spec)
    store = CheckpointStore(paths.checkpoints_dir)
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from mprl.config import RunSpec\n"
        "from mprl.train import TrainDeps, train_run\n"
        f"train_run(RunSpec('main', 4, 0, {long_spec.budget_minutes!r}, {EVERY_MIN!r}), Path(sys.argv[1]),"
        " TrainDeps(None, 'disabled', 'cpu'))\n"
    )
    # HOME stays the real one: Warp caches its compiled kernels under it (a fresh HOME recompiles them all).
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "PYTHONPATH": str(SRC_DIR), "WANDB_MODE": "disabled"}
    with (tmp_path / "child.log").open("w") as log:
        child = subprocesses.spawn([sys.executable, "-c", code, str(tmp_path)], env=env, stdout=log)
        try:
            _wait_for_first_checkpoint(child, store)
        finally:
            child.kill()  # SIGKILL: no handler, no cleanup, like a pod losing power (W&B is off: no grandchild)
            child.wait(timeout=30)
    killed_at = expect(store.latest(), f"no checkpoint within {KILL_DEADLINE_S} s; see {tmp_path / 'child.log'}")
    stored = expect(load_state(paths.state_file), "the child wrote no state.json")
    assert stored.status == "running"
    partial = paths.checkpoints_dir / ".model_it9999999_s999999.pt.tmp"
    partial.write_bytes(b"truncated torch payload")

    resumed_spec = run_spec(killed_at.train_seconds + 2.0)
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


def test_onnx_export_failure_is_recorded_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_export(self: TimedOnPolicyRunner, path: str, filename: str = "policy.onnx") -> None:
        raise RuntimeError("onnx exporter exploded")

    monkeypatch.setattr(TimedOnPolicyRunner, "export_policy_to_onnx", broken_export)
    spec = run_spec(2.0)

    state = train_run(spec, tmp_path, DEPS)

    assert state.status == "complete"
    assert state.uploads["onnx"] == "RuntimeError: onnx exporter exploded"
    assert state.uploads["post_training"] == "RuntimeError: onnx exporter exploded"
    assert CheckpointStore(RunPaths(tmp_path, spec).checkpoints_dir).latest() is not None
    assert load_state(RunPaths(tmp_path, spec).state_file) == state


def test_writer_stop_failure_still_exports_onnx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def stop_then_fail(self: ResumableWandbWriter) -> None:
        wandb.finish()
        raise RuntimeError("wandb down")

    monkeypatch.setattr(ResumableWandbWriter, "stop", stop_then_fail)
    spec = run_spec(2.0)
    paths = RunPaths(tmp_path, spec)

    state = train_run(spec, tmp_path, DEPS)

    final = expect(CheckpointStore(paths.checkpoints_dir).latest(), "no checkpoint")
    onnx = paths.checkpoints_dir / "checkpoints.onnx"
    assert state.status == "complete"
    assert state.uploads["post_training"] == "RuntimeError: wandb down"
    assert "onnx" not in state.uploads
    assert onnx.is_file()
    assert onnx.stat().st_mtime >= final.path.stat().st_mtime
    assert load_state(paths.state_file) == state


def test_failed_upload_of_real_run_files_is_durable_before_terminate(
    completed_run: CompletedRun, hub_http_error: Callable[[int], Exception]
) -> None:
    """The files a real run leaves (checkpoint, mjlab's ONNX name, params, state) are staged; DELETE is last."""
    spec = run_spec(2.0)
    paths = RunPaths(completed_run.volume, spec)
    events: list[tuple[str, object]] = []
    api = create_autospec(HfApi, instance=True)

    def failing_upload(**kwargs: Any) -> None:
        events.append(("upload", sorted(p.name for p in Path(kwargs["folder_path"]).rglob("*") if p.is_file())))
        raise hub_http_error(503)

    api.upload_folder.side_effect = failing_upload
    publisher = HubPublisher(api, REPO, max_attempts=1, sleep=lambda _s: None)

    # complete without an "ok" upload: train_run only retries the upload (pinned in tests/unit/test_train_run.py)
    state = train_run(spec, completed_run.volume, TrainDeps(publisher=publisher, wandb_mode="offline", device="cpu", now=Ticker(TICK_S)))
    finish_and_terminate([state], runpod_terminator(paths, events))

    final = expect(CheckpointStore(paths.checkpoints_dir).latest(), "no checkpoint")
    path_in_repo = f"runs/main/{spec.run_id}"
    assert events[0] == (
        "upload",
        sorted({final.path.name, "checkpoints.onnx", "env.yaml", "agent.yaml", "state.json"}),
    )
    assert [name for name, _ in events] == ["upload", "DELETE"]
    assert state.uploads[path_in_repo].startswith("UploadFailed:")
    assert events[-1][1] == state.uploads  # the failure was durable on disk before the DELETE
    assert not (paths.run_dir / "upload").exists()


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


SECRETS = {
    "WANDB_API_KEY": "SENTINEL-wandb-3f9a1c7e5b",
    "HF_TOKEN": "hf_SENTINEL-hf-8d2e6a4c0f",
    "RUNPOD_API_KEY": "rpa_SENTINEL-rp",
}


def test_wandb_run_config_never_contains_secrets() -> None:
    """Everything the W&B writer is configured with (extra_config) and stores (train_cfg, env_cfg), from a full environ."""
    spec = RunSpec("main", 64, 0, 45.0)
    environ = {**SECRETS, "WANDB_PROJECT": "proj", "WANDB_ENTITY": "team", "MPRL_IMAGE_DIGEST": "sha256:" + "d" * 64}

    logger = wandb_logger_cfg(spec, "rid", None, "offline", git_sha="c" * 40, environ=environ)
    env_cfg, agent_cfg = resolve_cfgs(spec, logger)

    stored = repr({"env": asdict(env_cfg), "agent": agent_cfg})
    assert (logger["project"], logger["entity"], logger["extra_config"]["image_digest"]) == (
        "proj",
        "team",
        environ["MPRL_IMAGE_DIGEST"],
    )  # control: the environ was read
    assert [name for name, value in SECRETS.items() if value in stored] == []
