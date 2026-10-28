"""One training run end to end: skip / resume, train on the training clock, record state, upload.

Control flow of ``train_run`` (all paths under ``RunPaths(root, spec)``):

1. A stored ``state.json`` whose ``config_hash`` differs from the spec's raises ``ConfigMismatch``.
2. A ``complete`` state whose stored budget covers ``spec.budget_minutes`` is returned at once.
3. With a checkpoint on disk the run resumes: the stored W&B id is reused with
   ``resume_from_iteration = checkpoint iteration``, the latest checkpoint is loaded into a fresh runner
   and the clock restored from it. With no checkpoint (fresh, or crashed before the first one) a new W&B
   id is minted: reusing the crashed run's id would mix its history with steps that restart at 0.
4. ``state.json`` is written ``running``, then ``learn_for`` runs to ``spec.budget_minutes * 60`` s of
   *total* training clock (cumulative across resumes). An error during training writes ``failed`` and
   re-raises. A checkpoint already at or past the budget (killed after the final save) trains nothing.
5. ``state.json`` is written ``complete``, then the final checkpoint, its ONNX, the params and
   ``state.json`` are uploaded; the outcome (or the ONNX export error) lands in ``state.uploads``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mjlab.tasks  # noqa: F401  # pyright: ignore[reportUnusedImport]  (registers the tasks)
from huggingface_hub.errors import HfHubHTTPError
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

from mprl import diagnostics
from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.clock import TrainingClock, cuda_sync_for
from mprl.config import RunSpec, batch_arithmetic
from mprl.errors import ConfigMismatch, UploadFailed
from mprl.negative_space import expect, require
from mprl.paths import RunPaths
from mprl.run_state import RunState, load_state, save_state
from mprl.timed_runner import TimedOnPolicyRunner
from mprl.wandb_writer import logger_cfg

if TYPE_CHECKING:
    from mprl.hub import HubPublisher
    from mprl.runpod import PodTerminator

log = logging.getLogger(__name__)

_REPO_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_WANDB_PROJECT = "mprl-humanoid-walk"  # .env.example default (plan, Task 14)
_ONNX_KEY = "onnx"


@dataclass(frozen=True)
class TrainDeps:
    publisher: HubPublisher | None
    wandb_mode: str
    device: str
    # Diagnostics every 10 iterations: ~1 point per 5-10 s of training clock at 4096 envs, while the
    # extra activation forward pass and metric writes stay a small fraction of off-clock time.
    diagnostics_every_k: int = 10


def git_sha(repo: Path = _REPO_DIR) -> str:
    """HEAD of the repo, or "unknown" when git or the checkout is unavailable (an image without .git)."""
    git = shutil.which("git")
    if git is None:
        log.warning("git not found; recording git_sha=unknown")
        return "unknown"
    try:
        out = subprocess.run(  # noqa: S603  fixed argv, no shell
            [git, "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError) as error:
        log.warning("git rev-parse failed (%s); recording git_sha=unknown", error)
        return "unknown"
    return out.stdout.strip() or "unknown"


def wandb_logger_cfg(
    spec: RunSpec,
    wandb_run_id: str,
    resume_from_iteration: int | None,
    mode: str,
    *,
    git_sha: str,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """``agent_cfg["logger"]``: the resumable writer plus run config (spec, batch arithmetic, SHA, digest)."""
    loaded = load_rl_cfg(spec.task_id)
    rl_cfg = expect(
        loaded if isinstance(loaded, RslRlOnPolicyRunnerCfg) else None, f"{spec.task_id} is not a PPO task"
    )
    batch = batch_arithmetic(spec.num_envs, rl_cfg.num_steps_per_env, rl_cfg.algorithm.num_mini_batches)
    extra: dict[str, Any] = {
        "run": asdict(spec),
        "run_id": spec.run_id,
        "batch": {**asdict(batch), "batch_size": batch.batch_size, "minibatch_size": batch.minibatch_size},
        "git_sha": git_sha,
    }
    if environ.get("MPRL_IMAGE_DIGEST"):
        extra["image_digest"] = environ["MPRL_IMAGE_DIGEST"]
    cfg = logger_cfg(
        environ.get("WANDB_PROJECT", _DEFAULT_WANDB_PROJECT), wandb_run_id, spec.sweep, resume_from_iteration, mode
    )
    cfg["extra_config"] = extra
    if environ.get("WANDB_ENTITY"):
        cfg["entity"] = environ["WANDB_ENTITY"]
    return cfg


def resolve_cfgs(spec: RunSpec, logger: dict[str, Any]) -> tuple[ManagerBasedRlEnvCfg, dict[str, Any]]:
    """The shipped task configs with only ``num_envs``, the seeds and the logger set.

    Seeds as ``mjlab/scripts/train.py`` does on one device: ``agent.seed = env.seed = seed``; the env's
    ``seed_rng`` then seeds random, numpy, warp and torch.
    """
    env_cfg = load_env_cfg(spec.task_id)
    env_cfg.scene.num_envs = spec.num_envs
    env_cfg.seed = spec.seed
    rl_cfg = load_rl_cfg(spec.task_id)
    rl_cfg.seed = spec.seed
    agent_cfg = asdict(rl_cfg)
    agent_cfg["logger"] = logger
    return env_cfg, agent_cfg


def train_run(spec: RunSpec, root: Path, deps: TrainDeps) -> RunState:
    paths = RunPaths(root, spec)
    state = load_state(paths.state_file)
    if state is not None and state.config_hash != spec.config_hash():
        raise ConfigMismatch(
            f"run {spec.run_id}: stored config hash {state.config_hash[:12]} != requested {spec.config_hash()[:12]}; "
            "sweep, num_envs, seed, checkpoint_every_minutes or task_id changed since the run started. "
            "Use a new sweep name or remove the run directory."
        )
    if state is not None and state.status == "complete" and state.budget_minutes >= spec.budget_minutes:
        log.info("%s: complete at %.1f min budget; skipping", spec.run_id, state.budget_minutes)
        return state

    latest = CheckpointStore(paths.checkpoints_dir).latest()
    resuming = latest is not None and state is not None
    wandb_run_id = state.wandb_run_id if resuming and state else f"{spec.run_id}-{uuid.uuid4().hex[:8]}"
    sha = git_sha()
    state = RunState(
        spec.run_id,
        spec.config_hash(),
        wandb_run_id,
        "running",
        state.train_seconds if state is not None and latest is not None else 0.0,
        spec.budget_minutes,
        sha,
        dict(state.uploads) if state is not None else {},
    )
    save_state(paths.state_file, state)
    logger = wandb_logger_cfg(
        spec,
        wandb_run_id,
        latest.iteration if resuming and latest else None,
        deps.wandb_mode,
        git_sha=sha,
        environ=os.environ,
    )
    try:
        final, train_seconds, onnx_error = _train(spec, paths, deps, logger, latest)
    except BaseException:
        state.status = "failed"
        save_state(paths.state_file, state)
        raise
    state.status = "complete"
    state.train_seconds = train_seconds
    if onnx_error is not None:
        state.uploads[_ONNX_KEY] = onnx_error
    save_state(paths.state_file, state)
    if deps.publisher is not None:
        path_in_repo = f"runs/{spec.sweep}/{spec.run_id}"
        state.uploads[path_in_repo] = _upload(deps.publisher, paths, final, path_in_repo)
        save_state(paths.state_file, state)
    return state


def _train(
    spec: RunSpec, paths: RunPaths, deps: TrainDeps, logger: dict[str, Any], latest: CheckpointRef | None
) -> tuple[CheckpointRef, float, str | None]:
    """Build env + fresh runner, resume from ``latest``, train to the budget: (final ref, clock s, ONNX error)."""
    budget_s = spec.budget_minutes * 60.0
    env_cfg, agent_cfg = resolve_cfgs(spec, logger)
    configure_torch_backends()  # as mjlab's train script
    env = ManagerBasedRlEnv(cfg=env_cfg, device=deps.device)
    try:
        # Written before the runner exists, as mjlab does: the runner mutates agent_cfg in place.
        dump_yaml(paths.run_dir / "params" / "env.yaml", asdict(env_cfg))
        dump_yaml(paths.run_dir / "params" / "agent.yaml", agent_cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg["clip_actions"])
        runner = TimedOnPolicyRunner(wrapped, agent_cfg, str(paths.tensorboard_dir), deps.device)
        sync = cuda_sync_for(deps.device)
        if latest is None:
            clock = TrainingClock(sync=sync)
        else:
            infos = runner.load(str(latest.path))
            clock = TrainingClock.from_state_dict(infos["clock"], sync=sync)
        if latest is not None and clock.elapsed_s >= budget_s:
            log.info("%s: checkpoint %s already covers the budget", spec.run_id, latest.path.name)
            return latest, clock.elapsed_s, _export_onnx(runner, latest)
        return _learn(spec, paths, deps, runner, clock, budget_s, fresh_start=latest is None)
    finally:
        env.close()


def _learn(
    spec: RunSpec,
    paths: RunPaths,
    deps: TrainDeps,
    runner: TimedOnPolicyRunner,
    clock: TrainingClock,
    budget_s: float,
    *,
    fresh_start: bool,
) -> tuple[CheckpointRef, float, str | None]:
    saved: list[CheckpointRef] = []
    diag, on_before_update, on_iteration = diagnostics.for_runner(runner, deps.diagnostics_every_k)
    diag.attach()
    try:
        result = runner.learn_for(
            clock,
            budget_s,
            spec.checkpoint_every_minutes * 60.0,
            CheckpointStore(paths.checkpoints_dir),
            fresh_start=fresh_start,
            on_iteration=on_iteration,
            on_before_update=on_before_update,
            on_checkpoint=saved.append,
        )
    except Exception as error:
        # After the final checkpoint learn_for only stops the writer and exports ONNX. A failure there
        # must not discard a finished run: record it. Anything earlier is a training failure.
        final_saved = bool(saved) and clock.elapsed_s >= budget_s
        if not final_saved or saved[-1].iteration != runner.current_learning_iteration:
            raise
        log.warning("%s: post-training step failed after the final checkpoint: %s", spec.run_id, error)
        return saved[-1], clock.elapsed_s, f"{type(error).__name__}: {error}"
    finally:
        diag.detach()
    require(result.final == saved[-1], "learn_for's final checkpoint was not reported to on_checkpoint")
    return result.final, clock.elapsed_s, None


def _onnx_path(checkpoint: CheckpointRef) -> Path:
    # MjlabOnPolicyRunner._get_export_paths: "<checkpoint dir>/<checkpoint dir name>.onnx"
    return checkpoint.path.parent / f"{checkpoint.path.parent.name}.onnx"


def _export_onnx(runner: TimedOnPolicyRunner, final: CheckpointRef) -> str | None:
    if _onnx_path(final).is_file():
        return None
    try:
        runner.export_policy_to_onnx(str(final.path.parent), _onnx_path(final).name)
    except Exception as error:
        log.warning("ONNX export for %s failed: %s", final.path.name, error)
        return f"{type(error).__name__}: {error}"
    return None


def _upload(publisher: HubPublisher, paths: RunPaths, final: CheckpointRef, path_in_repo: str) -> str:
    """Stage final checkpoint, ONNX, params and state.json; upload; return "ok" or the error text."""
    staging = paths.run_dir / "upload"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copy2(final.path, staging / final.path.name)
    if _onnx_path(final).is_file():
        shutil.copy2(_onnx_path(final), staging / _onnx_path(final).name)
    if (paths.run_dir / "params").is_dir():
        shutil.copytree(paths.run_dir / "params", staging / "params")
    shutil.copy2(paths.state_file, staging / paths.state_file.name)
    try:
        publisher.ensure_repo()
        publisher.upload_folder(staging, path_in_repo, f"{paths.spec.run_id}: {final.path.name}")
    except (UploadFailed, HfHubHTTPError) as error:
        log.error("upload of %s to %s failed: %s", paths.spec.run_id, path_in_repo, error)
        return f"{type(error).__name__}: {error}"
    return "ok"


def finish_and_terminate(states: list[RunState], terminator: PodTerminator | None) -> None:
    """Log the outcome of every run, flush everything to disk, then terminate the pod as the very last action.

    Every ``state.json`` and upload outcome was saved by ``train_run``; ``os.sync`` makes them durable on the
    volume before the DELETE, which may kill this process before the confirming read returns.
    """
    for state in states:
        log.info(
            "%s: %s at %.1f s, uploads %s", state.run_id, state.status, state.train_seconds, state.uploads or "none"
        )
    os.sync()
    if terminator is None:
        log.info("no pod terminator; leaving the pod running")
        return
    terminator.terminate()
