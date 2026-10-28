"""One training run end to end: skip / resume, train on the training clock, record state, upload.

Control flow of ``train_run`` (all paths under ``RunPaths(root, spec)``):

1. A stored ``state.json`` whose ``config_hash`` differs from the spec's raises ``ConfigMismatch``.
2. A ``complete`` state whose stored budget covers ``spec.budget_minutes`` is returned without building an
   env; if a publisher is given and the run's Hub upload is not "ok", only the upload is retried.
3. With a checkpoint on disk the run resumes: the stored W&B id is reused with
   ``resume_from_iteration = checkpoint iteration``, the latest checkpoint is loaded into a fresh runner
   and the clock restored from it. With no checkpoint (fresh, or crashed before the first one) a new W&B
   id is minted: reusing the crashed run's id would mix its history with steps that restart at 0.
4. ``state.json`` is written ``running``, then ``learn_for`` runs to ``spec.budget_minutes * 60`` s of
   *total* training clock (cumulative across resumes). An error during training writes ``failed`` and
   re-raises. A checkpoint already at or past the budget (killed after the final save) trains nothing.
5. ``state.json`` is written ``complete``, then ``mprl.publish.upload_run`` uploads the final checkpoint, its
   ONNX, the params and ``state.json``. Its outcome, and any error after the final checkpoint (writer stop,
   ONNX export: ``post_training`` / ``onnx``), land in ``state.uploads``; none of them raises.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mjlab.tasks  # noqa: F401  # pyright: ignore[reportUnusedImport]  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

from mprl import diagnostics
from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.clock import TrainingClock, cuda_sync_for
from mprl.config import RunSpec, batch_arithmetic
from mprl.errors import ConfigMismatch
from mprl.negative_space import expect, require
from mprl.paths import RunPaths
from mprl.publish import export_onnx, hub_path, upload_run
from mprl.run_state import RunState, load_state, save_state
from mprl.timed_runner import TimedOnPolicyRunner
from mprl.wandb_writer import logger_cfg

if TYPE_CHECKING:
    from mprl.hub import HubPublisher

log = logging.getLogger(__name__)

_REPO_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_WANDB_PROJECT = "mprl-humanoid-walk"  # .env.example default (plan, Task 14)
_ONNX_KEY = "onnx"  # state.uploads keys for errors that are not uploads
_POST_TRAINING_KEY = "post_training"


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
    rl_cfg = expect(loaded if isinstance(loaded, RslRlOnPolicyRunnerCfg) else None, f"{spec.task_id} is not a PPO task")
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


def resolve_cfgs(spec: RunSpec, logger: dict[str, Any] | str) -> tuple[ManagerBasedRlEnvCfg, dict[str, Any]]:
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


@contextmanager
def built_runner(
    spec: RunSpec,
    device: str,
    logger: dict[str, Any] | str,
    log_dir: Path,
    *,
    params_dir: Path | None = None,
    env_cfg: ManagerBasedRlEnvCfg | None = None,
) -> Generator[TimedOnPolicyRunner]:
    """The real env (``resolve_cfgs``) wrapped in a fresh ``TimedOnPolicyRunner``; the env is closed on exit.

    With ``params_dir`` the env and agent configs are dumped there before the runner exists, as mjlab
    does: the runner mutates ``agent_cfg`` in place. With ``env_cfg`` that env config is built instead of
    the training one (the evaluator's play env); the agent config, and so the policy, is unchanged.
    """
    training_env_cfg, agent_cfg = resolve_cfgs(spec, logger)
    env_cfg = training_env_cfg if env_cfg is None else env_cfg
    configure_torch_backends()  # as mjlab's train script
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    try:
        if params_dir is not None:
            dump_yaml(params_dir / "env.yaml", asdict(env_cfg))
            dump_yaml(params_dir / "agent.yaml", agent_cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg["clip_actions"])
        yield TimedOnPolicyRunner(wrapped, agent_cfg, str(log_dir), device)
    finally:
        env.close()


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
        if deps.publisher is not None and state.uploads.get(hub_path(spec)) != "ok":
            log.info("%s: complete; retrying its upload only", spec.run_id)
            final = expect(
                CheckpointStore(paths.checkpoints_dir).latest(), f"{spec.run_id} is complete without a checkpoint"
            )
            state.uploads[hub_path(spec)] = upload_run(deps.publisher, paths, final)
            save_state(paths.state_file, state)
        else:
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
        final, train_seconds, errors = _train(spec, paths, deps, logger, latest)
    except BaseException:
        state.status = "failed"
        save_state(paths.state_file, state)
        raise
    state.status = "complete"
    state.train_seconds = train_seconds
    state.uploads.update(errors)
    save_state(paths.state_file, state)
    if deps.publisher is not None:
        state.uploads[hub_path(spec)] = upload_run(deps.publisher, paths, final)
        save_state(paths.state_file, state)
    return state


def _train(
    spec: RunSpec, paths: RunPaths, deps: TrainDeps, logger: dict[str, Any], latest: CheckpointRef | None
) -> tuple[CheckpointRef, float, dict[str, str]]:
    """Build env + fresh runner, resume from ``latest``, train to the budget: (final ref, clock s, errors)."""
    budget_s = spec.budget_minutes * 60.0
    with built_runner(spec, deps.device, logger, paths.tensorboard_dir, params_dir=paths.run_dir / "params") as runner:
        sync = cuda_sync_for(deps.device)
        if latest is None:
            clock = TrainingClock(sync=sync)
        else:
            infos = runner.load(str(latest.path))
            clock = TrainingClock.from_state_dict(infos["clock"], sync=sync)
        if latest is not None and clock.elapsed_s >= budget_s:
            log.info("%s: checkpoint %s already covers the budget", spec.run_id, latest.path.name)
            onnx_error = export_onnx(runner, latest)
            return latest, clock.elapsed_s, {} if onnx_error is None else {_ONNX_KEY: onnx_error}
        return _learn(spec, paths, deps, runner, clock, budget_s, fresh_start=latest is None)


def _learn(
    spec: RunSpec,
    paths: RunPaths,
    deps: TrainDeps,
    runner: TimedOnPolicyRunner,
    clock: TrainingClock,
    budget_s: float,
    *,
    fresh_start: bool,
) -> tuple[CheckpointRef, float, dict[str, str]]:
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
        # The writer stop runs before the export, so its failure skipped the ONNX: export it here.
        errors = {_POST_TRAINING_KEY: f"{type(error).__name__}: {error}"}
        onnx_error = export_onnx(runner, saved[-1])
        if onnx_error is not None:
            errors[_ONNX_KEY] = onnx_error
        return saved[-1], clock.elapsed_s, errors
    finally:
        diag.detach()
    require(result.final == saved[-1], "learn_for's final checkpoint was not reported to on_checkpoint")
    return result.final, clock.elapsed_s, {}
