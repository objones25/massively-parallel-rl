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

import gc
import logging
import os
import re
import time
import uuid
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import torch

from mprl import diagnostics
from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.clock import TrainingClock, cuda_sync_for
from mprl.config import RunSpec, batch_arithmetic
from mprl.errors import ConfigMismatch, Terminated
from mprl.negative_space import expect, require
from mprl.paths import RunPaths
from mprl.publish import export_onnx, hub_path, raise_if_sigterm_received, upload_run
from mprl.run_state import RunState, load_state, save_state
from mprl.tasks import load_task_env_cfg, load_task_rl_cfg
from mprl.wandb_writer import CLOCK_METRIC, ResumableWandbWriter, logger_cfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnvCfg

    from mprl.hub import HubPublisher
    from mprl.timed_runner import TimedOnPolicyRunner

log = logging.getLogger(__name__)

_REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WANDB_PROJECT = "mprl-humanoid-walk"  # .env.example default (plan, Task 14)
_ONNX_KEY = "onnx"  # state.uploads keys for errors that are not uploads
_POST_TRAINING_KEY = "post_training"
_WANDB_KEY = "wandb"  # the run's W&B fell back to offline mode (``ResumableWandbWriter.offline_fallback``)


@dataclass(frozen=True)
class TrainDeps:
    publisher: HubPublisher | None
    wandb_mode: str
    device: str
    # Diagnostics every 10 iterations: ~1 point per 5-10 s of training clock at 4096 envs, while the
    # extra activation forward pass and metric writes stay a small fraction of off-clock time.
    diagnostics_every_k: int = 10
    # The training clock's time source (TrainingClock's ``now``). Tests inject a fake so a run's cadence does
    # not depend on how fast the machine is; production keeps the default.
    now: Callable[[], float] = time.perf_counter


_SHA_RE: Final = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")  # SHA-1 or SHA-256 object names
_UNKNOWN_SHA: Final = "unknown"


def _git_dir(repo: Path) -> Path | None:
    """``repo/.git``, or the directory a worktree's ``.git`` file points to (``gitdir: <path>``)."""
    dot_git = repo / ".git"
    if dot_git.is_dir():
        return dot_git
    if not dot_git.is_file():
        return None
    line = dot_git.read_text().strip()
    if not line.startswith("gitdir:"):
        return None
    target = Path(line.removeprefix("gitdir:").strip())
    return target if target.is_absolute() else (repo / target).resolve()


def _resolve_ref(git: Path, ref: str) -> str | None:
    """A ref's SHA from its loose file (which wins, as in git) or ``packed-refs``; refs live in the common dir."""
    commondir = git / "commondir"
    common = (git / commondir.read_text().strip()).resolve() if commondir.is_file() else git
    for base in dict.fromkeys((git, common)):  # a worktree's own dir first, then the shared one
        loose = base / ref
        if loose.is_file():
            return loose.read_text().strip()
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            sha, _, name = line.partition(" ")
            if name.strip() == ref:
                return sha
    return None


def git_sha(repo: Path = _REPO_DIR) -> str:
    """HEAD of the checkout at ``repo``, read from ``.git`` (HEAD, then the loose ref or packed-refs).

    No ``git`` process: this runs inside every ``train_run``, and forking from a process holding torch's
    and wandb's threads can deadlock the child before it execs. "unknown" when there is no checkout (an
    image without ``.git``) or HEAD does not resolve to a SHA.
    """
    try:
        git = _git_dir(repo)
        if git is None or not (git / "HEAD").is_file():
            log.warning("no git checkout at %s; recording git_sha=%s", repo, _UNKNOWN_SHA)
            return _UNKNOWN_SHA
        head = (git / "HEAD").read_text().strip()
        sha = _resolve_ref(git, head.removeprefix("ref:").strip()) if head.startswith("ref:") else head
    except OSError as error:
        log.warning("could not read %s/.git (%s); recording git_sha=%s", repo, error, _UNKNOWN_SHA)
        return _UNKNOWN_SHA
    if sha is None or _SHA_RE.fullmatch(sha) is None:
        log.warning("HEAD in %s does not resolve to a SHA (%r); recording git_sha=%s", repo, sha, _UNKNOWN_SHA)
        return _UNKNOWN_SHA
    return sha


def _mint_suffix() -> str:
    return uuid.uuid4().hex[:8]


def wandb_run_id_for(
    state: RunState | None, latest: CheckpointRef | None, run_id: str, *, mint: Callable[[], str] = _mint_suffix
) -> str:
    """The stored W&B id when the run resumes (a ``state.json`` and a checkpoint), else a new one.

    Without a checkpoint the run trains from iteration 0, so continuing a crashed run's W&B id would mix
    its history with steps that restart at 0.
    """
    require(bool(run_id), "run_id must be non-empty")
    if state is not None and latest is not None:
        return state.wandb_run_id
    suffix = mint()
    require(bool(suffix), "minted W&B id suffix is empty")
    return f"{run_id}-{suffix}"


def failed_after_final_checkpoint(
    saved: Sequence[CheckpointRef], elapsed_s: float, budget_s: float, current_iteration: int
) -> bool:
    """Whether an error inside ``learn_for`` came after its final checkpoint (writer stop, ONNX export).

    Such an error must not discard a finished run, so it is recorded; any other is a training failure.
    The final checkpoint exists once the clock reached the budget and the last save is the iteration the
    runner is at.
    """
    require(budget_s > 0.0, f"budget_s must be positive, got {budget_s}")
    require(elapsed_s >= 0.0, f"elapsed_s must be non-negative, got {elapsed_s}")
    return bool(saved) and elapsed_s >= budget_s and saved[-1].iteration == current_iteration


def wandb_tags(spec: RunSpec) -> list[str]:
    """W&B tags ``n<5-digit num_envs>`` and ``seed<seed>``: the group stays the sweep, so all counts compare in one
    group, and a tag (or ``config.run.num_envs``) filters or groups by count inside it."""
    require(spec.num_envs > 0, f"num_envs must be positive, got {spec.num_envs}")
    require(spec.seed >= 0, f"seed must be non-negative, got {spec.seed}")
    return [f"n{spec.num_envs:05d}", f"seed{spec.seed}"]


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
    rl_cfg = load_task_rl_cfg(spec.task_id)
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
        environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
        wandb_run_id,
        spec.sweep,
        resume_from_iteration,
        mode,
        tags=wandb_tags(spec),
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
    env_cfg = load_task_env_cfg(spec.task_id)
    env_cfg.scene.num_envs = spec.num_envs
    env_cfg.seed = spec.seed
    rl_cfg = load_task_rl_cfg(spec.task_id)
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
    render_mode: str | None = None,
) -> Generator[TimedOnPolicyRunner]:
    """The real env (``resolve_cfgs``) wrapped in a fresh ``TimedOnPolicyRunner``; the env is closed on exit.

    With ``params_dir`` the env and agent configs are dumped there before the runner exists, as mjlab
    does: the runner mutates ``agent_cfg`` in place. With ``env_cfg`` that env config is built instead of
    the training one (the evaluator's play env); the agent config, and so the policy, is unchanged.
    ``render_mode`` is passed to the env (``"rgb_array"`` for ``env.render()``).
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    from mprl.timed_runner import TimedOnPolicyRunner

    training_env_cfg, agent_cfg = resolve_cfgs(spec, logger)
    env_cfg = training_env_cfg if env_cfg is None else env_cfg
    configure_torch_backends()  # as mjlab's train script
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    try:
        if params_dir is not None:
            dump_yaml(params_dir / "env.yaml", asdict(env_cfg))
            dump_yaml(params_dir / "agent.yaml", agent_cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg["clip_actions"])
        yield TimedOnPolicyRunner(wrapped, agent_cfg, str(log_dir), device)
    finally:
        env.close()


def release_device_memory(device: str) -> None:
    """Free a finished run's device memory before the next one builds its env.

    mjlab's managers point back at their env, so a finished runner/env is a reference cycle only ``gc`` frees
    (``env.close()`` frees nothing on the GPU, and mjlab suspends gc while it captures CUDA graphs); then the
    caching allocator returns the freed blocks. Off the training clock. The caller must hold no reference to
    the runner (nor an exception whose traceback pins its frames).
    """
    require(device == "cpu" or device.startswith("cuda"), f"unsupported device {device!r}")
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


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
    wandb_run_id = wandb_run_id_for(state, latest, spec.run_id)
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
        raise_if_sigterm_received()  # a Terminated caught inside training (by a library) still fails the run
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
            clock = TrainingClock(now=deps.now, sync=sync)
        else:
            infos = runner.load(str(latest.path))
            clock = TrainingClock.from_state_dict(infos["clock"], now=deps.now, sync=sync)
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
    diag, on_before_update, diag_on_iteration = diagnostics.for_runner(runner, deps.diagnostics_every_k)

    def on_iteration(it: int) -> None:
        raise_if_sigterm_received()  # stop at the iteration end even if a library swallowed the Terminated
        diag_on_iteration(it)
        writer = runner.logger.writer
        if writer is not None:  # off the clock: a float the clock already holds, no device sync
            writer.add_scalar(CLOCK_METRIC, clock.elapsed_minutes, it)

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
    except Terminated:
        raise  # SIGTERM is a failure exit even after the final checkpoint (cli: ``sigterm_raises``)
    except Exception as error:
        # After the final checkpoint learn_for only stops the writer and exports ONNX. A failure there
        # must not discard a finished run: record it. Anything earlier is a training failure.
        if not failed_after_final_checkpoint(saved, clock.elapsed_s, budget_s, runner.current_learning_iteration):
            raise
        log.warning("%s: post-training step failed after the final checkpoint: %s", spec.run_id, error)
        # The writer stop runs before the export, so its failure skipped the ONNX: export it here.
        errors = {_POST_TRAINING_KEY: f"{type(error).__name__}: {error}", **_wandb_notes(runner)}
        onnx_error = export_onnx(runner, saved[-1])
        if onnx_error is not None:
            errors[_ONNX_KEY] = onnx_error
        return saved[-1], clock.elapsed_s, errors
    finally:
        diag.detach()
    require(result.final == saved[-1], "learn_for's final checkpoint was not reported to on_checkpoint")
    return result.final, clock.elapsed_s, _wandb_notes(runner)


def _wandb_notes(runner: TimedOnPolicyRunner) -> dict[str, str]:
    """``{"wandb": "offline fallback after <error>"}`` when the run's writer fell back to offline, else ``{}``."""
    writer = runner.logger.writer
    if isinstance(writer, ResumableWandbWriter) and writer.offline_fallback is not None:
        return {_WANDB_KEY: f"offline fallback after {writer.offline_fallback}"}
    return {}
