"""After-training side effects: ONNX export, Hugging Face upload of a run, and pod termination.

Every failure here is recorded as text (in ``state.uploads``) rather than raised, so a full volume or a
Hub outage can never keep ``finish_and_terminate`` from running and the pod from being deleted.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub.errors import HfHubHTTPError

from mprl.errors import UploadFailed
from mprl.negative_space import require

if TYPE_CHECKING:
    from mprl.checkpoints import CheckpointRef
    from mprl.config import RunSpec
    from mprl.hub import HubPublisher
    from mprl.paths import RunPaths
    from mprl.run_state import RunState
    from mprl.runpod import PodTerminator
    from mprl.timed_runner import TimedOnPolicyRunner

log = logging.getLogger(__name__)


def hub_path(spec: RunSpec) -> str:
    return f"runs/{spec.sweep}/{spec.run_id}"


def onnx_path(checkpoint: CheckpointRef) -> Path:
    # MjlabOnPolicyRunner._get_export_paths: "<checkpoint dir>/<checkpoint dir name>.onnx"
    return checkpoint.path.parent / f"{checkpoint.path.parent.name}.onnx"


def export_onnx(runner: TimedOnPolicyRunner, final: CheckpointRef) -> str | None:
    """Export the runner's policy (which must hold ``final``'s weights) next to it; None or the error text.

    Always re-exports: an ONNX file left by an earlier budget segment belongs to an older checkpoint.
    """
    require(runner.current_learning_iteration == final.iteration, "runner does not hold the final checkpoint")
    try:
        runner.export_policy_to_onnx(str(final.path.parent), onnx_path(final).name)
    except Exception as error:
        log.warning("ONNX export for %s failed: %s", final.path.name, error)
        return f"{type(error).__name__}: {error}"
    return None


def upload_run(publisher: HubPublisher, paths: RunPaths, final: CheckpointRef) -> str:
    """Stage final checkpoint, ONNX, params and state.json, upload them; "ok" or the error text.

    Staging I/O (volume full, permissions) is recorded like a Hub failure. The staging dir is removed
    after the attempt either way.
    """
    staging = paths.run_dir / "upload"
    try:
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        shutil.copy2(final.path, staging / final.path.name)
        if onnx_path(final).is_file():
            shutil.copy2(onnx_path(final), staging / onnx_path(final).name)
        if (paths.run_dir / "params").is_dir():
            shutil.copytree(paths.run_dir / "params", staging / "params")
        shutil.copy2(paths.state_file, staging / paths.state_file.name)
        publisher.ensure_repo()
        publisher.upload_folder(staging, hub_path(paths.spec), f"{paths.spec.run_id}: {final.path.name}")
    except (UploadFailed, HfHubHTTPError, OSError) as error:
        log.error("upload of %s to %s failed: %s", paths.spec.run_id, hub_path(paths.spec), error)
        return f"{type(error).__name__}: {error}"
    finally:
        _remove_staging(staging)
    return "ok"


def _remove_staging(staging: Path) -> None:
    try:
        if staging.exists():
            shutil.rmtree(staging)
    except OSError as error:
        log.warning("could not remove upload staging dir %s: %s", staging, error)


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
