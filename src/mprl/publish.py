"""After-training side effects: ONNX export, Hugging Face upload of a run, and pod termination.

Every failure here is recorded as text (in ``state.uploads``) rather than raised, so a full volume or a
Hub outage can never keep ``finish_and_terminate`` from running and the pod from being deleted.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Literal

from mprl.errors import Terminated
from mprl.hub import UPLOAD_ERRORS
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
    except Terminated:
        raise  # SIGTERM is a failure exit, never a recorded export error
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
    except UPLOAD_ERRORS as error:
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


class PodGuard:
    """Terminate the pod when the guarded block exits, however it exits (success, any exception).

    Two exits keep the pod: ``keep(reason)`` was called (the user refused at a prompt), and a
    ``KeyboardInterrupt`` while ``interactive`` (the user is at the terminal). An error is logged with its
    traceback before terminating; if termination then fails too, the original exception propagates with
    the termination failure attached as a note (``add_note``) and logged. ``states`` is what the block
    trained, for ``finish_and_terminate``'s summary. With ``terminator=None`` only the summary and the
    flush happen.
    """

    def __init__(self, terminator: PodTerminator | None, *, interactive: bool) -> None:
        self.states: list[RunState] = []
        self._terminator = terminator
        self._interactive = interactive
        self._keep_reason = ""

    def keep(self, reason: str) -> None:
        require(bool(reason), "keeping the pod needs a reason")
        self._keep_reason = reason

    def __enter__(self) -> PodGuard:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> Literal[False]:
        if isinstance(exc, KeyboardInterrupt) and self._interactive:
            log.warning("interrupted at the terminal; leaving the pod running")
            return False
        if self._keep_reason:
            log.info("leaving the pod running: %s", self._keep_reason)
            return False
        if exc is not None:
            log.error("stopping after an error; terminating the pod", exc_info=(exc_type or type(exc), exc, tb))
        try:
            finish_and_terminate(self.states, self._terminator)
        except Exception as error:
            if exc is None:
                raise
            note = f"pod termination also failed: {type(error).__name__}: {error}"
            log.error(note)
            exc.add_note(note)
        return False



class SigtermHandler:
    """The SIGTERM handler ``sigterm_raises`` installs: the first SIGTERM raises ``Terminated``, once.

    Python runs it in the main thread between bytecodes, so the exception surfaces wherever the main thread
    is (env step, PPO update, an upload) and unwinds through ``train_run`` (``state.json`` -> failed) into the
    ``PodGuard``. Later SIGTERMs, and any after ``shield()`` (the guarded block has ended and its cleanup is
    running), are logged and ignored: ``docker stop`` can repeat it, and the state save and the pod DELETE it
    would interrupt are what it is asking for.

    ``received`` stays set, so a ``Terminated`` that third-party code swallowed (a library catching
    ``Exception``, e.g. ``logging.Handler.handleError`` around ``emit``) is raised again at the next
    checkpoint: ``raise_if_sigterm_received`` (every iteration end, after training) and the guarded block's end.
    """

    def __init__(self) -> None:
        self.received = False
        self._shielded = False

    def shield(self) -> None:
        self._shielded = True

    def __call__(self, signum: int, _frame: object) -> None:
        require(signum == signal.SIGTERM, f"SigtermHandler installed for signal {signum}")
        if self._shielded or self.received:
            log.warning("SIGTERM again while stopping; finishing the cleanup first")
            return
        self.received = True
        raise Terminated(f"received SIGTERM (signal {signum}); stopping")

    def raise_if_received(self) -> None:
        """Raise ``Terminated`` again if a SIGTERM came in: its first ``Terminated`` was caught on the way."""
        if self.received:
            raise Terminated(
                f"received SIGTERM (signal {int(signal.SIGTERM)}) earlier, and something caught its Terminated; stopping"
            )


def raise_if_sigterm_received() -> None:
    """``SigtermHandler.raise_if_received`` of the installed handler; a no-op outside ``sigterm_raises``."""
    handler = signal.getsignal(signal.SIGTERM)
    if isinstance(handler, SigtermHandler):
        handler.raise_if_received()


@contextmanager
def sigterm_raises() -> Generator[SigtermHandler]:
    """Make SIGTERM a ``Terminated`` operating error for the block; restore the previous handler on exit.

    Without it Python's default SIGTERM action ends the process without running ``finally`` or ``__exit__``:
    no failed ``state.json``, no pod termination. Signal handlers can only be set from the main thread.
    """
    require(threading.current_thread() is threading.main_thread(), "SIGTERM handler set off the main thread")
    handler = SigtermHandler()
    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield handler
    finally:
        # None: the previous handler was not installed from Python; the default is the closest equivalent.
        signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
