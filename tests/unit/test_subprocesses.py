"""The test launcher (``support.Subprocesses``) stays on CPython's posix_spawn path, never a userland fork().

CPython 3.12 (``subprocess.Popen._execute_child``) uses ``os.posix_spawn`` only without
``start_new_session``, ``process_group``, ``preexec_fn``, ``pass_fds``, ``cwd``, uid/gid/umask, with
``close_fds=False`` and an executable that has a directory part. A fork from the multithreaded pytest
process (torch, wandb, GL threads) can deadlock the child before exec. These tests record the keyword
arguments the launcher passes, with a fake ``Popen``; no process is started.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from support import Subprocesses

ARGV = ["/bin/echo", "hi"]
ENV = {"PATH": "/bin"}
POSIX_SPAWN_BLOCKERS = ("start_new_session", "process_group", "preexec_fn", "pass_fds", "cwd", "user", "group", "umask")


class FakeChild:
    """Records the launch and the cleanup calls; with ``hang``, ``communicate`` times out until ``kill``."""

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        self.argv, self.kwargs = argv, kwargs
        self.calls: list[str] = []
        self.returncode: int | None = None
        self.hang = False

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        self.calls.append(f"communicate({timeout})")
        if self.hang and "kill" not in self.calls:  # a child that ignores SIGTERM
            raise subprocess.TimeoutExpired(self.argv, timeout or 0.0)
        self.returncode = 0
        return "out", "err"

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append(f"wait({timeout})")
        self.returncode = -9
        return -9

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")


class FakePopen:
    def __init__(self, *, hang: bool = False) -> None:
        self.children: list[FakeChild] = []
        self.hang = hang

    def __call__(self, argv: list[str], **kwargs: Any) -> FakeChild:
        child = FakeChild(argv, **kwargs)
        child.hang = self.hang
        self.children.append(child)
        return child


def _posix_spawn_blockers(child: FakeChild) -> list[str]:
    return [k for k in POSIX_SPAWN_BLOCKERS if k in child.kwargs]


def test_run_launch_kwargs_allow_posix_spawn() -> None:
    popen = FakePopen()

    Subprocesses(popen=popen).run(ARGV, env=ENV, timeout_s=5.0)

    (child,) = popen.children
    assert child.kwargs["close_fds"] is False
    assert child.kwargs["stdin"] is subprocess.DEVNULL
    assert _posix_spawn_blockers(child) == []
    assert child.argv == ARGV


def test_spawn_launch_kwargs_allow_posix_spawn(tmp_path: Path) -> None:
    popen = FakePopen()

    with (tmp_path / "log").open("w") as log:
        Subprocesses(popen=popen).spawn(ARGV, env=ENV, stdout=log)

    (child,) = popen.children
    assert child.kwargs["close_fds"] is False
    assert child.kwargs["stdin"] is subprocess.DEVNULL
    assert _posix_spawn_blockers(child) == []
    assert child.argv == ARGV


def test_posix_spawn_is_available_to_popen_here() -> None:
    assert subprocess._USE_POSIX_SPAWN is True  # pyright: ignore[reportAttributeAccessIssue]  CPython detail


def test_relative_executable_is_refused() -> None:
    procs = Subprocesses(popen=FakePopen())

    with pytest.raises(ValueError, match=r"^executable must be an absolute path"):
        procs.run(["echo", "hi"], env=ENV, timeout_s=5.0)


def test_timeout_terminates_then_kills_and_reaps() -> None:
    popen = FakePopen(hang=True)
    procs = Subprocesses(popen=popen)

    with pytest.raises(subprocess.TimeoutExpired):
        procs.run(ARGV, env=ENV, timeout_s=5.0)

    (child,) = popen.children
    assert child.calls[0] == "communicate(5.0)"
    assert child.calls[1] == "terminate"
    assert "kill" in child.calls[2:]
    assert child.returncode is not None  # reaped


def test_close_kills_children_still_running() -> None:
    popen = FakePopen()
    procs = Subprocesses(popen=popen)
    child = procs.spawn(ARGV, env=ENV, stdout=subprocess.DEVNULL)

    procs.close()

    assert child.calls[0] == "terminate"
    assert child.returncode is not None
