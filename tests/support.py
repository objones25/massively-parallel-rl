"""Helpers shared by the smoke tests and their conftest (importable: ``tests`` is on pytest's pythonpath).

Only constants, plain functions and types live here; fixtures stay in the conftest files.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any

import httpx

from mprl.config import TASK_ID, RunSpec
from mprl.negative_space import expect
from mprl.paths import RunPaths
from mprl.run_state import RunState, load_state
from mprl.runpod import PodTerminator

# Environment variable prefixes owned by the project; tests/conftest.py strips them before every test.
APP_ENV_PREFIXES = ("MPRL_", "WANDB_", "HF_", "RUNPOD_")
DEFAULT_MARKEXPR = "not smoke and not network and not gpu"  # must equal the -m in pyproject addopts
POD = "abc123podid"

NUM_ENVS = 4
STEPS_PER_ITERATION = 24  # num_steps_per_env in the shipped G1 PPO config
LONG_S = 3600.0  # a budget / checkpoint interval no smoke run reaches
EVERY_MIN = 1.0 / 60.0  # train_run tests: a checkpoint every 1 s of training clock


def run_spec(budget_s: float, *, every_minutes: float = EVERY_MIN, num_envs: int = NUM_ENVS, seed: int = 0) -> RunSpec:
    """The ``main`` sweep's run with a budget in seconds (train_run takes minutes)."""
    return RunSpec("main", num_envs, seed, budget_s / 60.0, every_minutes)


def make_env_runner(
    log_dir: Path,
    *,
    runner_cls: type | None = None,
    logger: object = "tensorboard",
    play: bool = False,
    num_envs: int = NUM_ENVS,
    render_mode: str | None = None,
) -> tuple[Any, Any]:
    """(env, runner) for the shipped G1 config, seeded with its own seed (as mjlab's train script)."""
    import mjlab.tasks  # noqa: F401  (registers the tasks)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    from mprl.timed_runner import TimedOnPolicyRunner

    runner_cls = TimedOnPolicyRunner if runner_cls is None else runner_cls
    env_cfg = load_env_cfg(TASK_ID, play=play)
    env_cfg.scene.num_envs = num_envs
    rl_cfg = load_rl_cfg(TASK_ID)
    env_cfg.seed = rl_cfg.seed
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=render_mode)
    agent_cfg = asdict(rl_cfg)
    agent_cfg["logger"] = logger
    wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    return env, runner_cls(wrapped, agent_cfg, str(log_dir), "cpu")


@dataclass(frozen=True)
class CompletedRun:
    volume: Path  # the train_run root
    wandb_dir: Path  # WANDB_DIR of the run (its offline-run-* directory is under wandb_dir / "wandb")
    state: RunState  # what train_run returned


def _uploads(state: RunState) -> object:
    return state.uploads


def runpod_terminator(
    paths: RunPaths, events: list[tuple[str, object]], snapshot: Callable[[RunState], object] = _uploads
) -> PodTerminator:
    """A real ``PodTerminator`` over a RunPod double: RUNNING until DELETE, then 404.

    The DELETE appends ``("DELETE", snapshot(state))`` with the ``state.json`` read from disk at that
    moment (by default its uploads), so a test can assert what was durable before the pod went away.
    """

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            on_disk = expect(load_state(paths.state_file), "no state.json at DELETE time")
            events.append(("DELETE", snapshot(on_disk)))
            return httpx.Response(204)
        if any(name == "DELETE" for name, _ in events):
            return httpx.Response(404)
        return httpx.Response(200, json={"id": POD, "status": "RUNNING"})

    client = httpx.Client(transport=httpx.MockTransport(handle))
    return PodTerminator(client, "key", POD, base_url="https://api.test/v2")


TERMINATE_WAIT_S = 10.0  # after terminate(), before kill()
KILL_WAIT_S = 10.0  # after kill(), the reap itself is bounded too


class Subprocesses:
    """Launch processes from a test on CPython's ``posix_spawn`` path; bounded waits; survivors are killed.

    ``posix_spawn`` (CPython 3.12 ``Popen._execute_child``) needs: an executable with a directory part,
    ``close_fds=False`` (pipes are non-inheritable anyway, PEP 446), no ``cwd``, no ``start_new_session`` /
    ``process_group`` / ``preexec_fn`` / ``pass_fds`` / uid / gid / umask. Anything else makes CPython
    ``fork()`` the multithreaded pytest process, and the child can deadlock before exec.
    ``tests/unit/test_subprocesses.py`` pins these keyword arguments.

    A child is tracked from the moment ``Popen`` returns. If ``Popen`` itself raises, no child is left to
    track: CPython reports the spawn failure only after the child is gone, so the error simply propagates.
    """

    def __init__(self, popen: Callable[..., Any] = subprocess.Popen) -> None:
        self._popen = popen
        self._started: list[Any] = []

    def _launch(self, argv: Sequence[str], **kwargs: Any) -> Any:
        if not Path(argv[0]).is_absolute():
            raise ValueError(f"executable must be an absolute path (posix_spawn needs a directory part): {argv[0]!r}")
        child = self._popen(list(argv), stdin=subprocess.DEVNULL, close_fds=False, **kwargs)
        self._started.append(child)
        return child

    def spawn(self, argv: Sequence[str], *, env: dict[str, str], stdout: IO[Any] | int) -> Any:
        """Start ``argv`` with stdout (and stderr) to ``stdout``; the caller waits, ``close`` cleans up."""
        return self._launch(argv, env=env, stdout=stdout, stderr=subprocess.STDOUT)

    def run(self, argv: Sequence[str], *, env: dict[str, str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        """Run ``argv`` to completion within ``timeout_s``; on timeout stop it (``stop``) and re-raise."""
        child = self._launch(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out, err = child.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.stop(child)
            raise
        return subprocess.CompletedProcess(list(argv), child.returncode, out, err)

    @staticmethod
    def stop(child: Any) -> None:
        """``terminate()``, then ``kill()`` if it is still running after TERMINATE_WAIT_S; reap it (bounded)."""
        child.terminate()
        try:
            child.communicate(timeout=TERMINATE_WAIT_S)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate(timeout=KILL_WAIT_S)

    def close(self) -> None:
        for child in self._started:
            if child.poll() is None:
                self.stop(child)
