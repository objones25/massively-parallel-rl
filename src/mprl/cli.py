"""The ``mprl`` command line (tyro subcommands). ``scripts/setup.sh`` and ``scripts/train.sh`` call it.

``.env`` (``$MPRL_ENV_FILE``, default the repo root's, written by ``setup``) is loaded at startup without
overriding variables already set.

Termination: ``train-run``, ``pilot`` and ``sweep`` build the pod's ``PodTerminator`` before anything else
(no key or no pod id: refuse, naming ``--no-terminate``). From then on one ``PodGuard`` terminates the pod
on any exit (success, a missing key, corrupt measurements, a crash, SIGTERM) except a cut plan refused at the
prompt and Ctrl-C at a terminal. SIGTERM (``kill``, ``docker stop``, a RunPod stop) is a failure: it raises
``Terminated`` in the main thread, so a run in progress is saved as failed, the pod is terminated (unless
``--no-terminate``) and the exit status is 1. ``results`` does the same only with ``--terminate``; ``setup``, ``preflight`` and
``throughput`` never terminate.
"""

from __future__ import annotations

import getpass
import logging
import os
import shutil
import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal, get_args

import httpx
import torch
import tyro
from dotenv import dotenv_values, load_dotenv
from huggingface_hub import HfApi

from mprl.budget import CostEstimate, load_measurements, save_measurements
from mprl.config import RunSpec, SweepSpec
from mprl.curves import METRICS
from mprl.env_file import parse_env_example, prompt_values, render_env, write_env
from mprl.errors import EnvDoesNotFit, MeasurementMissing, MissingSecret, MprlOperatingError, PreflightFailed
from mprl.evaluator import EvalConfig
from mprl.hub import HubPublisher
from mprl.negative_space import require
from mprl.paths import volume_root
from mprl.preflight import default_checks, run_checks
from mprl.publish import PodGuard, sigterm_raises
from mprl.results import build_results
from mprl.runpod import own_terminator
from mprl.sweep import run_pilot, run_sweep
from mprl.throughput import measure_eval_seconds, measure_throughput
from mprl.train import TrainDeps, train_run

log = logging.getLogger(__name__)

REPO_DIR: Final = Path(__file__).resolve().parents[2]
ENV_FILE: Final = REPO_DIR / ".env"
ENV_EXAMPLE: Final = REPO_DIR / ".env.example"
MEASUREMENTS: Final = "measurements.json"  # under the volume root: written by throughput, read by sweep
_RUNPOD_TIMEOUT_S: Final = 30.0
_DEVICE: Final = "cuda:0" if torch.cuda.is_available() else "cpu"
_NO_TERMINATE_HINT: Final = "refusing to start without auto-termination; pass --no-terminate to keep the pod running"

WandbMode = Literal["online", "offline", "disabled"]
Metric = Literal["vel_error_xy", "vel_error_yaw", "no_fall_fraction", "mean_time_to_fall_s", "mean_tracking_reward"]
METRIC_CHOICES: Final[tuple[str, ...]] = get_args(Metric)
require(set(METRIC_CHOICES) == set(METRICS), f"cli.Metric {METRIC_CHOICES} differs from curves.METRICS {METRICS}")


def _interactive() -> bool:
    return sys.stdin.isatty()


def _env_file() -> Path:
    return Path(os.environ.get("MPRL_ENV_FILE") or ENV_FILE)


@contextmanager
def _pod_guard(no_terminate: bool, hint: str = _NO_TERMINATE_HINT) -> Generator[PodGuard]:
    """Build this pod's terminator (key and pod id checked) first, then guard the block with it.

    While the block runs SIGTERM raises ``Terminated`` (``sigterm_raises``); from the block's end until the
    guard has terminated the pod a SIGTERM is ignored; then the previous handler is back.
    """
    with httpx.Client(timeout=_RUNPOD_TIMEOUT_S) as client:
        terminator = None
        if not no_terminate:
            try:
                terminator = own_terminator(os.environ, client)
            except MprlOperatingError as error:
                raise type(error)(f"{error}; {hint}") from error
        with sigterm_raises() as sigterm, PodGuard(terminator, interactive=_interactive()) as guard:
            try:
                yield guard
            finally:
                sigterm.shield()  # the guard's cleanup (summary, os.sync, DELETE) runs to the end


def _deps(device: str, wandb_mode: str) -> TrainDeps:
    """Hugging Face upload is part of every training command, so its keys are checked before training."""
    for name in ("HF_TOKEN", "HF_REPO_ID"):
        if not os.environ.get(name):
            raise MissingSecret(f"{name} is not set; run scripts/setup.sh")
    publisher = HubPublisher(HfApi(), os.environ["HF_REPO_ID"], private=True)
    return TrainDeps(publisher=publisher, wandb_mode=wandb_mode, device=device)


@dataclass(frozen=True)
class Setup:
    """Ask for the keys in .env.example, write .env (mode 0600), then run the preflight checks."""

    cpu: bool = False
    """Skip the CUDA and RunPod checks (local testing)."""

    def run(self) -> int:
        vars = parse_env_example(ENV_EXAMPLE.read_text())
        env_file = _env_file()
        existing = dotenv_values(env_file) if env_file.exists() else {}
        values = prompt_values(vars, existing, ask=input, ask_secret=getpass.getpass)
        write_env(env_file, render_env(vars, values), overwrite=True)
        print(f"wrote {env_file} (mode 0600)")
        return _preflight({**os.environ, **values}, require_gpu=not self.cpu, only=())


def _preflight(env: dict[str, str], *, require_gpu: bool, only: Sequence[str]) -> int:
    checks = [c for c in default_checks(env, require_gpu) if not only or c[0] in only]
    if not checks:
        raise PreflightFailed(f"--only {' '.join(only)} selects no check (--cpu skips cuda and runpod)")
    results = run_checks(checks)
    for result in results:
        print(f"{'PASS' if result.ok else 'FAIL'} {result.name}: {result.detail}")
    return 0 if all(r.ok for r in results) else 1


@dataclass(frozen=True)
class Preflight:
    """Check the GPU, the env, W&B, Hugging Face, RunPod and the volume; exit 1 if any check fails."""

    cpu: bool = False
    """Skip the CUDA and RunPod checks; the env check runs on CPU."""
    only: tuple[Literal["cuda", "env", "wandb", "hf", "runpod", "volume"], ...] = ()
    """Run only these checks (default: all)."""

    def run(self) -> int:
        return _preflight(dict(os.environ), require_gpu=not self.cpu, only=self.only)


@dataclass(frozen=True)
class Throughput:
    """Measure env steps/s and fit per count, and one checkpoint evaluation's time; write measurements.json."""

    counts: tuple[int, ...] = SweepSpec("main").env_counts
    iterations: int = 20
    device: str = _DEVICE

    def run(self) -> int:
        root = volume_root(os.environ)
        work = root / "throughput"
        if work.exists():
            shutil.rmtree(work)  # scratch from an earlier measurement; its checkpoints must not be reused
        m = measure_throughput(self.counts, self.iterations, self.device, work_dir=work, eval_seconds_per_checkpoint=0.0)
        fitting = [n for n in self.counts if m.fits[n]]
        if not fitting:
            raise EnvDoesNotFit(f"no env count in {list(self.counts)} fits on {self.device}")
        # EvalConfig(): the config `mprl results` evaluates with.
        m = replace(m, eval_seconds_per_checkpoint=measure_eval_seconds(work, fitting[0], self.device, EvalConfig()))
        save_measurements(root / MEASUREMENTS, m)
        print(f"wrote {root / MEASUREMENTS}: {m}")
        return 0


@dataclass(frozen=True)
class TrainRun:
    """Train (or resume) one run, upload it, then terminate the pod."""

    sweep: str
    num_envs: int
    seed: int
    minutes: float
    no_terminate: bool = False
    """Leave the pod running when training ends."""
    device: str = _DEVICE
    wandb_mode: WandbMode = "online"

    def run(self) -> int:
        with _pod_guard(self.no_terminate) as guard:
            deps = _deps(self.device, self.wandb_mode)
            spec = RunSpec(self.sweep, self.num_envs, self.seed, self.minutes)
            guard.states.append(train_run(spec, volume_root(os.environ), deps))
        return 0


@dataclass(frozen=True)
class Pilot:
    """Train the pilot (the long run's first 45 minutes), then terminate the pod."""

    name: str = "main"
    """Sweep the pilot belongs to (``sweep --reuse-pilot`` with the same name can reuse it)."""
    no_terminate: bool = False
    device: str = _DEVICE
    wandb_mode: WandbMode = "online"

    def run(self) -> int:
        with _pod_guard(self.no_terminate) as guard:
            deps = _deps(self.device, self.wandb_mode)
            guard.states.append(run_pilot(SweepSpec(self.name), volume_root(os.environ), deps, terminator=None))
        return 0


def _confirm(est: CostEstimate, cuts: tuple[str, ...], *, yes: bool) -> bool:
    print(f"The sweep fits the cap only with cuts: {', '.join(cuts)}\n{est.describe()}")
    if yes:
        print("--yes: training with these cuts")
        return True
    if not _interactive():
        print("no terminal to confirm the cuts; re-run with --yes to accept them")
        return False
    return input("Train with these cuts? [y/N] ").strip().lower() in {"y", "yes"}


@dataclass(frozen=True)
class Sweep:
    """Run every (count, seed) of the sweep within the budget, then the long run; then terminate the pod."""

    name: str
    reuse_pilot: bool = False
    """Let a complete pilot on the same code stand in for the long-run count's seed-0 run."""
    no_terminate: bool = False
    yes: bool = False
    """Accept the budget cuts without asking (they are printed first)."""
    device: str = _DEVICE
    wandb_mode: WandbMode = "online"

    def run(self) -> int:
        root = volume_root(os.environ)
        with _pod_guard(self.no_terminate) as guard:
            deps = _deps(self.device, self.wandb_mode)
            if not (root / MEASUREMENTS).is_file():
                raise MeasurementMissing(f"{root / MEASUREMENTS} does not exist; run `mprl throughput` first")
            guard.states += run_sweep(
                SweepSpec(self.name),
                root,
                deps,
                load_measurements(root / MEASUREMENTS),
                confirm=lambda est, cuts: _confirm(est, cuts, yes=self.yes),
                terminator=None,  # the guard terminates, once, on every exit
                reuse_pilot=self.reuse_pilot,
            )
            if not guard.states:
                print(f"sweep {self.name}: cuts not confirmed; nothing trained")
                if _interactive() and not self.yes:
                    guard.keep("the cut plan was refused at the prompt")
                return 1
        return 0


@dataclass(frozen=True)
class Results:
    """Evaluate a finished sweep and build its CSV, learning curves, race video and model card; publish them."""

    name: str
    primary_metric: Metric
    """Metric on the learning-curve y-axis."""
    terminate: bool = False
    """Terminate the pod afterwards (off by default: results usually run interactively)."""
    device: str = _DEVICE
    wandb_mode: WandbMode = "online"

    def run(self) -> int:
        with _pod_guard(not self.terminate, "omit --terminate to keep the pod running"):
            deps = _deps(self.device, self.wandb_mode)
            out = build_results(
                SweepSpec(self.name),
                volume_root(os.environ),
                primary_metric=self.primary_metric,
                publisher=deps.publisher,
                wandb_mode=self.wandb_mode,
                device=self.device,
            )
            print(f"results in {out}")
        return 0


COMMANDS: Final = {
    "setup": Setup,
    "preflight": Preflight,
    "throughput": Throughput,
    "train-run": TrainRun,
    "pilot": Pilot,
    "sweep": Sweep,
    "results": Results,
}


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv(_env_file(), override=False)
    command = tyro.extras.subcommand_cli_from_dict(
        COMMANDS, prog="mprl", args=argv, config=(tyro.conf.FlagCreatePairsOff,)
    )
    try:
        return command.run()
    except MprlOperatingError as error:
        notes = "".join(f"\n  {note}" for note in getattr(error, "__notes__", []))
        print(f"mprl: {type(error).__name__}: {error}{notes}", file=sys.stderr)
        return 1
