"""Sweep orchestration: exclusions, the budget gate, pilot reuse, the runs, then pod termination.

``run_sweep`` in order:

1. A count absent from ``m.fits`` raises ``MeasurementMissing`` (unmeasured counts never train).
   Counts with ``m.fits[n] is False`` are removed and the reason (CUDA OOM, plus the steps/s measured
   for the counts that fit) is recorded in ``sweep_results_dir/sweep_manifest.json``. If 64 or the
   long-run count does not fit, ``EnvDoesNotFit`` is raised instead.
2. ``plan_within_budget`` applies the cut order at ``price_per_hour`` (``--price-per-hour``); the estimate is
   printed and the price recorded in the manifest. If any cut was applied,
   ``confirm(estimate, cuts)`` must return True, else nothing trains and ``[]`` is returned. The
   estimate counts the whole project (setup and the pilot's 45 minutes included), because the $40
   cap is on total spend; it also still counts a reused pilot run, so it errs high.
3. With ``reuse_pilot``, the pilot (``pilot_spec``: the long run's first 45 minutes) stands in for the
   sweep's ``n=long_run_envs, s=0`` run when the sweep's budget is at most 45 minutes and (a) HEAD is known and equals the pilot's ``git_sha`` and
   the sha the sweep was first started on, (b) the pilot's ``config_hash`` equals that sweep run's hash
   with the sweep name substituted (every field but the sweep name and budget equal), and (c) the
   pilot is ``complete`` with at least 45 training minutes, or this same reuse was already recorded
   (the long run, resuming the pilot past 45 minutes, may have crashed). The reuse is written to
   ``sweep_results_dir/pilot_reuse.json`` (results read that run's checkpoints up to 45 minutes) and
   the run is skipped. When no reuse is accepted, a ``pilot_reuse.json`` left by an earlier
   invocation is removed, since that run now trains for real.
4. Every remaining run trains through ``train_run`` (count-major), then the long run, which resumes
   the pilot, inside a ``PodGuard``; ``release_device_memory`` runs after each one (the finished
   runner/env is a gc-only cycle holding GPU memory): a crash still terminates the pod, then re-raises (a failed
   termination is attached to it as a note).

Refusing the confirmation or exceeding the cap returns or raises *before* any training and leaves the
pod running, since the user is at the prompt.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any, Final

from mprl.budget import NEVER_DROPPED, PRICE_PER_HOUR, CostEstimate, Measurements, plan_within_budget
from mprl.config import CHECKPOINT_EVERY_MINUTES, RunSpec, SweepSpec
from mprl.errors import EnvDoesNotFit, InvalidInput, MeasurementMissing
from mprl.negative_space import require
from mprl.paths import RunPaths, sweep_results_dir
from mprl.publish import PodGuard
from mprl.run_state import RunState, load_state, write_json_atomic
from mprl.train import git_sha, release_device_memory, train_run

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mprl.runpod import PodTerminator
    from mprl.train import TrainDeps

log = logging.getLogger(__name__)

PILOT_MINUTES: Final = 45.0
_UNKNOWN_SHA: Final = "unknown"  # git_sha() without git or a checkout
_MANIFEST: Final = "sweep_manifest.json"
_PILOT_REUSE: Final = "pilot_reuse.json"


def configured_sweep(
    name: str, *, seeds: tuple[int, ...], budget_minutes: float, long_run_minutes: float
) -> SweepSpec:
    """``SweepSpec(name)`` with the command line's overrides; ``InvalidInput`` naming the option for a bad value.

    The budget must cover one checkpoint interval; the long run must cover the pilot it resumes.
    """
    every = SweepSpec(name).checkpoint_every_minutes
    if not seeds:
        raise InvalidInput("--seeds must name at least one seed")
    if len(set(seeds)) != len(seeds):
        raise InvalidInput(f"--seeds must not repeat a seed, got {list(seeds)}")
    if any(seed < 0 for seed in seeds):
        raise InvalidInput(f"--seeds must be non-negative, got {list(seeds)}")
    if not (math.isfinite(budget_minutes) and budget_minutes >= every):
        raise InvalidInput(
            f"--budget-minutes must be a finite number of at least {every:g} (one checkpoint interval), "
            f"got {budget_minutes}"
        )
    if not (math.isfinite(long_run_minutes) and long_run_minutes >= PILOT_MINUTES):
        raise InvalidInput(
            f"--long-run-minutes must be a finite number of at least {PILOT_MINUTES:g} (the pilot it resumes), "
            f"got {long_run_minutes}"
        )
    return replace(SweepSpec(name), seeds=seeds, budget_minutes=budget_minutes, long_run_minutes=long_run_minutes)


def configured_run(sweep: str, *, num_envs: int, seed: int, minutes: float) -> RunSpec:
    """``RunSpec`` for ``train-run``; ``InvalidInput`` naming the option for a bad value (checked before the pod
    guard, so a typo keeps the pod instead of deleting it on a ``CheckFailed``)."""
    every = CHECKPOINT_EVERY_MINUTES  # train-run uses RunSpec's default interval
    if not sweep:
        raise InvalidInput("--sweep must be non-empty")
    if num_envs <= 0:
        raise InvalidInput(f"--num-envs must be positive, got {num_envs}")
    if seed < 0:
        raise InvalidInput(f"--seed must be non-negative, got {seed}")
    if not (math.isfinite(minutes) and minutes >= every):
        raise InvalidInput(
            f"--minutes must be a finite number of at least {every:g} (one checkpoint interval), got {minutes}"
        )
    spec = RunSpec(sweep, num_envs, seed, minutes)
    require(spec.checkpoint_every_minutes == every, "train-run's checkpoint interval is not RunSpec's default")
    return spec


def pilot_spec(sweep: SweepSpec) -> RunSpec:
    """The pilot: the first 45 training minutes of ``sweep.long_run()`` (same run id and config hash)."""
    return replace(sweep.long_run(), budget_minutes=PILOT_MINUTES)


def run_pilot(sweep: SweepSpec, root: Path, deps: TrainDeps, *, terminator: PodTerminator | None) -> RunState:
    """Train the pilot, then terminate the pod through ``PodGuard`` (also on a crash, which re-raises)."""
    with PodGuard(terminator, interactive=False) as guard:
        guard.states.append(train_run(pilot_spec(sweep), root, deps))
        release_device_memory(deps.device)
    return guard.states[0]


def run_sweep(
    sweep: SweepSpec,
    root: Path,
    deps: TrainDeps,
    m: Measurements,
    *,
    confirm: Callable[[CostEstimate, tuple[str, ...]], bool],
    terminator: PodTerminator | None,
    reuse_pilot: bool,
    price_per_hour: float = PRICE_PER_HOUR,
) -> list[RunState]:
    """Run ``sweep`` within the budget (steps in the module docstring); the states of the runs trained."""
    results = sweep_results_dir(root, sweep.name)
    sha = git_sha()
    prior = _read_json(results / _MANIFEST)
    started_sha = prior.get("git_sha", sha)
    fitting, excluded = _exclude_unfit(sweep, m)
    planned, cuts, est = plan_within_budget(fitting, m, price_per_hour=price_per_hour)
    print(f"sweep {sweep.name}: estimated cost (cuts: {', '.join(cuts) or 'none'})\n{est.describe()}")
    if cuts and not confirm(est, cuts):
        log.info("sweep %s: cuts %s not confirmed; nothing trained", sweep.name, cuts)
        return []

    reused: dict[str, dict[str, Any]] = {}
    rejected = ""
    if reuse_pilot:
        reused, rejected = _pilot_reuse(planned, root, sha, started_sha)
        if rejected:
            print(f"sweep {sweep.name}: pilot not reused: {rejected}")
    write_json_atomic(
        results / _MANIFEST,
        {
            "requested": asdict(sweep),
            "planned": asdict(planned),
            "excluded": {str(n): reason for n, reason in excluded.items()},
            "cuts": list(cuts),
            "estimate": {"lines": [list(line) for line in est.lines], "hours": est.total_hours, "usd": est.total_usd},
            "measurements": asdict(m),
            "price_per_hour": price_per_hour,
            "git_sha": started_sha,
            "pilot_reuse_rejected": rejected,
        },
    )
    if reused:
        write_json_atomic(results / _PILOT_REUSE, reused)
    else:
        # A reuse recorded by an earlier invocation no longer holds: this run trains for real, and a
        # stale file would give results two sources for it.
        (results / _PILOT_REUSE).unlink(missing_ok=True)

    specs = [spec for spec in planned.runs() if spec.run_id not in reused] + [planned.long_run()]
    require(len(specs) == len(planned.runs()) - len(reused) + 1, f"reused ids {sorted(reused)} are not sweep runs")
    require(all(spec.num_envs in planned.env_counts for spec in specs), "a run uses an excluded or cut env count")
    with PodGuard(terminator, interactive=False) as guard:
        for spec in specs:
            guard.states.append(train_run(spec, root, deps))
            release_device_memory(deps.device)  # the finished runner/env cycle, before the next env is built
    return guard.states


def _exclude_unfit(sweep: SweepSpec, m: Measurements) -> tuple[SweepSpec, dict[int, str]]:
    """``sweep`` without the counts measured not to fit, and the reason for each one removed.

    Raises ``MeasurementMissing`` if a count was never measured: whether it fits is unknown.
    """
    missing = sorted(n for n in sweep.env_counts if n not in m.fits)
    if missing:
        raise MeasurementMissing(
            f"sweep {sweep.name}: env counts {missing} have no throughput measurement; "
            f"measured counts are {sorted(m.fits)}. Re-run the throughput test for them."
        )
    fit_sps = ", ".join(f"{n}: {sps:.0f}" for n, sps in sorted(m.steps_per_second.items()))
    excluded = {
        n: (
            f"num_envs={n} does not fit: CUDA out of memory in the throughput test; "
            f"measured env steps/s for the counts that fit: {{{fit_sps}}}"
        )
        for n in sweep.env_counts
        if m.fits.get(n) is False
    }
    for required in (NEVER_DROPPED, sweep.long_run_envs):
        if required in excluded:
            raise EnvDoesNotFit(f"sweep {sweep.name} cannot run without num_envs={required}: {excluded[required]}")
    for reason in excluded.values():
        log.warning("sweep %s: excluding %s", sweep.name, reason)
    return replace(sweep, env_counts=tuple(n for n in sweep.env_counts if n not in excluded)), excluded


def _pilot_reuse(planned: SweepSpec, root: Path, sha: str, started_sha: str) -> tuple[dict[str, dict[str, Any]], str]:
    """({sweep run id: reuse entry}, "") when the pilot may stand in for that run, else ({}, why not)."""
    target = next((s for s in planned.runs() if s.num_envs == planned.long_run_envs and s.seed == 0), None)
    if target is None:
        return {}, f"the sweep has no n={planned.long_run_envs} s=0 run"
    pilot = pilot_spec(planned)
    paths = RunPaths(root, pilot)
    expected_hash = replace(target, sweep=pilot.sweep).config_hash()
    entry: dict[str, Any] = {
        "source_run_dir": str(paths.run_dir),
        "max_minutes": PILOT_MINUTES,
        "git_sha": sha,
        "config_hash": expected_hash,
    }
    state = load_state(paths.state_file)
    recorded = _read_json(sweep_results_dir(root, planned.name) / _PILOT_REUSE).get(target.run_id) == entry
    reason = ""
    if planned.budget_minutes > PILOT_MINUTES:
        reason = f"the sweep's {planned.budget_minutes:g} min budget is longer than the pilot's {PILOT_MINUTES:g} min"
    elif sha == _UNKNOWN_SHA:
        reason = "HEAD is unknown, so the code cannot be shown unchanged"
    elif started_sha != sha:
        reason = f"the sweep started on {started_sha[:12]} but HEAD is {sha[:12]}"
    elif state is None:
        reason = f"no pilot state at {paths.state_file}"
    elif state.config_hash != expected_hash:
        reason = f"pilot config hash {state.config_hash[:12]} != sweep run's {expected_hash[:12]}"
    elif state.git_sha != sha:
        reason = f"pilot ran on {state.git_sha[:12]} but HEAD is {sha[:12]}"
    elif not recorded and (state.status != "complete" or state.train_seconds < PILOT_MINUTES * 60.0):
        reason = f"pilot is {state.status} at {state.train_seconds / 60.0:.1f} of {PILOT_MINUTES:g} min"
    if reason:
        log.warning("pilot %s not reused for %s: %s", pilot.run_id, target.run_id, reason)
        return {}, reason
    log.info("pilot %s reused for %s (first %g min)", pilot.run_id, target.run_id, PILOT_MINUTES)
    return {target.run_id: entry}, ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    require(isinstance(data, dict), f"{path} does not hold a JSON object")
    return data
