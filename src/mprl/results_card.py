"""The results ``README.md``: setup, batch arithmetic, cuts and exclusions, curriculum disclosure,
evaluation protocol and the final-mark summary. Pure text from the sweep, manifest and curve points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from mprl.config import TASK_ID, batch_arithmetic
from mprl.curves import iqm_note

if TYPE_CHECKING:
    from mprl.config import SweepSpec
    from mprl.curves import CurvePoint
    from mprl.evaluator import EvalConfig

_METRIC_DEFINITIONS = """\
| Metric | Definition |
| --- | --- |
| `vel_error_xy` | mean over standing env-steps of the norm of command xy minus base linear velocity xy, body frame (m/s) |
| `vel_error_yaw` | mean over standing env-steps of the absolute command yaw rate minus base yaw rate (rad/s) |
| `no_fall_fraction` | fraction of envs in which `fell_over` (70 deg tilt) never fired within the horizon |
| `mean_time_to_fall_s` | time at which `fell_over` first fired, the horizon if never (s) |
| `mean_tracking_reward` | the task's `track_linear_velocity` reward term only, `exp(-err^2 / std^2)` in (0, 1], \
unweighted, mean over standing env-steps. Yaw tracking is not in it; it is `vel_error_yaw`. |"""

# "Every evaluation command lies inside stage 0": EVAL_COMMANDS keep |lin_x| <= 1.0 and |ang_z| <= 0.5, and
# the training lin_vel_y range is (-1.0, 1.0) at every curriculum stage (mjlab 1.6.0
# tasks/velocity/velocity_env_cfg.py:192), so (0.0, 0.5, 0.0) is inside it too.
_CURRICULUM = """\
The shipped G1 flat task keeps its `command_vel` curriculum (the config is used unchanged). It advances
on `env.common_step_counter`, which increments once per `env.step` regardless of `num_envs`, with stages
at steps 0 (lin_vel_x +-1.0, ang_vel_z +-0.5), 5000*24 (lin_vel_x -1.5..2.0, ang_vel_z +-0.7) and
10000*24 (lin_vel_x -2.0..3.0). Small-N runs complete more iterations per training minute and so reach
the harder stages sooner. That favors them, so the measured speed-up of large N is, if anything,
understated. Every evaluation command lies inside stage 0, so no run is scored on commands it was
never trained on.

Fixed on-clock costs per env step or per iteration (rsl_rl's per-step `.cpu()` episode bookkeeping and
`check_nan` syncs, and the gradient-norm hooks armed every 10th iteration) are the same for every run, so they
weigh proportionally more on small-N runs, whose steps each do less work. This biases against small N: on
its own it makes the measured speed-up of large N larger than the simulator and PPO alone would give."""


def _cuts_section(manifest: Mapping[str, Any]) -> list[str]:
    if not manifest:
        return ["No sweep manifest was found, so no cuts or exclusions were recorded for this sweep."]
    cuts = list(manifest.get("cuts", []))
    excluded = dict(manifest.get("excluded", {}))
    lines = [f"Cuts applied (in the spec's cut order): {', '.join(f'`{c}`' for c in cuts) if cuts else 'none'}."]
    noun = "env count" if len(excluded) == 1 else "env counts"
    lines.append(f"{len(excluded)} {noun} excluded{':' if excluded else '.'}")
    lines += [f"- {n}: {reason}" for n, reason in sorted(excluded.items(), key=lambda kv: int(kv[0]))]
    return lines


def _long_run_lines(
    sweep: SweepSpec, marks: Sequence[float], command: tuple[float, float, float], seconds: float, problem: str
) -> list[str]:
    run_id = sweep.long_run().run_id
    note = (
        [f"The long run did not complete ({problem}): its videos stop at the last checkpoint it saved.", ""]
        if problem
        else []
    )
    if not marks:
        return [
            *note,
            f"`long_run_videos/`: none; the long run `{run_id}` has no checkpoint at any evaluation mark.",
            "",
        ]
    return [
        *note,
        f"`long_run_videos/`: the long run `{run_id}` ({sweep.long_run_envs} envs, {sweep.long_run_minutes:g} "
        f"training minutes) at {len(marks)} evaluation marks ({marks[0]:g} to {marks[-1]:g} min), each the "
        f"checkpoint that crossed the mark rendered for {seconds:g} s in real time on command {command}; "
        "named `t<minutes>min_it<iteration>.mp4`.",
        "",
    ]


def model_card(
    sweep: SweepSpec,
    manifest: Mapping[str, Any],
    *,
    reuse: Mapping[str, Mapping[str, Any]],
    eval_cfg: EvalConfig,
    primary_metric: str,
    final_points: Sequence[CurvePoint],
    race: tuple[tuple[float, float, float], float, float, Mapping[int, str]],
    long_run_video_marks: Sequence[float] = (),
    long_run_problem: str = "",
) -> str:
    """The card. ``long_run_problem`` is why the long run is not ``complete`` ("" when it is)."""
    command, race_seconds, median_at, median = race
    batch_rows = []
    for n in sweep.env_counts:
        b = batch_arithmetic(n)
        batch_rows.append(f"| {n} | {b.steps_per_env} | {b.batch_size} | {b.num_mini_batches} | {b.minibatch_size} |")
    reuse_lines = [
        f"- `{run_id}` is the pilot's first {entry['max_minutes']:g} training minutes "
        f"(`{entry['source_run_dir']}`), same code and config."
        for run_id, entry in sorted(reuse.items())
    ] or ["- none: every run trained on its own."]
    summary = [f"| {p.num_envs} | {p.mean:.4g} | {p.low:.4g} to {p.high:.4g} | {p.n_seeds} |" for p in final_points]
    at = f" at {final_points[0].minutes:g} training minutes" if final_points else ""
    lines = [
        f"# Massively parallel RL: {TASK_ID}, sweep `{sweep.name}`",
        "",
        "## Setup",
        "",
        f"mjlab's shipped G1 flat-ground velocity task and PPO config, unchanged except `num_envs` and the seed. "
        f"Env counts {', '.join(str(n) for n in sweep.env_counts)}; seeds {', '.join(str(s) for s in sweep.seeds)}; "
        f"{sweep.budget_minutes:g} training-clock minutes per run (rollout + returns + PPO update only; checkpoint "
        f"I/O, logging, evaluation and rendering are off the clock). Checkpoints every "
        f"{sweep.checkpoint_every_minutes:g} min; evaluation every {sweep.eval_every_minutes:g} min.",
        "",
        "## Batch arithmetic",
        "",
        "| num_envs | steps/env | batch (samples/iter) | minibatches | minibatch size |",
        "| --- | --- | --- | --- | --- |",
        *batch_rows,
        "",
        "## Cuts and exclusions",
        "",
        *_cuts_section(manifest),
        "",
        "Pilot reuse:",
        *reuse_lines,
        "",
        "## Curriculum disclosure",
        "",
        _CURRICULUM,
        "",
        "## Evaluation",
        "",
        f"Every evaluated checkpoint is scored identically (cached rows are reused only under the same "
        f"evaluation-config hash), whatever `num_envs` it trained at: the play env "
        f"(no observation noise, no pushes, no curriculum) with {eval_cfg.envs_per_command} env(s) per command, "
        f"seeds {', '.join(str(s) for s in eval_cfg.seeds)}, a {eval_cfg.horizon_s:g} s horizon, commands pinned.",
        "",
        "| Command (lin_x, lin_y, ang_z) |",
        "| --- |",
        *(f"| {tuple(float(c) for c in cmd)} |" for cmd in eval_cfg.commands),
        "",
        _METRIC_DEFINITIONS,
        "",
        f"## Results (primary metric: `{primary_metric}`)",
        "",
        f"`learning_curves.png` / `.svg`: mean across seeds, band min-max ({iqm_note(len(sweep.seeds))}). "
        f"Each mark (every {sweep.eval_every_minutes:g} min, plus the {sweep.budget_minutes:g}-min budget) "
        "shows the checkpoint that crossed it: the first saved at or after the mark and before the next "
        f"{sweep.checkpoint_every_minutes:g}-min checkpoint mark. A mark no checkpoint crossed has no point.",
        "",
        f"Final mark{at}:",
        "",
        "| num_envs | mean | range | seeds |",
        "| --- | --- | --- | --- |",
        *summary,
        "",
        f"`race.mp4`: per env count, the median seed by `{primary_metric}` at {median_at:g} training minutes "
        f"({', '.join(f'{n}: `{r}`' for n, r in sorted(median.items()))}; seeds ranked by ascending value, "
        "and with an even seed count the lower-middle one), its checkpoint crossing every "
        f"{sweep.checkpoint_every_minutes:g}-min mark and the budget, rendered for {race_seconds:g} s in real "
        "time on command "
        f"{command} (1.0 m/s straight ahead, the most legible walking demo).",
        "",
        "`results.csv`: one row per run and evaluation mark (absent marks have no row).",
        "",
        *_long_run_lines(sweep, long_run_video_marks, command, race_seconds, long_run_problem),
    ]
    return "\n".join(lines)
