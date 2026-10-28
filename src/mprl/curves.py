"""Seed aggregation and the learning-curve figure.

A run's value at a mark is its latest evaluated checkpoint at or before that mark (the policy that
existed by then on the training clock, as ``CheckpointStore.at_or_before`` picks it). A mark at which
no run of a count has a checkpoint yet has no point: absent, never zero.

The curve is the mean across seeds with the min-max range as the band. With 3 seeds an IQM is just
the median, so mean plus range says more; the chart caption states this.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Final

from matplotlib import rc_context
from matplotlib.figure import Figure

from mprl.config import batch_arithmetic
from mprl.evaluator import EvalResult
from mprl.negative_space import require

if TYPE_CHECKING:
    from pathlib import Path

METRICS: Final[tuple[str, ...]] = tuple(
    f.name for f in fields(EvalResult) if f.type == "float" and f.name != "train_minutes"
)
"""The metric names a curve can plot: ``EvalResult``'s float fields except the x-axis, ``train_minutes``."""

_MARK_TOLERANCE_MIN: Final = 1e-9
# Ordinal blue ramp (steps 250..650 of the dataviz reference palette), light to dark with num_envs;
# the lightest step clears 2:1 against the light surface.
_RAMP: Final = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
_SURFACE: Final = "#fcfcfb"
_TEXT: Final = "#0b0b0b"
_MUTED: Final = "#52514e"
_GRID: Final = "#e4e3df"


@dataclass(frozen=True)
class CurvePoint:
    num_envs: int
    minutes: float
    mean: float
    low: float
    high: float
    n_seeds: int


def validate_metric(metric: str) -> None:
    """Raise ``ValueError`` (bad user input) naming the valid metrics."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; valid metrics: {', '.join(METRICS)}")


def _value_at(results: Sequence[EvalResult], metric: str, mark: float) -> float | None:
    """``metric`` of the latest result (by training time) at or before ``mark``; None if there is none."""
    eligible = [r for r in results if r.train_minutes <= mark + _MARK_TOLERANCE_MIN]
    if not eligible:
        return None
    latest = max(eligible, key=lambda r: (r.train_minutes, r.iteration))
    return float(getattr(latest, metric))


def _by_run(results: Sequence[EvalResult], run_envs: Mapping[str, int]) -> dict[str, list[EvalResult]]:
    grouped: dict[str, list[EvalResult]] = {run_id: [] for run_id in run_envs}
    for result in results:
        if result.run_id in grouped:
            grouped[result.run_id].append(result)
    return grouped


def aggregate(
    results: Sequence[EvalResult], metric: str, run_envs: Mapping[str, int], marks_minutes: Sequence[float]
) -> list[CurvePoint]:
    """One point per (count, mark) with at least one seed; count-major, then by mark.

    ``run_envs`` maps each run id of the sweep to its env count; results of other runs are ignored.
    """
    validate_metric(metric)
    grouped = _by_run(results, run_envs)
    points: list[CurvePoint] = []
    for num_envs in sorted(set(run_envs.values())):
        runs = [run_id for run_id, n in run_envs.items() if n == num_envs]
        for mark in marks_minutes:
            values = [v for run_id in runs if (v := _value_at(grouped[run_id], metric, mark)) is not None]
            if values:
                points.append(
                    CurvePoint(num_envs, float(mark), sum(values) / len(values), min(values), max(values), len(values))
                )
    return points


def median_seed_run(
    results: Sequence[EvalResult], metric: str, run_envs: Mapping[str, int], at_minutes: float
) -> dict[int, str]:
    """Per env count, the run whose ``metric`` at ``at_minutes`` is the median of its seeds.

    Seeds are ranked ascending by (value, run id) with NaN last, and the median is the lower middle
    for an even count. A count with no result by ``at_minutes`` is left out.
    """
    validate_metric(metric)
    grouped = _by_run(results, run_envs)
    chosen: dict[int, str] = {}
    for num_envs in sorted(set(run_envs.values())):
        scored = [
            (math.isnan(v), v, run_id)
            for run_id, n in run_envs.items()
            if n == num_envs and (v := _value_at(grouped[run_id], metric, at_minutes)) is not None
        ]
        if scored:
            scored.sort()
            chosen[num_envs] = scored[(len(scored) - 1) // 2][2]
    return chosen


def caption(counts: Sequence[int]) -> str:
    """Chart caption: the band's meaning and the batch arithmetic per count."""
    batches = [batch_arithmetic(n) for n in counts]
    per_count = "; ".join(
        f"{b.num_envs:,} envs: {b.batch_size:,} samples/iter ({b.num_mini_batches} x {b.minibatch_size:,})"
        for b in batches
    )
    return (
        "Line: mean across seeds; band: min-max across seeds (with 3 seeds an IQM is just the median, "
        f"so mean + range is shown). Batch = envs x {batches[0].steps_per_env} steps per iteration: {per_count}."
    )


def plot_learning_curves(points: Sequence[CurvePoint], metric_label: str, out: Path) -> None:
    """``out`` (a ``.png``) and the same figure as ``.svg`` next to it: one line + band per env count."""
    require(out.suffix == ".png", f"out must be a .png path, got {out}")
    require(len(points) > 0, "no curve points to plot")
    counts = sorted({p.num_envs for p in points})
    require(len(counts) <= len(_RAMP), f"at most {len(_RAMP)} env counts can be plotted, got {counts}")
    colors = dict(zip(counts, _RAMP[len(_RAMP) - len(counts) :], strict=True))  # darkest = most envs
    fig = Figure(figsize=(9.0, 5.6), facecolor=_SURFACE)
    ax = fig.add_subplot()
    ax.set_facecolor(_SURFACE)
    for n in counts:
        series = sorted((p for p in points if p.num_envs == n), key=lambda p: p.minutes)
        xs = [p.minutes for p in series]
        label = f"{n} envs"
        ax.fill_between(xs, [p.low for p in series], [p.high for p in series], color=colors[n], alpha=0.18, lw=0)
        ax.plot(xs, [p.mean for p in series], color=colors[n], lw=2, marker="o", markersize=4, label=label)
        ax.annotate(
            label, (xs[-1], series[-1].mean), xytext=(6, 0), textcoords="offset points", va="center", color=_TEXT
        )
    ax.set_xlabel("training-clock minutes", color=_MUTED)
    ax.set_ylabel(metric_label, color=_MUTED)
    ax.grid(color=_GRID, lw=0.8)
    ax.tick_params(colors=_MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.legend(frameon=False, labelcolor=_TEXT)
    fig.text(0.01, 0.01, caption(counts), wrap=True, fontsize=8, color=_MUTED, va="bottom")
    fig.subplots_adjust(left=0.09, right=0.88, top=0.95, bottom=0.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=_SURFACE)
    with rc_context({"svg.fonttype": "none"}):  # keep SVG text as text (searchable, editable)
        fig.savefig(out.with_suffix(".svg"), facecolor=_SURFACE)
