"""Seed aggregation and the learning-curve figure.

Results arrive keyed by mark (``run id -> {mark: result or None}``), as evaluation produced them from
the checkpoint that crossed each mark (``CheckpointStore.crossed``); nothing here re-selects by
training time, and a point's x is its mark. A mark at which no run of a count has a result has no
point: absent, never zero.

The curve is the mean across seeds with the min-max range as the band. With 3 seeds an IQM is just
the median (with 2, the mean), so mean plus range says more; the chart caption states this with the
actual seed count. A NaN metric value counts as no seed for the curve (mean, band and seed count).
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


MarkedResults = Mapping[str, Mapping[float, EvalResult | None]]
"""Run id -> {mark minutes: the evaluation of the checkpoint that crossed the mark, or None}."""


def _values_at(marked: MarkedResults, runs: Sequence[str], metric: str, mark: float) -> list[tuple[float, str]]:
    """``(metric value, run id)`` for each of ``runs`` that has a result at ``mark``."""
    out: list[tuple[float, str]] = []
    for run_id in runs:
        result = marked.get(run_id, {}).get(mark)
        if result is not None:
            out.append((float(getattr(result, metric)), run_id))
    return out


def aggregate(
    marked: MarkedResults, metric: str, run_envs: Mapping[str, int], marks_minutes: Sequence[float]
) -> list[CurvePoint]:
    """One point per (count, mark) with at least one seed; count-major, then by mark.

    A NaN value is left out of the mean, the band and ``n_seeds`` alike; a mark where every seed is NaN
    has no point. ``run_envs`` maps each run id of the sweep to its env count; results of other runs are ignored.
    """
    validate_metric(metric)
    points: list[CurvePoint] = []
    for num_envs in sorted(set(run_envs.values())):
        runs = [run_id for run_id, n in run_envs.items() if n == num_envs]
        for mark in marks_minutes:
            # NaN (a metric with no defined value, e.g. no standing env-step) is no seed: min()/max() with a
            # NaN would depend on its position, and the mean would be NaN.
            values = [v for v, _ in _values_at(marked, runs, metric, mark) if not math.isnan(v)]
            if values:
                points.append(
                    CurvePoint(num_envs, float(mark), sum(values) / len(values), min(values), max(values), len(values))
                )
    return points


def median_seed_run(
    marked: MarkedResults, metric: str, run_envs: Mapping[str, int], at_minutes: float
) -> dict[int, str]:
    """Per env count, the run whose ``metric`` at the mark ``at_minutes`` is the median of its seeds.

    Seeds are ranked ascending by (value, run id) with NaN last, and the median is the lower middle
    for an even count. A count with no result at that mark is left out.
    """
    validate_metric(metric)
    chosen: dict[int, str] = {}
    for num_envs in sorted(set(run_envs.values())):
        runs = [run_id for run_id, n in run_envs.items() if n == num_envs]
        scored = [(math.isnan(v), v, run_id) for v, run_id in _values_at(marked, runs, metric, at_minutes)]
        if scored:
            scored.sort()
            chosen[num_envs] = scored[(len(scored) - 1) // 2][2]
    return chosen


def iqm_note(n_seeds: int) -> str:
    """Why the band is mean + range, for the actual seed count."""
    if n_seeds == 3:
        return "with 3 seeds an IQM is just the median, so mean + range is shown"
    if n_seeds == 2:
        return "with 2 seeds an IQM is just the mean, so mean + range is shown"
    return f"with {n_seeds} seed(s), mean + range is shown"


def caption(counts: Sequence[int], n_seeds: int) -> str:
    """Chart caption: the band's meaning (for ``n_seeds``) and the batch arithmetic per count."""
    batches = [batch_arithmetic(n) for n in counts]
    per_count = "; ".join(
        f"{b.num_envs:,} envs: {b.batch_size:,} samples/iter ({b.num_mini_batches} x {b.minibatch_size:,})"
        for b in batches
    )
    return f"Line: mean across seeds; band: min-max across seeds ({iqm_note(n_seeds)}). Batch = envs x {batches[0].steps_per_env} steps per iteration: {per_count}."


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
    fig.text(
        0.01, 0.01, caption(counts, max(p.n_seeds for p in points)), wrap=True, fontsize=8, color=_MUTED, va="bottom"
    )
    fig.subplots_adjust(left=0.09, right=0.88, top=0.95, bottom=0.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=_SURFACE)
    with rc_context({"svg.fonttype": "none"}):  # keep SVG text as text (searchable, editable)
        fig.savefig(out.with_suffix(".svg"), facecolor=_SURFACE)
