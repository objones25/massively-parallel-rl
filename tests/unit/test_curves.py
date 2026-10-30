"""Unit tests for mprl.curves: seed aggregation, median-seed selection, the learning-curve figure."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from mprl.curves import METRICS, CurvePoint, aggregate, caption, median_seed_run, plot_learning_curves
from mprl.evaluator import EvalResult

RUN_ENVS = {"a-s0": 64, "a-s1": 64, "a-s2": 64, "b-s0": 256}


def _r(run_id: str, minutes: float, xy: float, *, iteration: int = 1) -> EvalResult:
    return EvalResult(run_id, iteration, minutes, xy, 0.1, 0.5, 2.0, 0.3)


def test_aggregate_mean_and_range():
    marked = {
        "a-s0": {3.0: _r("a-s0", 3.02, 0.2)},
        "a-s1": {3.0: _r("a-s1", 3.05, 0.5)},
        "a-s2": {3.0: _r("a-s2", 3.01, 0.8)},
    }

    points = aggregate(marked, "vel_error_xy", RUN_ENVS, [3.0])

    assert points == [CurvePoint(64, 3.0, pytest.approx(0.5), 0.2, 0.8, 3)]


def test_aggregate_uses_the_result_evaluated_for_each_mark():
    # Checkpoints stamped past their marks (3:02, 6:03) still plot at the mark they crossed; the
    # curve never re-selects by training time.
    marked = {"a-s0": {3.0: _r("a-s0", 3.0333, 0.4), 6.0: _r("a-s0", 6.05, 0.3, iteration=2)}}

    points = aggregate(marked, "vel_error_xy", {"a-s0": 64}, [3.0, 6.0])

    assert [(p.minutes, p.mean, p.n_seeds) for p in points] == [(3.0, 0.4, 1), (6.0, 0.3, 1)]


def test_aggregate_missing_seed_reduces_n():
    marked = {
        "a-s0": {3.0: _r("a-s0", 3.0, 0.2), 6.0: _r("a-s0", 6.0, 0.1)},
        "a-s1": {3.0: _r("a-s1", 3.0, 0.6), 6.0: None},  # no checkpoint crossed 6:00
        "b-s0": {6.0: _r("b-s0", 6.0, 0.9)},
    }  # a-s2 has nothing at all

    points = aggregate(marked, "vel_error_xy", RUN_ENVS, [3.0, 6.0])

    assert points == [
        CurvePoint(64, 3.0, pytest.approx(0.4), 0.2, 0.6, 2),
        CurvePoint(64, 6.0, 0.1, 0.1, 0.1, 1),
        CurvePoint(256, 6.0, 0.9, 0.9, 0.9, 1),
    ]


def test_aggregate_mark_without_any_result_is_absent():
    marked = {"b-s0": {3.0: None, 6.0: _r("b-s0", 6.0, 0.9)}}

    points = aggregate(marked, "vel_error_xy", RUN_ENVS, [3.0, 6.0])

    assert points == [CurvePoint(256, 6.0, 0.9, 0.9, 0.9, 1)]  # no point at 3 min, not a zero


def test_aggregate_ignores_runs_outside_the_sweep():
    marked = {"a-s0": {3.0: _r("a-s0", 3.0, 0.2)}, "other": {3.0: _r("other", 3.0, 100.0)}}

    assert aggregate(marked, "vel_error_xy", RUN_ENVS, [3.0]) == [CurvePoint(64, 3.0, 0.2, 0.2, 0.2, 1)]


def test_median_seed_selection():
    marked = {
        "a-s0": {45.0: _r("a-s0", 45.05, 0.9)},
        "a-s1": {42.0: _r("a-s1", 42.0, 0.0), 45.0: _r("a-s1", 45.03, 0.1)},
        "a-s2": {45.0: _r("a-s2", 45.04, 0.5)},
        "b-s0": {42.0: _r("b-s0", 42.0, 0.7)},  # nothing at 45:00: 256 is left out
    }

    assert median_seed_run(marked, "vel_error_xy", RUN_ENVS, at_minutes=45.0) == {64: "a-s2"}


def test_median_of_two_seeds_is_the_lower_middle():
    marked = {"a-s0": {45.0: _r("a-s0", 45.0, 0.9)}, "a-s1": {45.0: _r("a-s1", 45.0, 0.1)}}

    assert median_seed_run(marked, "vel_error_xy", RUN_ENVS, at_minutes=45.0) == {64: "a-s1"}


def test_median_ranks_nan_last():
    marked = {
        "a-s0": {45.0: _r("a-s0", 45.0, math.nan)},
        "a-s1": {45.0: _r("a-s1", 45.0, 0.4)},
        "a-s2": {45.0: _r("a-s2", 45.0, 0.2)},
    }

    assert median_seed_run(marked, "vel_error_xy", RUN_ENVS, at_minutes=45.0) == {64: "a-s1"}


@pytest.mark.parametrize(
    ("n_seeds", "phrase"),
    [(3, "with 3 seeds an IQM is just the median"), (2, "with 2 seeds an IQM is just the mean")],
)
def test_caption_names_the_actual_seed_count(n_seeds: int, phrase: str):
    assert phrase in caption([64], n_seeds)


def test_plot_writes_png_and_svg(tmp_path: Path):
    points = [
        CurvePoint(64, 3.0, 0.8, 0.7, 0.9, 3),
        CurvePoint(64, 6.0, 0.6, 0.5, 0.7, 3),
        CurvePoint(4096, 3.0, 0.5, 0.4, 0.6, 2),
        CurvePoint(4096, 6.0, 0.2, 0.1, 0.3, 2),
    ]

    plot_learning_curves(points, "velocity error xy (m/s)", tmp_path / "learning_curves.png")

    assert (tmp_path / "learning_curves.png").stat().st_size > 0
    assert (tmp_path / "learning_curves.svg").stat().st_size > 0
    svg = (tmp_path / "learning_curves.svg").read_text()
    assert "4096 envs" in svg
    assert "98,304" in svg  # batch arithmetic in the caption: 4096 envs x 24 steps
    assert "with 3 seeds" in svg  # the most seeds any point has


def test_metric_name_validated():
    with pytest.raises(ValueError, match=r"'reward'.*vel_error_xy.*mean_tracking_reward"):
        aggregate({}, "reward", RUN_ENVS, [3.0])
    with pytest.raises(ValueError, match=r"'reward'.*vel_error_xy.*mean_tracking_reward"):
        median_seed_run({}, "reward", RUN_ENVS, at_minutes=3.0)


def test_metrics_are_the_float_fields():
    assert METRICS == (
        "vel_error_xy",
        "vel_error_yaw",
        "no_fall_fraction",
        "mean_time_to_fall_s",
        "mean_tracking_reward",
    )


@pytest.mark.parametrize("order", [("a-s0", "a-s1", "a-s2"), ("a-s1", "a-s0", "a-s2"), ("a-s1", "a-s2", "a-s0")])
def test_aggregate_band_ignores_nan_seeds_in_any_order(order: tuple[str, str, str]):
    # min()/max() with a NaN depend on where the NaN sits; the band, mean and seed count must not.
    values = {"a-s0": math.nan, "a-s1": 0.2, "a-s2": 0.6}
    marked = {run_id: {3.0: _r(run_id, 3.0, values[run_id])} for run_id in order}

    points = aggregate(marked, "vel_error_xy", dict.fromkeys(order, 64), [3.0])

    assert points == [CurvePoint(64, 3.0, pytest.approx(0.4), 0.2, 0.6, 2)]


def test_aggregate_mark_where_every_seed_is_nan_is_absent():
    marked = {"a-s0": {3.0: _r("a-s0", 3.0, math.nan)}, "a-s1": {3.0: _r("a-s1", 3.0, math.nan)}}

    assert aggregate(marked, "vel_error_xy", {"a-s0": 64, "a-s1": 64}, [3.0]) == []
