"""Unit tests for mprl.curves: seed aggregation, median-seed selection, the learning-curve figure."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from mprl.curves import METRICS, CurvePoint, aggregate, median_seed_run, plot_learning_curves
from mprl.evaluator import EvalResult

RUN_ENVS = {"a-s0": 64, "a-s1": 64, "a-s2": 64, "b-s0": 256}


def _r(run_id: str, minutes: float, xy: float, *, iteration: int = 1) -> EvalResult:
    return EvalResult(run_id, iteration, minutes, xy, 0.1, 0.5, 2.0, 0.3)


def test_aggregate_mean_and_range():
    results = [_r("a-s0", 3.0, 0.2), _r("a-s1", 3.0, 0.5), _r("a-s2", 3.0, 0.8)]

    points = aggregate(results, "vel_error_xy", RUN_ENVS, [3.0])

    assert points == [CurvePoint(64, 3.0, pytest.approx(0.5), 0.2, 0.8, 3)]


def test_aggregate_uses_latest_result_at_or_before_each_mark():
    # A checkpoint stamped 2.98 min belongs to the 3-min mark; one at 3.02 min does not.
    results = [_r("a-s0", 2.98, 0.4), _r("a-s0", 3.02, 0.1, iteration=2), _r("a-s0", 5.9, 0.3, iteration=3)]

    points = aggregate(results, "vel_error_xy", {"a-s0": 64}, [3.0, 6.0])

    assert [(p.minutes, p.mean, p.n_seeds) for p in points] == [(3.0, 0.4, 1), (6.0, 0.3, 1)]


def test_aggregate_missing_seed_reduces_n():
    results = [_r("a-s0", 3.0, 0.2), _r("a-s1", 3.0, 0.6), _r("a-s0", 6.0, 0.1), _r("b-s0", 6.0, 0.9)]

    points = aggregate(results, "vel_error_xy", RUN_ENVS, [3.0, 6.0])

    # a-s2 has no result at all; at 6 min a-s1's latest checkpoint is still its 3-min one.
    assert points == [
        CurvePoint(64, 3.0, pytest.approx(0.4), 0.2, 0.6, 2),
        CurvePoint(64, 6.0, pytest.approx(0.35), 0.1, 0.6, 2),
        CurvePoint(256, 6.0, 0.9, 0.9, 0.9, 1),
    ]


def test_aggregate_mark_without_any_result_is_absent():
    results = [_r("b-s0", 6.0, 0.9)]

    points = aggregate(results, "vel_error_xy", RUN_ENVS, [3.0, 6.0])

    assert points == [CurvePoint(256, 6.0, 0.9, 0.9, 0.9, 1)]  # no point at 3 min, not a zero


def test_aggregate_ignores_results_of_runs_outside_the_sweep():
    results = [_r("a-s0", 3.0, 0.2), _r("other", 3.0, 100.0)]

    assert aggregate(results, "vel_error_xy", RUN_ENVS, [3.0]) == [CurvePoint(64, 3.0, 0.2, 0.2, 0.2, 1)]


def test_median_seed_selection():
    results = [
        _r("a-s0", 45.0, 0.9),
        _r("a-s1", 45.0, 0.1),
        _r("a-s2", 45.0, 0.5),
        _r("a-s1", 48.0, 0.0, iteration=9),  # past the mark: ignored
        _r("b-s0", 42.0, 0.7),
    ]

    assert median_seed_run(results, "vel_error_xy", RUN_ENVS, at_minutes=45.0) == {64: "a-s2", 256: "b-s0"}


def test_median_of_two_seeds_is_the_lower_middle():
    results = [_r("a-s0", 45.0, 0.9), _r("a-s1", 45.0, 0.1)]

    assert median_seed_run(results, "vel_error_xy", RUN_ENVS, at_minutes=45.0) == {64: "a-s1"}


def test_median_ranks_nan_last():
    results = [_r("a-s0", 45.0, math.nan), _r("a-s1", 45.0, 0.4), _r("a-s2", 45.0, 0.2)]

    assert median_seed_run(results, "vel_error_xy", RUN_ENVS, at_minutes=45.0) == {64: "a-s1"}


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


def test_metric_name_validated():
    with pytest.raises(ValueError, match=r"'reward'.*vel_error_xy.*mean_tracking_reward"):
        aggregate([], "reward", RUN_ENVS, [3.0])
    with pytest.raises(ValueError, match=r"'reward'.*vel_error_xy.*mean_tracking_reward"):
        median_seed_run([], "reward", RUN_ENVS, at_minutes=3.0)


def test_metrics_are_the_float_fields():
    assert METRICS == (
        "vel_error_xy",
        "vel_error_yaw",
        "no_fall_fraction",
        "mean_time_to_fall_s",
        "mean_tracking_reward",
    )
