"""Unit tests for mprl.budget: the spec's cost table, the cut order and the measurements file."""

from __future__ import annotations

from pathlib import Path

import pytest

from mprl.budget import (
    CUT_ORDER,
    Measurements,
    estimate,
    load_measurements,
    plan_within_budget,
    save_measurements,
)
from mprl.config import SweepSpec
from mprl.errors import BudgetExceeded

COUNTS = (64, 256, 1024, 4096, 16384)


def _m(io: float = 0.0, eval_s: float = 0.0) -> Measurements:
    return Measurements(
        eval_seconds_per_checkpoint=eval_s,
        io_overhead_fraction=io,
        fits=dict.fromkeys(COUNTS, True),
        steps_per_second={n: 1000.0 * n for n in COUNTS},
    )


def test_estimate_matches_spec_table() -> None:
    est = estimate(SweepSpec("main"), _m())

    assert est.total_hours == pytest.approx(18.5)  # 0.25 + 11.25 + 3 + 4
    assert est.total_usd == pytest.approx(33.115)
    assert sum(hours for _, hours in est.lines) == pytest.approx(est.total_hours)


def test_estimate_counts_overhead_evaluations_and_finished_long_run() -> None:
    # runs: 15 x 0.75 h x 1.5 = 16.875; evals: (15 x 15 + 135/3) x 36 s = 270 x 0.01 h = 2.7;
    # long run: 135 of 180 min left x 1.5 = 3.375; setup 4.25.
    est = estimate(SweepSpec("main"), _m(io=0.5, eval_s=36.0), long_run_done_minutes=45.0)

    assert est.total_hours == pytest.approx(16.875 + 2.7 + 3.375 + 4.25)
    assert est.total_usd == pytest.approx(est.total_hours * 1.79)


def test_cuts_apply_in_order() -> None:
    sweep, cuts, est = plan_within_budget(SweepSpec("main"), _m(io=0.35, eval_s=10.0))

    assert cuts == ("eval_every_6min", "long_run_half")
    assert cuts == CUT_ORDER[:2]
    assert sweep.eval_every_minutes == pytest.approx(6.0)
    assert sweep.long_run_minutes == pytest.approx(90.0)
    assert sweep.env_counts == COUNTS
    assert sweep.seeds == (0, 1, 2)
    assert est.total_usd <= 40.0
    assert est == estimate(sweep, _m(io=0.35, eval_s=10.0))


def test_no_cuts_when_within_cap() -> None:
    sweep, cuts, est = plan_within_budget(SweepSpec("main"), _m())

    assert cuts == ()
    assert sweep == SweepSpec("main")
    assert est.total_usd == pytest.approx(33.115)


def test_64_never_dropped() -> None:
    sweep, cuts, est = plan_within_budget(SweepSpec("main"), _m(io=1.0))

    assert cuts == CUT_ORDER
    assert sweep.env_counts == (64, 256, 1024, 4096)
    assert sweep.seeds == (0, 1)
    assert est.total_usd <= 40.0


def test_drop_16384_is_skipped_when_already_absent() -> None:
    sweep, cuts, _ = plan_within_budget(SweepSpec("main", env_counts=(64, 256, 1024, 4096)), _m(io=1.0))

    assert cuts == ("eval_every_6min", "long_run_half", "two_seeds")
    assert 64 in sweep.env_counts


def test_budget_exceeded_after_all_cuts() -> None:
    with pytest.raises(BudgetExceeded, match=r"\$.*exceeds the \$40\.00 cap after cuts"):
        plan_within_budget(SweepSpec("main"), _m(io=5.0))


def test_measurements_round_trip_with_int_keys(tmp_path: Path) -> None:
    m = Measurements(2.5, 0.04, {64: True, 16384: False}, {64: 5000.0})
    path = tmp_path / "measurements.json"

    save_measurements(path, m)

    assert load_measurements(path) == m
    assert all(isinstance(n, int) for n in load_measurements(path).fits)
