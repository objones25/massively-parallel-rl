"""Unit tests for mprl.results' pure parts: marks, the planned sweep, run sources, the model card.

The evaluation, rendering, W&B and Hub steps run for real in tests/smoke/test_results_cpu.py.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import create_autospec

import pytest

import mprl.results as results_mod
from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.config import SweepSpec
from mprl.curves import CurvePoint
from mprl.errors import UploadFailed
from mprl.evaluator import EvalConfig, EvalResult, evaluate_checkpoint
from mprl.hub import HubPublisher
from mprl.negative_space import CheckFailed
from mprl.paths import RunPaths, sweep_results_dir
from mprl.results import (
    RunSource,
    build_results,
    eval_marks,
    evaluate_source,
    marked_checkpoints,
    planned_sweep,
    publish_hub,
    run_sources,
)
from mprl.results_card import model_card
from mprl.sweep import pilot_spec

SWEEP = SweepSpec("main")


OVERSHOOT_S = (1, 2, 3)  # real saves land after the iteration that crosses their mark


def _fill(store: CheckpointStore, minutes: list[float]) -> None:
    """A checkpoint per mark, each stamped 1-3 s past it (e.g. 181 s, 362 s, 543 s, ...)."""
    for i, m in enumerate(minutes, start=1):
        store.save_atomic(lambda p: p.write_bytes(b"w"), i * 10, m * 60.0 + OVERSHOOT_S[i % 3])


def test_eval_marks_every_interval_and_always_the_budget():
    assert eval_marks(3.0, 45.0) == [3.0 * k for k in range(1, 16)]
    assert eval_marks(6.0, 45.0) == [*(6.0 * k for k in range(1, 8)), 45.0]  # eval_every_6min: ..., 42, 45
    assert eval_marks(0.05, 0.1) == [0.05, 0.1]


def test_planned_sweep_comes_from_the_manifest(tmp_path: Path):
    planned = replace(SWEEP, env_counts=(64, 256, 1024, 4096), seeds=(0, 1), eval_every_minutes=6.0)
    manifest = {"planned": asdict(planned), "cuts": ["eval_every_6min", "drop_16384", "two_seeds"], "excluded": {}}
    results = sweep_results_dir(tmp_path, "main")
    results.mkdir(parents=True)
    (results / "sweep_manifest.json").write_text(json.dumps(manifest))

    assert planned_sweep(SWEEP, tmp_path) == (planned, json.loads(json.dumps(manifest)))


def test_planned_sweep_without_manifest_is_the_requested_one(tmp_path: Path):
    assert planned_sweep(SWEEP, tmp_path) == (SWEEP, {})


def test_reused_pilot_run_reads_the_pilot_checkpoints_up_to_max_minutes(tmp_path: Path):
    small = replace(SWEEP, env_counts=(64, 4096), seeds=(0,))
    pilot = RunPaths(tmp_path, pilot_spec(small))
    reuse = {"main-n04096-s0": {"source_run_dir": str(pilot.run_dir), "max_minutes": 45.0}}
    results = sweep_results_dir(tmp_path, "main")
    results.mkdir(parents=True)
    (results / "pilot_reuse.json").write_text(json.dumps(reuse))

    sources = {s.run_id: s for s in run_sources(small, tmp_path)}

    reused = sources["main-n04096-s0"]
    assert reused.checkpoints_dir == pilot.checkpoints_dir
    assert reused.eval_csv == pilot.eval_dir / "eval.csv"
    assert reused.source_run_id == pilot.spec.run_id
    assert reused.max_minutes == pytest.approx(45.0)
    assert reused.interval_minutes == pytest.approx(3.0)
    own = sources["main-n00064-s0"]
    assert own.checkpoints_dir == RunPaths(tmp_path, small.runs()[0]).checkpoints_dir
    assert own.source_run_id == "main-n00064-s0"
    assert own.max_minutes == small.budget_minutes


def test_marked_checkpoints_pick_the_checkpoint_that_crossed_each_mark(tmp_path: Path):
    store = CheckpointStore(tmp_path / "ck")
    _fill(store, [3.0 * k for k in range(1, 16)])  # 3:00 .. 45:00, every stamp a little late

    marked = marked_checkpoints(store, [3.0 * k for k in range(1, 16)], interval_minutes=3.0, max_minutes=45.0)

    assert all(ref is not None for ref in marked.values())  # every mark has its checkpoint, the budget too
    assert [round(ref.train_seconds // 60) for ref in marked.values() if ref is not None] == list(range(3, 46, 3))


def test_reused_pilot_cap_is_the_checkpoint_that_crossed_max_minutes(tmp_path: Path):
    store = CheckpointStore(tmp_path / "ck")
    _fill(store, [3.0 * k for k in range(1, 21)])  # a pilot that went on to 60 minutes as the long run

    marked = marked_checkpoints(store, [42.0, 45.0], interval_minutes=3.0, max_minutes=45.0)

    final = marked[45.0]
    assert final is not None
    assert 45.0 * 60.0 <= final.train_seconds < 48.0 * 60.0


def test_marks_beyond_max_minutes_are_never_selected(tmp_path: Path):
    store = CheckpointStore(tmp_path / "ck")
    _fill(store, [3.0 * k for k in range(1, 21)])

    with pytest.raises(CheckFailed, match="past"):
        marked_checkpoints(store, [45.0, 48.0], interval_minutes=3.0, max_minutes=45.0)


def test_mark_without_a_crossing_checkpoint_is_absent(tmp_path: Path):
    store = CheckpointStore(tmp_path / "ck")
    _fill(store, [6.0])  # the run's first save came after 6:00

    marked = marked_checkpoints(store, [3.0, 6.0], interval_minutes=3.0, max_minutes=45.0)

    assert marked[3.0] is None
    assert marked[6.0] is not None


def _fake_eval(run_id: str, ref: CheckpointRef, cfg: EvalConfig, device: str) -> EvalResult:
    return EvalResult(run_id, ref.iteration, ref.train_seconds / 60.0, 0.5, 0.1, 1.0, cfg.horizon_s, 0.3)


def test_eval_cache_is_reused_only_for_the_same_eval_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run = tmp_path / "run"
    _fill(CheckpointStore(run / "checkpoints"), [3.0, 6.0])
    source = RunSource(
        "main-n00064-s0", 64, 0, run / "checkpoints", run / "eval" / "eval.csv", "main-n00064-s0", 6.0, 3.0
    )
    fake = create_autospec(evaluate_checkpoint, side_effect=_fake_eval)
    monkeypatch.setattr(results_mod, "evaluate_checkpoint", fake)
    short, longer = EvalConfig(horizon_s=1.0), EvalConfig(horizon_s=2.0)

    first = evaluate_source(source, [3.0, 6.0], short, "cpu")
    again = evaluate_source(source, [3.0, 6.0], short, "cpu")
    assert fake.call_count == 2  # the second pass was served from the cache
    assert again == first

    changed = evaluate_source(source, [3.0, 6.0], longer, "cpu")

    assert fake.call_count == 4  # a different EvalConfig re-evaluates every row
    assert {r.mean_time_to_fall_s for r in changed.values() if r is not None} == {2.0}
    assert {r.run_id for r in changed.values() if r is not None} == {"main-n00064-s0"}


def test_model_card_lists_cuts_exclusions_and_curriculum():
    reason = "num_envs=16384 does not fit: CUDA out of memory in the throughput test"
    manifest = {"cuts": ["eval_every_6min", "two_seeds"], "excluded": {"16384": reason}}
    planned = replace(SWEEP, env_counts=(64, 4096), seeds=(0, 1), eval_every_minutes=6.0)

    card = model_card(
        planned,
        manifest,
        reuse={"main-n04096-s0": {"source_run_dir": "/v/main-long-n04096-s0", "max_minutes": 45.0}},
        eval_cfg=EvalConfig(),
        primary_metric="vel_error_xy",
        final_points=[CurvePoint(64, 42.0, 0.5, 0.4, 0.6, 2), CurvePoint(4096, 42.0, 0.2, 0.1, 0.3, 2)],
        race=((1.0, 0.0, 0.0), 4.0, 45.0, {64: "main-n00064-s1", 4096: "main-n04096-s0"}),
    )

    assert "eval_every_6min" in card
    assert "two_seeds" in card
    assert "1 env count excluded" in card
    assert reason in card
    assert "| 4096 | 24 | 98304 | 4 | 24576 |" in card
    assert "common_step_counter" in card
    assert "track_linear_velocity" in card
    assert "main-n04096-s0" in card  # the reused pilot is disclosed
    assert "| (1.0, 0.0, 0.0) |" in card
    assert "median seed by `vel_error_xy` at 45 training minutes" in card
    assert "with 2 seeds an IQM is just the mean" in card  # two_seeds cut: never "with 3 seeds"
    assert "3 seeds" not in card
    assert "lower-middle" in card
    assert "| 4096 | 0.2 | 0.1 to 0.3 | 2 |" in card  # the final-mark summary


def test_model_card_without_a_manifest_says_no_cuts_recorded():
    card = model_card(
        SWEEP,
        {},
        reuse={},
        eval_cfg=EvalConfig(),
        primary_metric="no_fall_fraction",
        final_points=[],
        race=((1.0, 0.0, 0.0), 4.0, 45.0, {}),
    )

    assert "No sweep manifest" in card


def test_build_results_rejects_unknown_metric_before_any_work(tmp_path: Path):
    with pytest.raises(ValueError, match="valid metrics: vel_error_xy"):
        build_results(SWEEP, tmp_path, primary_metric="reward", publisher=None, wandb_mode="disabled", device="cpu")

    assert not (tmp_path / "sweeps").exists()


def test_publish_hub_uploads_the_results_dir(tmp_path: Path):
    publisher = create_autospec(HubPublisher, instance=True)

    assert publish_hub(SWEEP, tmp_path, publisher) == "ok"

    publisher.upload_folder.assert_called_once_with(tmp_path, "results/main", "main: results")


def test_publish_hub_failure_is_recorded_and_keeps_local_results(tmp_path: Path):
    (tmp_path / "results.csv").write_text("x")
    publisher = create_autospec(HubPublisher, instance=True)
    publisher.upload_folder.side_effect = UploadFailed("upload_folder failed after 5 attempts: 503")

    outcome = publish_hub(SWEEP, tmp_path, publisher)

    assert outcome == "UploadFailed: upload_folder failed after 5 attempts: 503"
    assert (tmp_path / "results.csv").read_text() == "x"


def test_publish_hub_without_publisher_is_skipped(tmp_path: Path):
    assert publish_hub(SWEEP, tmp_path, None) == "skipped: no publisher"
