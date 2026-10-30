"""Unit tests for mprl.results' pure parts: marks, the planned sweep, run sources, the model card.

The evaluation, rendering, W&B and Hub steps run for real in tests/smoke/test_results_cpu.py.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
import wandb
from huggingface_hub.errors import HFValidationError

import mprl.results as results_mod
from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.config import SweepSpec
from mprl.curves import CurvePoint
from mprl.errors import UploadFailed
from mprl.evaluator import EVAL_COMMANDS, EvalConfig, EvalResult, evaluate_checkpoint
from mprl.hub import HubPublisher
from mprl.negative_space import CheckFailed
from mprl.paths import RunPaths, sweep_results_dir
from mprl.race_video import write_video
from mprl.render import render_rollout
from mprl.results import (
    RunSource,
    build_results,
    eval_marks,
    evaluate_source,
    marked_checkpoints,
    planned_sweep,
    publish_hub,
    render_long_run_videos,
    run_sources,
)
from mprl.results_card import model_card
from mprl.run_state import RunState, save_state
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


# --- long-run videos (off the clock, during results) ---------------------------------------------------

LONG = replace(SWEEP, long_run_minutes=9.0)  # long-run evaluation marks 3, 6, 9


def _frames(*_args: object, **_kwargs: object) -> np.ndarray:
    return np.zeros((2, 4, 6, 3), dtype=np.uint8)


def _write(frames: object, path: Path, fps: int) -> None:
    path.write_bytes(b"mp4")


@pytest.fixture
def renderer(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """``render_rollout`` and ``write_video`` as autospec doubles; the render double is returned."""
    render = create_autospec(render_rollout, side_effect=_frames)
    monkeypatch.setattr(results_mod, "render_rollout", render)
    monkeypatch.setattr(results_mod, "write_video", create_autospec(write_video, side_effect=_write))
    return render


def _long_videos(root: Path) -> list[tuple[float, Path]]:
    return render_long_run_videos(LONG, root, seconds=4.0, size=(4, 6), device="cpu")


def test_long_run_video_per_evaluation_mark_from_its_checkpoint(tmp_path: Path, renderer: MagicMock):
    long_paths = RunPaths(tmp_path, LONG.long_run())
    _fill(CheckpointStore(long_paths.checkpoints_dir), [3.0, 6.0, 9.0])

    videos = _long_videos(tmp_path)

    assert [(mark, p.name) for mark, p in videos] == [
        (3.0, "t0003.000min_it0000010.mp4"),
        (6.0, "t0006.000min_it0000020.mp4"),
        (9.0, "t0009.000min_it0000030.mp4"),
    ]
    assert {p.parent for _, p in videos} == {long_paths.videos_dir}
    assert [c.args[0].iteration for c in renderer.call_args_list] == [10, 20, 30]
    assert {(c.args[1], c.args[2]) for c in renderer.call_args_list} == {(EVAL_COMMANDS[1], 4.0)}


def test_long_run_videos_are_cached(tmp_path: Path, renderer: MagicMock):
    _fill(CheckpointStore(RunPaths(tmp_path, LONG.long_run()).checkpoints_dir), [3.0, 6.0, 9.0])
    first = _long_videos(tmp_path)
    renderer.reset_mock()

    again = _long_videos(tmp_path)

    renderer.assert_not_called()
    assert again == first


def test_long_run_mark_without_a_checkpoint_has_no_video_and_stale_videos_go(tmp_path: Path, renderer: MagicMock):
    long_paths = RunPaths(tmp_path, LONG.long_run())
    _fill(CheckpointStore(long_paths.checkpoints_dir), [3.0])  # the long run stopped after 3 min
    long_paths.videos_dir.mkdir(parents=True)
    (long_paths.videos_dir / "t0003.000min_it0000099.mp4").write_bytes(b"old")  # from an earlier checkpoint set

    videos = _long_videos(tmp_path)

    assert [p.name for _, p in videos] == ["t0003.000min_it0000010.mp4"]
    assert sorted(p.name for p in long_paths.videos_dir.iterdir()) == ["t0003.000min_it0000010.mp4"]


def test_long_run_without_checkpoints_renders_nothing(tmp_path: Path, renderer: MagicMock):
    assert _long_videos(tmp_path) == []

    renderer.assert_not_called()


def test_publish_hub_uploads_the_long_run_videos(tmp_path: Path):
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "t0003.000min_it0000010.mp4").write_bytes(b"mp4")
    publisher = create_autospec(HubPublisher, instance=True)

    assert publish_hub(SWEEP, tmp_path, publisher, long_run_videos=videos) == "ok"

    assert [c.args for c in publisher.upload_folder.call_args_list] == [
        (tmp_path, "results/main", "main: results"),
        (videos, "results/main/long_run_videos", "main: long-run videos"),
    ]


def test_wandb_results_run_logs_each_long_run_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run = create_autospec(wandb.Run, instance=True)  # Image and Video are doubles too: no media files needed
    monkeypatch.setattr(wandb, "init", create_autospec(wandb.init, return_value=run))
    monkeypatch.setattr(wandb, "Image", create_autospec(wandb.Image))
    monkeypatch.setattr(wandb, "Video", create_autospec(wandb.Video))
    videos = [(3.0, tmp_path / "a.mp4"), (6.0, tmp_path / "b.mp4")]

    outcome = results_mod._publish_wandb(SWEEP, tmp_path, "disabled", {}, videos)

    assert outcome == "ok"
    long_logs = [c.args[0] for c in run.log.call_args_list if "results/long_run_video" in c.args[0]]
    assert [entry["results/long_run_train_minutes"] for entry in long_logs] == [3.0, 6.0]
    assert [c.args[0] for c in wandb.Video.call_args_list][-2:] == [str(p) for _, p in videos]  # type: ignore[attr-defined]
    run.define_metric.assert_any_call("results/long_run_video", step_metric="results/long_run_train_minutes")


def test_model_card_names_the_long_run_videos_and_the_small_n_overhead_bias():
    card = model_card(
        SWEEP,
        {},
        reuse={},
        eval_cfg=EvalConfig(),
        primary_metric="vel_error_xy",
        final_points=[],
        race=((1.0, 0.0, 0.0), 4.0, 45.0, {}),
        long_run_video_marks=(3.0, 6.0),
    )

    assert "`long_run_videos/`" in card
    assert "main-long-n04096-s0" in card
    assert "2 evaluation marks" in card
    assert "weigh proportionally more on small-N runs" in card


def test_long_run_videos_render_at_the_requested_height_and_width(tmp_path: Path, renderer: MagicMock):
    _fill(CheckpointStore(RunPaths(tmp_path, LONG.long_run()).checkpoints_dir), [3.0])

    _long_videos(tmp_path)  # size=(4, 6): (height, width)

    assert [(c.kwargs["height"], c.kwargs["width"]) for c in renderer.call_args_list] == [(4, 6)]


def _replace_failing_for(name: str, real: Any) -> Any:
    """``os.replace`` that fails (volume full) for a destination called ``name`` and works for every other."""

    def replace(src: object, dst: object) -> None:
        if Path(str(dst)).name == name:
            raise OSError(28, "No space left on device")
        real(src, dst)

    return replace


def test_failed_eval_config_write_keeps_the_previous_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run = tmp_path / "run"
    _fill(CheckpointStore(run / "checkpoints"), [3.0])
    source = RunSource(
        "main-n00064-s0", 64, 0, run / "checkpoints", run / "eval" / "eval.csv", "main-n00064-s0", 3.0, 3.0
    )
    monkeypatch.setattr(
        results_mod, "evaluate_checkpoint", create_autospec(evaluate_checkpoint, side_effect=_fake_eval)
    )
    evaluate_source(source, [3.0], EvalConfig(horizon_s=1.0), "cpu")
    config_file = run / "eval" / "eval_config.json"
    before = config_file.read_text()
    monkeypatch.setattr(os, "replace", _replace_failing_for("eval_config.json", os.replace))

    with pytest.raises(OSError, match="No space left on device"):
        evaluate_source(source, [3.0], EvalConfig(horizon_s=2.0), "cpu")

    assert config_file.read_text() == before  # never a torn or half-written config
    assert sorted(p.name for p in config_file.parent.iterdir()) == ["eval.csv", "eval_config.json"]


def test_publish_hub_mirrors_the_long_run_videos_and_keeps_the_results_files(tmp_path: Path):
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "t0003.000min_it0000010.mp4").write_bytes(b"mp4")
    publisher = create_autospec(HubPublisher, instance=True)

    publish_hub(SWEEP, tmp_path, publisher, long_run_videos=videos)

    kwargs = [c.kwargs for c in publisher.upload_folder.call_args_list]
    assert kwargs == [{}, {"delete_patterns": "*.mp4"}]  # stale videos go from the Hub; results/main is untouched


def test_publish_hub_records_a_hub_validation_error(tmp_path: Path):
    publisher = create_autospec(HubPublisher, instance=True)
    publisher.ensure_repo.side_effect = HFValidationError("Repo id must use alphanumeric chars: 'bad repo'.")

    assert (
        publish_hub(SWEEP, tmp_path, publisher) == "HFValidationError: Repo id must use alphanumeric chars: 'bad repo'."
    )


# --- build_results on the volume as the sweep left it ---------------------------------------------------
# Manifest with the cuts eval_every_6min and long_run_half, main-n04096-s0 supplied by the pilot (the first
# 45 min of the 90-min long run), main-n16384-s2 failed at 21 min. Evaluation, rendering, the shipped-config
# check (it imports mjlab), W&B and the Hub are autospec doubles; marks, sources, caching, CSV, curves, the
# card and results_publish.json are real.

PLANNED = replace(SWEEP, eval_every_minutes=6.0, long_run_minutes=90.0)
PARTIAL = "main-n16384-s2"
REUSED = "main-n04096-s0"
PILOT_RUN = "main-long-n04096-s0"


def _eval_by_run(run_id: str, ref: CheckpointRef, cfg: EvalConfig, device: str) -> EvalResult:
    xy = 1.0 / (1.0 + ref.train_seconds / 60.0) + 0.001 * (sum(map(ord, run_id)) % 7)
    return EvalResult(run_id, ref.iteration, ref.train_seconds / 60.0, xy, 0.1, 1.0, cfg.horizon_s, 0.3)


def _long_state(root: Path, status: str, minutes: float) -> None:
    spec = PLANNED.long_run()
    state = RunState(spec.run_id, spec.config_hash(), f"{spec.run_id}-w", status, minutes * 60.0, 90.0, "a" * 40)  # type: ignore[arg-type]
    save_state(RunPaths(root, spec).state_file, state)


@pytest.fixture
def volume(tmp_path: Path) -> Path:
    root = tmp_path / "volume"
    out = sweep_results_dir(root, "main")
    out.mkdir(parents=True)
    pilot = RunPaths(root, pilot_spec(PLANNED))
    manifest = {
        "requested": asdict(SWEEP),
        "planned": asdict(PLANNED),
        "excluded": {},
        "cuts": ["eval_every_6min", "long_run_half"],
    }
    (out / "sweep_manifest.json").write_text(json.dumps(manifest))
    reuse = {
        REUSED: {"source_run_dir": str(pilot.run_dir), "max_minutes": 45.0, "git_sha": "a" * 40, "config_hash": "h"}
    }
    (out / "pilot_reuse.json").write_text(json.dumps(reuse))
    for spec in PLANNED.runs():
        if spec.run_id != REUSED:
            end = 21.0 if spec.run_id == PARTIAL else 45.0
            _fill(
                CheckpointStore(RunPaths(root, spec).checkpoints_dir), [3.0 * k for k in range(1, round(end / 3) + 1)]
            )
    _fill(CheckpointStore(pilot.checkpoints_dir), [3.0 * k for k in range(1, 31)])  # the long run, to 90 min
    _long_state(root, "complete", 90.05)
    return root


@pytest.fixture
def doubles(monkeypatch: pytest.MonkeyPatch, renderer: MagicMock) -> dict[str, MagicMock]:
    evaluate = create_autospec(evaluate_checkpoint, side_effect=_eval_by_run)
    monkeypatch.setattr(results_mod, "evaluate_checkpoint", evaluate)
    monkeypatch.setattr(
        results_mod, "_check_shipped_batch_config", create_autospec(results_mod._check_shipped_batch_config)
    )
    publish_wandb = create_autospec(results_mod._publish_wandb, return_value="ok")
    monkeypatch.setattr(results_mod, "_publish_wandb", publish_wandb)
    return {"evaluate": evaluate, "render": renderer, "wandb": publish_wandb}


def _build(
    root: Path, publisher: HubPublisher | None = None, *, metric: str = "vel_error_xy", mode: str = "disabled"
) -> Path:
    return build_results(
        SWEEP,
        root,
        primary_metric=metric,
        publisher=publisher,
        wandb_mode=mode,
        device="cpu",
        render_seconds=0.04,
        render_size=(4, 6),
    )


def _status(root: Path) -> dict[str, str]:
    return json.loads((sweep_results_dir(root, "main").parent / "results_publish.json").read_text())


def test_build_results_over_cuts_pilot_reuse_and_a_failed_run(volume: Path, doubles: dict[str, MagicMock]):
    out = _build(volume, create_autospec(HubPublisher, instance=True))

    rows = list(csv.DictReader((out / "results.csv").read_text().splitlines()))
    marks = ["6.0", "12.0", "18.0", "24.0", "30.0", "36.0", "42.0", "45.0"]  # eval_every_6min, plus the budget
    assert len(rows) == 14 * 8 + 3  # 13 own runs and the reused one at every mark; the failed run up to 18 min
    assert sorted({r["mark_minutes"] for r in rows}, key=float) == marks
    assert [r["mark_minutes"] for r in rows if r["run_id"] == PARTIAL] == ["6.0", "12.0", "18.0"]
    assert [r["mark_minutes"] for r in rows if r["run_id"] == REUSED] == marks
    pilot_calls = [c.args[1] for c in doubles["evaluate"].call_args_list if c.args[0] == PILOT_RUN]
    assert len(pilot_calls) == 8
    assert max(ref.train_seconds for ref in pilot_calls) < 48.0 * 60.0  # nothing past the crossing of 45 min
    assert doubles["evaluate"].call_count == 13 * 8 + 3 + 8
    videos = sorted(p.name for p in RunPaths(volume, PLANNED.long_run()).videos_dir.iterdir())
    assert len(videos) == 15  # long_run_half: marks every 6 min up to 90
    assert videos[-1] == "t0090.000min_it0000300.mp4"
    card = (out / "README.md").read_text()
    assert "`eval_every_6min`, `long_run_half`" in card
    assert f"`{REUSED}` is the pilot's first 45 training minutes" in card
    assert "did not complete" not in card
    assert _status(volume) == {"wandb": "ok", "hub": "ok"}


HUB_FAILURE = UploadFailed("upload_folder failed after 5 attempts: 503")


@pytest.mark.parametrize(
    ("target", "failure", "hub_error", "wandb_outcome"),
    [
        ("hub", "UploadFailed: upload_folder failed after 5 attempts: 503", HUB_FAILURE, "ok"),
        ("wandb", "CommError: 502", None, "CommError: 502"),
    ],
)
def test_build_results_publish_failure_raises_after_recording_it(
    volume: Path,
    doubles: dict[str, MagicMock],
    caplog: pytest.LogCaptureFixture,
    target: str,
    failure: str,
    hub_error: Exception | None,
    wandb_outcome: str,
):
    publisher = create_autospec(HubPublisher, instance=True)
    publisher.upload_folder.side_effect = hub_error
    doubles["wandb"].return_value = wandb_outcome

    with pytest.raises(UploadFailed, match=rf"^sweep main results not published: {target}: {failure}"):
        _build(volume, publisher)

    assert _status(volume)[target] == failure
    assert (sweep_results_dir(volume, "main") / "README.md").is_file()  # local results are complete
    errors = [r for r in caplog.records if r.levelname == "ERROR" and r.name == "mprl.results"]
    assert errors[-1].getMessage() == f"sweep main results not published: {target}: {failure}"


@pytest.mark.parametrize(
    ("status", "minutes", "note"),
    [("failed", 52.0, "`failed` at 52 training minutes"), ("running", 61.5, "`running` at 61.5 training minutes")],
)
def test_build_results_flags_a_long_run_that_did_not_complete(
    volume: Path,
    doubles: dict[str, MagicMock],
    caplog: pytest.LogCaptureFixture,
    status: str,
    minutes: float,
    note: str,
):
    _long_state(volume, status, minutes)

    out = _build(volume)

    card = (out / "README.md").read_text()
    assert f"The long run did not complete ({note})" in card
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING" and r.name == "mprl.results"]
    assert f"long run {PILOT_RUN} did not complete ({note}); results use the checkpoints it saved" in warnings


def test_build_results_flags_a_long_run_without_state(volume: Path, doubles: dict[str, MagicMock]):
    RunPaths(volume, PLANNED.long_run()).state_file.unlink()

    out = _build(volume)

    assert "The long run did not complete (no `state.json`)" in (out / "README.md").read_text()


def test_results_rerun_does_not_log_the_same_media_to_wandb_again(volume: Path, doubles: dict[str, MagicMock]):
    _build(volume, mode="online")
    _build(volume, mode="online")

    assert doubles["wandb"].call_count == 1
    assert _status(volume)["wandb"] == "ok"


def test_results_rerun_logs_to_wandb_when_the_results_changed(volume: Path, doubles: dict[str, MagicMock]):
    _build(volume, mode="online")

    _build(volume, metric="no_fall_fraction", mode="online")

    assert doubles["wandb"].call_count == 2


def test_results_rerun_logs_to_wandb_after_a_failed_publish(volume: Path, doubles: dict[str, MagicMock]):
    doubles["wandb"].return_value = "CommError: 502"
    with pytest.raises(UploadFailed, match="wandb: CommError: 502"):
        _build(volume, mode="online")
    doubles["wandb"].return_value = "ok"

    _build(volume, mode="online")

    assert doubles["wandb"].call_count == 2
