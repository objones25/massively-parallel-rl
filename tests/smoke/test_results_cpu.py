"""Smoke test: build_results end to end on a tiny sweep (G1 flat, CPU, W&B offline, no publisher).

Training is not the behavior here, so the four runs (4 and 8 envs x seeds 0 and 1, a 6 s budget with a
checkpoint every 3 s: 0.1 / 0.05 min) are laid out from copies of the shared real ``g1_checkpoint``,
saved through ``CheckpointStore`` at 3 s and 6 s of training clock, the stamps a real run leaves at the
two marks. Everything build_results does is real: evaluation of all 8 checkpoints (one env per command
for 0.2 s, one seed), the curves, the race clips (0.1 s at 64 x 96) and the video encode.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import pytest

import mprl.results as results_mod
from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.config import SweepSpec
from mprl.evaluator import EvalConfig, read_eval_csv
from mprl.paths import RunPaths, sweep_results_dir
from mprl.results import build_results
from mprl.run_state import RunState, save_state

pytestmark = [pytest.mark.smoke, pytest.mark.usefixtures("offline_wandb")]

SWEEP = SweepSpec(
    "smoke",
    env_counts=(4, 8),
    seeds=(0, 1),
    budget_minutes=0.1,
    checkpoint_every_minutes=0.05,
    eval_every_minutes=0.05,
    long_run_minutes=0.1,  # the long run's two marks get a video each
)
EVAL = EvalConfig(seeds=(0,), envs_per_command=1, horizon_s=0.2)
SMALL: dict[str, Any] = {"eval_cfg": EVAL, "render_seconds": 0.1, "render_size": (64, 96)}


CHECKPOINTS = ((1, 3.0), (3, 6.0))  # (iteration, training-clock s): one crossing each mark


def _lay_out_sweep(root: Path, weights: CheckpointRef) -> None:
    """Every run of SWEEP and its long run as completed runs whose checkpoints hold ``weights``' real parameters."""
    for spec in [*SWEEP.runs(), SWEEP.long_run()]:
        paths = RunPaths(root, spec)
        store = CheckpointStore(paths.checkpoints_dir)
        for iteration, seconds in CHECKPOINTS:
            store.save_atomic(lambda path: shutil.copyfile(weights.path, path), iteration, seconds)
        state = RunState(spec.run_id, spec.config_hash(), f"{spec.run_id}-w", "complete", 6.0, 0.1, "sha", {})
        save_state(paths.state_file, state)


def test_build_results_end_to_end_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, g1_checkpoint: CheckpointRef):
    root = tmp_path / "volume"
    _lay_out_sweep(root, g1_checkpoint)

    out = build_results(
        SWEEP, root, primary_metric="vel_error_xy", publisher=None, wandb_mode="offline", device="cpu", **SMALL
    )

    assert out == sweep_results_dir(root, "smoke")
    sizes = {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}
    assert {"results.csv", "learning_curves.png", "learning_curves.svg", "race.mp4", "README.md"} <= set(sizes)
    assert min(sizes.values()) > 0
    reader = imageio.get_reader(str(out / "race.mp4"))
    try:
        assert reader.count_frames() > 0
    finally:
        reader.close()
    csv_lines = (out / "results.csv").read_text().splitlines()
    assert csv_lines[0].startswith("mark_minutes,num_envs,seed,run_id,iteration,train_minutes,vel_error_xy")
    rows = {tuple(line.split(",")[:4]) for line in csv_lines[1:]}
    every_mark = {
        (repr(mark), str(spec.num_envs), str(spec.seed), spec.run_id) for spec in SWEEP.runs() for mark in (0.05, 0.1)
    }
    assert rows == every_mark  # every run has its crossing checkpoint at every mark, the budget mark included
    assert "vel_error_xy" in (out / "README.md").read_text()
    videos = sorted(p.name for p in RunPaths(root, SWEEP.long_run()).videos_dir.iterdir())
    assert videos == ["t0000.050min_it0000001.mp4", "t0000.100min_it0000003.mp4"]
    assert "`long_run_videos/`: the long run `smoke-long-n04096-s0`" in (out / "README.md").read_text()
    status = json.loads((out.parent / "results_publish.json").read_text())
    assert status == {"wandb": "ok", "hub": "skipped: no publisher"}
    first_eval = {spec.run_id: read_eval_csv(RunPaths(root, spec).eval_dir / "eval.csv") for spec in SWEEP.runs()}
    assert all(first_eval.values())

    # Re-running evaluates nothing new and leaves every cached evaluation as it was.
    calls: list[str] = []
    real = results_mod.evaluate_checkpoint

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(results_mod, "evaluate_checkpoint", counting)
    build_results(
        SWEEP, root, primary_metric="vel_error_xy", publisher=None, wandb_mode="offline", device="cpu", **SMALL
    )

    assert calls == []
    for spec in SWEEP.runs():
        assert read_eval_csv(RunPaths(root, spec).eval_dir / "eval.csv") == first_eval[spec.run_id]
    assert (out / "results.csv").read_text().splitlines() == csv_lines
