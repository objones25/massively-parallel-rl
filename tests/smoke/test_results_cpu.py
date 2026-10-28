"""Smoke test: build_results end to end on a tiny real sweep (G1 flat, CPU, W&B offline, no publisher).

Four runs (4 and 8 envs x seeds 0 and 1) train for 6 s of training clock each with a checkpoint every
3 s (0.1 / 0.05 min). Evaluation is one env per command for 0.2 s and one seed; the race clips are
0.1 s at 64 x 96 so the whole test stays around a minute and a half.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import pytest
import wandb

import mprl.results as results_mod
from mprl.config import SweepSpec
from mprl.evaluator import EvalConfig, read_eval_csv
from mprl.paths import RunPaths, sweep_results_dir
from mprl.results import build_results
from mprl.train import TrainDeps, train_run

pytestmark = pytest.mark.smoke

SWEEP = SweepSpec(
    "smoke", env_counts=(4, 8), seeds=(0, 1), budget_minutes=0.1, checkpoint_every_minutes=0.05, eval_every_minutes=0.05
)
EVAL = EvalConfig(seeds=(0,), envs_per_command=1, horizon_s=0.2)
SMALL: dict[str, Any] = {"eval_cfg": EVAL, "render_seconds": 0.1, "render_size": (64, 96)}


@pytest.fixture(autouse=True)
def _offline_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """wandb reads WANDB_DIR once per service; restart it so the test sees its own tmp_path."""
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    wandb.teardown()
    yield
    wandb.teardown()


def test_build_results_end_to_end_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "volume"
    deps = TrainDeps(publisher=None, wandb_mode="offline", device="cpu")
    for spec in SWEEP.runs():
        train_run(spec, root, deps)

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
