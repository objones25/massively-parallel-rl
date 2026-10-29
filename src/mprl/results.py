"""Results for a finished sweep: evaluation at the marks, CSV, learning curves, race video, model card,
then a W&B ``results`` run and the Hugging Face upload.

The runs are those of the *planned* sweep in ``sweep_manifest.json`` (counts, seeds and eval cadence
after any cut); without a manifest, the requested ``sweep``. A run the pilot stood in for
(``pilot_reuse.json``) reads the pilot's checkpoints, up to the one that crossed ``max_minutes``, and its
evaluations are cached in the pilot's own ``eval/eval.csv`` (they belong to its checkpoints).

The long run (``<sweep>-long``) gets a rollout video at each of its evaluation marks (every
``eval_every_minutes`` up to ``long_run_minutes``): ``render_rollout`` of the checkpoint that crossed the mark on
``RACE_COMMAND`` (``EVAL_COMMANDS[1]``), written to the long run's ``videos/`` as
``t<mark minutes>min_it<iteration>.mp4``, cached by file name (a re-run renders only what is new and removes
videos of checkpoints no longer marked), logged to the W&B results run and uploaded to
``results/<sweep>/long_run_videos`` on the Hub. Training is untouched: this runs during ``results`` only.

Evaluation marks are every ``eval_every_minutes`` plus always the budget; race-video marks every
``checkpoint_every_minutes`` plus the budget. A mark takes the checkpoint that crossed it
(``CheckpointStore.crossed`` with the checkpoint interval); a mark with none is absent. Results stay
keyed by mark from here on (CSV, curves, median seed). An evaluation is cached per (run, checkpoint
iteration) in the run's ``eval/eval.csv``, with the ``EvalConfig`` hash in ``eval/eval_config.json``;
a row is reused only under the same config hash and while its training time still matches, so
re-running evaluates only what is new. Everything is off the
training clock. Publishing outcomes go to ``<sweep>/results_publish.json`` (next to the results dir, so
it is not uploaded stale) and the log; a publish failure never touches the local results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, astuple, dataclass, fields
from pathlib import Path
from typing import Any, Final

import numpy as np
import wandb
from huggingface_hub.errors import HfHubHTTPError
from wandb.errors import Error as WandbError

from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.config import TASK_ID, SweepSpec, batch_arithmetic
from mprl.curves import aggregate, median_seed_run, plot_learning_curves, validate_metric
from mprl.errors import UploadFailed
from mprl.evaluator import EVAL_COMMANDS, EvalConfig, EvalResult, evaluate_checkpoint, read_eval_csv, write_eval_csv
from mprl.hub import HubPublisher
from mprl.negative_space import bounded, require
from mprl.paths import RunPaths, sweep_results_dir
from mprl.race_video import race_frames, race_timeline, write_video
from mprl.render import render_rollout
from mprl.results_card import model_card
from mprl.sweep import pilot_spec
from mprl.tasks import load_task_rl_cfg
from mprl.train import DEFAULT_WANDB_PROJECT

log = logging.getLogger(__name__)

RACE_COMMAND: Final = EVAL_COMMANDS[1]  # 1.0 m/s straight ahead: the most legible walking demo
RACE_SECONDS_PER_MARK: Final = 4.0  # 15 marks x 4 s = a 60 s video at real time
RACE_FPS: Final = 50  # 1 / step_dt (decimation 4 x 0.005 s): real time
_MAX_MARKS: Final = 10_000
_MARK_TOLERANCE_MIN: Final = 1e-9


@dataclass(frozen=True)
class RunSource:
    """Where a sweep run's checkpoints and evaluation cache live (the pilot's, for a reused run)."""

    run_id: str
    num_envs: int
    seed: int
    checkpoints_dir: Path
    eval_csv: Path
    source_run_id: str
    max_minutes: float
    interval_minutes: float  # checkpoint interval: the width of a mark's crossing window


def eval_marks(every_minutes: float, budget_minutes: float) -> list[float]:
    """``every, 2 * every, ...`` below ``budget``, then ``budget`` itself (always a mark)."""
    require(0 < every_minutes <= budget_minutes, f"bad cadence {every_minutes} for budget {budget_minutes}")
    count = math.ceil(budget_minutes / every_minutes - _MARK_TOLERANCE_MIN) - 1  # multiples strictly below
    require(count < _MAX_MARKS, f"{count} marks exceed {_MAX_MARKS}")
    return [*(round(every_minutes * k, 9) for k in range(1, count + 1)), budget_minutes]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    require(isinstance(data, dict), f"{path} does not hold a JSON object")
    return data


def planned_sweep(sweep: SweepSpec, root: Path) -> tuple[SweepSpec, dict[str, Any]]:
    """The sweep as planned (after cuts and exclusions) and its manifest; ``(sweep, {})`` without one."""
    manifest = _read_json(sweep_results_dir(root, sweep.name) / "sweep_manifest.json")
    if not manifest:
        return sweep, {}
    planned = dict(manifest["planned"])
    planned["env_counts"], planned["seeds"] = tuple(planned["env_counts"]), tuple(planned["seeds"])
    spec = SweepSpec(**planned)
    require(spec.name == sweep.name, f"manifest is for sweep {spec.name!r}, not {sweep.name!r}")
    return spec, manifest


def run_sources(sweep: SweepSpec, root: Path) -> list[RunSource]:
    reuse = _read_json(sweep_results_dir(root, sweep.name) / "pilot_reuse.json")
    pilot = RunPaths(root, pilot_spec(sweep))
    sources: list[RunSource] = []
    for spec in sweep.runs():
        own = RunPaths(root, spec)
        entry = reuse.get(spec.run_id)
        if entry is None:
            paths, max_minutes = own, sweep.budget_minutes
        else:
            require(entry["source_run_dir"] == str(pilot.run_dir), f"unexpected reuse source {entry}")
            paths, max_minutes = pilot, float(entry["max_minutes"])
            require(sweep.budget_minutes <= max_minutes, f"budget exceeds the reused {max_minutes} min")
        sources.append(
            RunSource(
                spec.run_id,
                spec.num_envs,
                spec.seed,
                paths.checkpoints_dir,
                paths.eval_dir / "eval.csv",
                paths.spec.run_id,
                max_minutes,
                sweep.checkpoint_every_minutes,
            )
        )
    return sources


def marked_checkpoints(
    store: CheckpointStore, marks: Sequence[float], *, interval_minutes: float, max_minutes: float
) -> dict[float, CheckpointRef | None]:
    """Per mark, the checkpoint that crossed it. No mark may lie past ``max_minutes``, so nothing is taken
    beyond the checkpoint that crossed ``max_minutes`` (a reused pilot's cap)."""
    for mark in marks:
        require(mark <= max_minutes + _MARK_TOLERANCE_MIN, f"mark {mark} is past {max_minutes} min")
    marked = {mark: store.crossed(mark, interval_minutes) for mark in marks}
    cap_s = (max_minutes + interval_minutes) * 60.0
    for ref in marked.values():
        require(ref is None or ref.train_seconds < cap_s, f"{ref} is past the checkpoint crossing {max_minutes} min")
    return marked


def eval_config_hash(cfg: EvalConfig) -> str:
    return hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()


def _config_file(eval_csv: Path) -> Path:
    return eval_csv.with_name("eval_config.json")


def _read_eval_cache(eval_csv: Path, config_hash: str) -> dict[int, EvalResult]:
    """Cached rows by iteration, or nothing when they were scored under another (or unknown) config."""
    stored = _read_json(_config_file(eval_csv)).get("config_hash")
    if not eval_csv.is_file() or stored != config_hash:
        return {}
    return {r.iteration: r for r in read_eval_csv(eval_csv)}


def _write_eval_cache(results: Sequence[EvalResult], eval_csv: Path, cfg: EvalConfig) -> None:
    """CSV first, then the config hash: a crash between the two leaves a mismatch, which re-evaluates."""
    tmp = eval_csv.with_name(f".{eval_csv.name}.tmp")
    write_eval_csv(sorted(results, key=lambda r: r.iteration), tmp)
    os.replace(tmp, eval_csv)
    config = {"config_hash": eval_config_hash(cfg), "eval_config": asdict(cfg)}
    _config_file(eval_csv).write_text(json.dumps(config, indent=2) + "\n")


def evaluate_source(
    source: RunSource, marks: Sequence[float], cfg: EvalConfig, device: str
) -> dict[float, EvalResult | None]:
    """Per mark, the evaluation of its checkpoint (cached or new), labelled with the sweep run id."""
    cache = _read_eval_cache(source.eval_csv, eval_config_hash(cfg))
    marked = marked_checkpoints(
        CheckpointStore(source.checkpoints_dir),
        marks,
        interval_minutes=source.interval_minutes,
        max_minutes=source.max_minutes,
    )
    for ref in sorted({ref for ref in marked.values() if ref is not None}):
        hit = cache.get(ref.iteration)
        if hit is None or not math.isclose(hit.train_minutes, ref.train_seconds / 60.0):
            log.info("evaluating %s iteration %d", source.source_run_id, ref.iteration)
            cache[ref.iteration] = evaluate_checkpoint(source.source_run_id, ref, cfg, device)
            _write_eval_cache(list(cache.values()), source.eval_csv, cfg)  # progress survives a crash
    out: dict[float, EvalResult | None] = {}
    for mark, ref in marked.items():
        result = None if ref is None else cache[ref.iteration]
        out[mark] = None if result is None else _relabel(result, source.run_id)
    return out


def _relabel(result: EvalResult, run_id: str) -> EvalResult:
    return EvalResult(run_id, *astuple(result)[1:])


def write_results_csv(sources: Sequence[RunSource], evals: Mapping[str, Mapping[float, EvalResult | None]], path: Path):
    """One row per (run, mark) that has an evaluation; absent marks have no row."""
    header = ["mark_minutes", "num_envs", "seed", *(f.name for f in fields(EvalResult))]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for source in sources:
            for mark, result in evals[source.run_id].items():
                if result is not None:
                    row = [mark, source.num_envs, source.seed, *astuple(result)]
                    writer.writerow([repr(v) if isinstance(v, float) else str(v) for v in row])


def _render_race(
    sources: Sequence[RunSource],
    median: Mapping[int, str],
    sweep: SweepSpec,
    out: Path,
    *,
    seconds: float,
    size: tuple[int, int],
    device: str,
) -> None:
    require(len(median) > 0, "no env count has an evaluated checkpoint; nothing to race")
    require(size[0] % 2 == 0 and size[1] % 2 == 0, f"render size must be even, got {size}")
    by_id = {s.run_id: s for s in sources}
    chosen = {n: by_id[run_id] for n, run_id in sorted(median.items())}
    stores = {n: CheckpointStore(s.checkpoints_dir) for n, s in chosen.items()}
    marks = eval_marks(sweep.checkpoint_every_minutes, sweep.budget_minutes)
    require(all(marks[-1] <= s.max_minutes for s in chosen.values()), "a race mark is past a run's cap")
    timeline = race_timeline(stores, marks, sweep.checkpoint_every_minutes)

    def render(ref: CheckpointRef) -> np.ndarray:
        return render_rollout(ref, RACE_COMMAND, seconds, height=size[0], width=size[1], device=device)

    labels = {n: f"{n} envs (seed {s.seed})" for n, s in chosen.items()}
    frames = race_frames(
        timeline, labels, sweep.budget_minutes, render, frames_per_mark=round(seconds * RACE_FPS), size=size, columns=3
    )
    tmp = out.with_name(f".{out.stem}.tmp{out.suffix}")
    write_video(frames, tmp, RACE_FPS)
    os.replace(tmp, out)


def _long_run_video_name(mark: float, ref: CheckpointRef) -> str:
    return f"t{mark:08.3f}min_it{ref.iteration:07d}.mp4"


def render_long_run_videos(
    sweep: SweepSpec, root: Path, *, seconds: float, size: tuple[int, int], device: str
) -> list[tuple[float, Path]]:
    """``(mark, mp4)`` per long-run evaluation mark that a checkpoint crossed, rendering only the missing ones."""
    require(seconds > 0.0, f"seconds must be positive, got {seconds}")
    require(size[0] % 2 == 0 and size[1] % 2 == 0, f"render size must be even, got {size}")
    paths = RunPaths(root, sweep.long_run())
    marks = eval_marks(sweep.eval_every_minutes, sweep.long_run_minutes)
    marked = marked_checkpoints(
        CheckpointStore(paths.checkpoints_dir),
        marks,
        interval_minutes=sweep.checkpoint_every_minutes,
        max_minutes=sweep.long_run_minutes,
    )
    videos: list[tuple[float, Path]] = []
    for mark, ref in bounded(marked.items(), len(marked), "long-run marks"):
        if ref is None:
            continue
        path = paths.videos_dir / _long_run_video_name(mark, ref)
        if not path.is_file():
            log.info("rendering %s at %g min (iteration %d)", paths.spec.run_id, mark, ref.iteration)
            frames = render_rollout(ref, RACE_COMMAND, seconds, height=size[0], width=size[1], device=device)
            tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            write_video(iter(frames), tmp, RACE_FPS)
            os.replace(tmp, path)
        videos.append((mark, path))
    keep = {path.name for _, path in videos}
    if paths.videos_dir.is_dir():
        for stale in paths.videos_dir.glob("t*min_it*.mp4"):
            if stale.name not in keep:
                stale.unlink()  # a checkpoint no longer marked (e.g. the run was resumed): not to be uploaded
    require(len(videos) <= len(marks), f"{len(videos)} videos for {len(marks)} marks")
    return videos


def _check_shipped_batch_config() -> None:
    """The chart caption and model card use ``batch_arithmetic``'s defaults; they must be the shipped config."""
    rl = load_task_rl_cfg(TASK_ID)
    shipped = batch_arithmetic(1, rl.num_steps_per_env, rl.algorithm.num_mini_batches)
    require(shipped == batch_arithmetic(1), f"shipped batch config {shipped} differs from the defaults")


def _publish_wandb(
    sweep: SweepSpec,
    out: Path,
    mode: str,
    config: Mapping[str, Any],
    long_run_videos: Sequence[tuple[float, Path]] = (),
) -> str:
    """Log the chart, the race video and the long-run videos (x-axis: training minutes) to the sweep group's
    ``results`` run; "ok" or the error text."""
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
            entity=os.environ.get("WANDB_ENTITY") or None,
            id=f"{sweep.name}-results",  # one results run per sweep: re-running resumes it
            resume="allow",
            name=f"{sweep.name}-results",
            group=sweep.name,
            job_type="results",
            mode=mode,  # pyright: ignore[reportArgumentType]  validated by wandb
            config=dict(config),
        )
        run.log(
            {
                "results/learning_curves": wandb.Image(str(out / "learning_curves.png")),
                "results/race": wandb.Video(str(out / "race.mp4"), format="mp4"),
            }
        )
        run.define_metric("results/long_run_train_minutes")
        run.define_metric("results/long_run_video", step_metric="results/long_run_train_minutes")
        for mark, video in long_run_videos:
            run.log(
                {
                    "results/long_run_video": wandb.Video(str(video), format="mp4"),
                    "results/long_run_train_minutes": mark,
                }
            )
    except (WandbError, OSError) as error:
        log.error("W&B results run for %s failed: %s", sweep.name, error)
        return f"{type(error).__name__}: {error}"
    finally:
        wandb.finish()
    return "ok"


def publish_hub(
    sweep: SweepSpec, out: Path, publisher: HubPublisher | None, *, long_run_videos: Path | None = None
) -> str:
    """Upload the results dir, then the long-run videos dir when it holds any; "ok" or the error text."""
    if publisher is None:
        return "skipped: no publisher"
    try:
        publisher.ensure_repo()
        publisher.upload_folder(out, f"results/{sweep.name}", f"{sweep.name}: results")
        if long_run_videos is not None and any(long_run_videos.glob("*.mp4")):
            publisher.upload_folder(
                long_run_videos, f"results/{sweep.name}/long_run_videos", f"{sweep.name}: long-run videos"
            )
    except (UploadFailed, HfHubHTTPError, OSError) as error:
        log.error("upload of %s results failed: %s", sweep.name, error)
        return f"{type(error).__name__}: {error}"
    return "ok"


def build_results(
    sweep: SweepSpec,
    root: Path,
    *,
    primary_metric: str,
    publisher: HubPublisher | None,
    wandb_mode: str,
    device: str,
    eval_cfg: EvalConfig | None = None,
    render_seconds: float = RACE_SECONDS_PER_MARK,
    render_size: tuple[int, int] = (360, 480),
) -> Path:
    """Evaluate, write ``results.csv``, the chart, ``race.mp4``, the long-run videos and ``README.md``, then
    publish; the results dir."""
    validate_metric(primary_metric)
    cfg = eval_cfg or EvalConfig()
    _check_shipped_batch_config()
    planned, manifest = planned_sweep(sweep, root)
    out = sweep_results_dir(root, planned.name)
    out.mkdir(parents=True, exist_ok=True)
    sources = run_sources(planned, root)
    marks = eval_marks(planned.eval_every_minutes, planned.budget_minutes)
    evals = {s.run_id: evaluate_source(s, marks, cfg, device) for s in bounded(sources, len(sources), "runs")}
    write_results_csv(sources, evals, out / "results.csv")

    run_envs = {s.run_id: s.num_envs for s in sources}
    points = aggregate(evals, primary_metric, run_envs, marks)
    require(len(points) > 0, f"sweep {planned.name}: no run has a checkpoint at any mark; nothing to show")
    plot_learning_curves(points, primary_metric, out / "learning_curves.png")
    last = max(p.minutes for p in points)  # the budget, unless every run stopped before it
    median = median_seed_run(evals, primary_metric, run_envs, at_minutes=last)
    _render_race(sources, median, planned, out / "race.mp4", seconds=render_seconds, size=render_size, device=device)
    long_videos = render_long_run_videos(planned, root, seconds=render_seconds, size=render_size, device=device)

    reuse = _read_json(out / "pilot_reuse.json")
    card = model_card(
        planned,
        manifest,
        reuse=reuse,
        eval_cfg=cfg,
        primary_metric=primary_metric,
        final_points=[p for p in points if p.minutes == last],
        race=(RACE_COMMAND, render_seconds, last, median),
        long_run_video_marks=[mark for mark, _ in long_videos],
    )
    (out / "README.md").write_text(card)

    config = {"sweep": planned.name, "primary_metric": primary_metric, "median_runs": median, "median_at": last}
    videos_dir = RunPaths(root, planned.long_run()).videos_dir
    status = {
        "wandb": _publish_wandb(planned, out, wandb_mode, config, long_videos),
        "hub": publish_hub(planned, out, publisher, long_run_videos=videos_dir),
    }
    (out.parent / "results_publish.json").write_text(json.dumps(status, indent=2) + "\n")
    for target, outcome in status.items():
        if outcome != "ok":
            log.warning("sweep %s results %s: %s", planned.name, target, outcome)
    return out
