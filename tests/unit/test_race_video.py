from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.negative_space import CheckFailed
from mprl.race_video import (
    BANNER_HEIGHT,
    Panel,
    compose_frame,
    race_frames,
    race_timeline,
    starting_panel,
    write_video,
)

H, W = 360, 480


def _solid(color: int, frames: int = 2) -> np.ndarray:
    return np.full((frames, H, W, 3), color, dtype=np.uint8)


OVERSHOOT_S = 2.0  # a save lands after the iteration that crosses its mark


def _store(tmp_path: Path, name: str, minutes: list[float]) -> CheckpointStore:
    """Checkpoints for the given marks, each stamped a realistic 2 s past its mark."""
    store = CheckpointStore(tmp_path / name)
    for i, m in enumerate(minutes, start=1):
        store.save_atomic(lambda p: p.write_bytes(b"w"), i, m * 60.0 + OVERSHOOT_S)
    return store


def _mark_of(ref: CheckpointRef | None) -> float | None:
    return None if ref is None else (ref.train_seconds - OVERSHOOT_S) / 60.0


def test_compose_frame_shape():
    panels = [Panel(f"{n} robots", _solid(10 * n)) for n in range(1, 6)]
    frame = compose_frame(panels, "Training time 12:00 / 45:00", columns=3)
    assert frame.dtype == np.uint8
    assert frame.shape == (BANNER_HEIGHT + 2 * H, 3 * W, 3)


def test_panel_order_preserved():
    colors = [20, 60, 100, 140, 180]
    panels = [Panel(f"p{i}", _solid(c)) for i, c in enumerate(colors)]
    frame = compose_frame(panels, "clock", columns=3)
    for i, color in enumerate(colors):
        row, col = divmod(i, 3)
        y = BANNER_HEIGHT + row * H + H // 2
        x = col * W + W // 2
        assert frame[y, x].tolist() == [color] * 3


def test_label_is_drawn_in_panel_corner():
    plain = compose_frame([Panel("", _solid(0))], "", columns=1)
    labelled = compose_frame([Panel("4,096 robots", _solid(0))], "", columns=1)
    corner = (slice(BANNER_HEIGHT, BANNER_HEIGHT + 40), slice(0, 200))
    assert plain[corner].max() == 0
    assert labelled[corner].max() > 0


def test_clock_banner_is_drawn():
    plain = compose_frame([Panel("a", _solid(0))], "", columns=1)
    banner = compose_frame([Panel("a", _solid(0))], "Training time 12:00 / 45:00", columns=1)
    assert not np.array_equal(plain[:BANNER_HEIGHT], banner[:BANNER_HEIGHT])


def test_frame_index_selects_and_loops():
    ramp = np.stack([np.full((H, W, 3), c, np.uint8) for c in (10, 20, 30)])
    mid = (BANNER_HEIGHT + H // 2, W // 2)
    assert compose_frame([Panel("a", ramp)], "", columns=1, index=1)[mid].tolist() == [20] * 3
    assert compose_frame([Panel("a", ramp)], "", columns=1, index=4)[mid].tolist() == [20] * 3


def test_starting_panel_is_a_labelled_solid_frame():
    panel = starting_panel("64 robots", height=H, width=W)
    assert panel.label == "64 robots"
    assert panel.frames.shape == (1, H, W, 3)
    frame = panel.frames[0]
    assert frame[H // 2, 0].tolist() == frame[H - 1, W - 1].tolist()
    assert frame.max() > frame.min()  # the word "starting" is drawn on the solid background
    composed = compose_frame([panel, Panel("b", _solid(200))], "t", columns=2)
    assert composed.shape == (BANNER_HEIGHT + H, 2 * W, 3)


def test_compose_rejects_mismatched_panel_sizes():
    with pytest.raises(CheckFailed):
        compose_frame([Panel("a", _solid(1)), Panel("b", np.zeros((1, H, W + 2, 3), np.uint8))], "", columns=2)


def test_timeline_picks_the_checkpoint_that_crossed_each_mark(tmp_path: Path):
    stores = {64: _store(tmp_path, "a", [3, 6, 9]), 256: _store(tmp_path, "b", [3, 6, 9, 12])}
    timeline = race_timeline(stores, [6.0, 9.0, 12.0], 3.0)
    assert [mark for mark, _ in timeline] == [6.0, 9.0, 12.0]
    picked = {mark: {n: _mark_of(ref) for n, ref in refs.items()} for mark, refs in timeline}
    # 64 stopped after 9:00, so 12:00 has nothing for it (absent, not its 9:00 policy again).
    assert picked == {6.0: {64: 6.0, 256: 6.0}, 9.0: {64: 9.0, 256: 9.0}, 12.0: {64: None, 256: 12.0}}


def test_timeline_none_before_first_checkpoint(tmp_path: Path):
    stores = {64: _store(tmp_path, "a", [6, 9]), 256: _store(tmp_path, "b", [])}
    (_, first), (_, second) = race_timeline(stores, [3.0, 6.0], 3.0)
    assert first == {64: None, 256: None}
    assert second[64] is not None
    assert second[256] is None


def test_timeline_never_picks_checkpoint_after_last_mark(tmp_path: Path):
    stores = {64: _store(tmp_path, "a", [3, 6, 45, 48, 90])}
    timeline = race_timeline(stores, [3.0, 45.0], 3.0)
    assert [_mark_of(refs[64]) for _, refs in timeline] == [3.0, 45.0]


def test_write_video_rejects_empty(tmp_path: Path):
    with pytest.raises(CheckFailed):
        write_video(iter(()), tmp_path / "x.mp4", fps=5)


def test_race_frames_render_each_checkpoint_once(tmp_path: Path):
    store = _store(tmp_path, "n64", [3.0])
    ref = store.list()[0]
    timeline = [(0.0, {64: None, 256: None}), (3.0, {64: ref, 256: None}), (6.0, {64: ref, 256: None})]
    rendered: list[Path] = []

    def render(ref):
        rendered.append(ref.path)
        return _solid(200, frames=3)

    frames = list(race_frames(timeline, {64: "64", 256: "256"}, 6.0, render, frames_per_mark=4, size=(H, W), columns=3))

    assert len(frames) == 3 * 4
    assert rendered == [store.list()[0].path]
    assert frames[0][BANNER_HEIGHT + H // 2, W // 2].tolist() != [200, 200, 200]  # "starting" before 3 min
    assert frames[-1][BANNER_HEIGHT + H // 2, W // 2].tolist() == [200, 200, 200]
