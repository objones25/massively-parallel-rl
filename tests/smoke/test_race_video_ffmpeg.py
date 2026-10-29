"""Smoke test: the race video writer round-trips through the real ffmpeg (a subprocess, so not the fast tier)."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from mprl.race_video import write_video

pytestmark = pytest.mark.smoke


def test_write_video_round_trip(tmp_path: Path) -> None:
    frames = (np.full((64, 96, 3), 40 * i, dtype=np.uint8) for i in range(6))
    path = tmp_path / "out" / "race.mp4"

    write_video(frames, path, fps=5)

    with imageio.get_reader(str(path)) as reader:
        assert reader.count_frames() == 6
