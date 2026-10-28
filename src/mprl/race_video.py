"""Race-video composition: one panel per robot count, one shared training clock banner.

Pure numpy / Pillow / imageio; no simulator. Frames come from ``mprl.render.render_rollout``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mprl.negative_space import require

if TYPE_CHECKING:
    from mprl.checkpoints import CheckpointRef, CheckpointStore

BANNER_HEIGHT: Final = 48
_BANNER_FONT_PX: Final = 28
_LABEL_FONT_PX: Final = 22
_LABEL_MARGIN: Final = 8
_BANNER_BG: Final = (24, 24, 24)
_STARTING_BG: Final = (40, 40, 48)
_TEXT: Final = (255, 255, 255)
_LABEL_SHADOW: Final = (0, 0, 0)
_STARTING_WORD: Final = "starting"


@dataclass(frozen=True)
class Panel:
    label: str
    frames: np.ndarray  # (T, height, width, 3) uint8, T >= 1

    def __post_init__(self) -> None:
        require(
            self.frames.ndim == 4 and self.frames.shape[0] >= 1 and self.frames.shape[3] == 3,
            f"panel {self.label!r}: frames must be (T>=1, H, W, 3), got {self.frames.shape}",
        )
        require(self.frames.dtype == np.uint8, f"panel {self.label!r}: frames must be uint8, got {self.frames.dtype}")


def starting_panel(label: str, *, height: int, width: int) -> Panel:
    """Stand-in for a count with no checkpoint yet: a solid background reading "starting"."""
    image = Image.new("RGB", (width, height), _STARTING_BG)
    font = ImageFont.load_default(size=_BANNER_FONT_PX)
    ImageDraw.Draw(image).text((width // 2, height // 2), _STARTING_WORD, fill=_TEXT, font=font, anchor="mm")
    return Panel(label, np.asarray(image)[None])


def compose_frame(panels: Sequence[Panel], clock_label: str, *, columns: int, index: int = 0) -> np.ndarray:
    """Grid of panels under one clock banner. Panel frame ``index`` is shown (looping if its clip is shorter)."""
    require(len(panels) > 0, "need at least one panel")
    require(columns >= 1, f"columns must be >= 1, got {columns}")
    require(index >= 0, f"index must be >= 0, got {index}")
    height, width = panels[0].frames.shape[1:3]
    for panel in panels:
        require(
            panel.frames.shape[1:3] == (height, width),
            f"panel {panel.label!r} is {panel.frames.shape[1:3]}, expected {(height, width)}",
        )
    columns = min(columns, len(panels))
    rows = -(-len(panels) // columns)
    canvas = Image.new("RGB", (columns * width, BANNER_HEIGHT + rows * height), _BANNER_BG)
    draw = ImageDraw.Draw(canvas)
    banner_font = ImageFont.load_default(size=_BANNER_FONT_PX)
    draw.text((canvas.width // 2, BANNER_HEIGHT // 2), clock_label, fill=_TEXT, font=banner_font, anchor="mm")
    label_font = ImageFont.load_default(size=_LABEL_FONT_PX)
    for i, panel in enumerate(panels):
        row, col = divmod(i, columns)
        left, top = col * width, BANNER_HEIGHT + row * height
        canvas.paste(Image.fromarray(panel.frames[index % panel.frames.shape[0]]), (left, top))
        if panel.label:
            at = (left + _LABEL_MARGIN, top + _LABEL_MARGIN)
            draw.text((at[0] + 1, at[1] + 1), panel.label, fill=_LABEL_SHADOW, font=label_font)
            draw.text(at, panel.label, fill=_TEXT, font=label_font)
    return np.asarray(canvas)


def race_timeline(
    checkpoints: Mapping[int, CheckpointStore], marks_minutes: Sequence[float]
) -> list[tuple[float, dict[int, CheckpointRef | None]]]:
    """Per mark, each env count's latest checkpoint at or before it (``None`` before its first).

    A checkpoint later than a mark is never chosen for it, so a store holding more training than the
    last mark (a reused longer pilot) is capped by the marks.
    """
    require(len(checkpoints) > 0, "need at least one env count")
    return [(float(mark), {n: store.at_or_before(mark) for n, store in checkpoints.items()}) for mark in marks_minutes]


def write_video(frames: Iterable[np.ndarray], path: Path, fps: int) -> None:
    """Stream ``frames`` (H, W, 3 uint8, even H and W for yuv420p) to an mp4 via imageio's ffmpeg writer."""
    require(fps >= 1, f"fps must be >= 1, got {fps}")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    count = 0
    try:
        for frame in frames:
            if writer is None:
                require(
                    frame.ndim == 3 and frame.shape[2] == 3 and frame.shape[0] % 2 == 0 and frame.shape[1] % 2 == 0,
                    f"frames must be (even H, even W, 3), got {frame.shape}",
                )
                writer = imageio.get_writer(str(path), fps=fps, macro_block_size=1)
            writer.append_data(frame)
            count += 1
    finally:
        if writer is not None:
            writer.close()
    require(count > 0, "no frames to write")
