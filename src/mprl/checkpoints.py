"""Atomic, minute-indexed checkpoint store.

A checkpoint becomes visible only through ``os.replace`` of a fully written temp file in the same
directory, and ``list()`` parses only names matching the exact final pattern, so a partial file
(pod killed mid-write) is never reported. The temp file is fsynced before the rename and the directory
after it, so a host or volume failure cannot leave a truncated file under a final name, which
``latest()`` would pick and every resume would fail to load.

Should a file under a final name still be damaged (a volume fault after the fsync, a copy gone wrong),
``integrity_error`` finds it without unpickling (every checkpoint is a ``torch.save`` zip archive whose
members carry CRC-32s) and ``quarantine`` renames it to ``<name>.corrupt``, which ``list()`` no longer
matches, so a resume falls back to the next-older checkpoint.
"""

from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mprl.negative_space import require

if TYPE_CHECKING:
    from collections.abc import Callable

_STAMP_TOLERANCE_S = 1e-6  # marks are float minutes (0.05 * 60 = 3.0000000000000004); stamps are whole seconds
_NAME = re.compile(r"^model_it(\d{7})_s(\d{6})\.pt$")
_CORRUPT_SUFFIX = ".corrupt"
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)


@dataclass(frozen=True, order=True)
class CheckpointRef:
    iteration: int
    train_seconds: float
    path: Path


def _fsync_path(path: Path, flags: int) -> None:
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class CheckpointStore:
    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)

    def save_atomic(
        self, write: Callable[[Path], None], iteration: int, train_seconds: float
    ) -> CheckpointRef:
        require(0 <= iteration <= 9_999_999, f"iteration out of range: {iteration}")
        require(train_seconds >= 0.0, f"train_seconds out of range: {train_seconds}")
        seconds = round(train_seconds)
        require(seconds <= 999_999, f"train_seconds out of range: {train_seconds}")
        name = f"model_it{iteration:07d}_s{seconds:06d}.pt"
        final = self._dir / name
        tmp = self._dir / f".{name}.tmp"
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            write(tmp)
            require(tmp.is_file(), f"write() did not create {tmp}")
            _fsync_path(tmp, os.O_RDONLY)
            os.replace(tmp, final)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        _fsync_path(self._dir, _DIR_FLAGS)  # the rename itself
        return CheckpointRef(iteration, float(seconds), final)

    def list(self) -> list[CheckpointRef]:
        if not self._dir.is_dir():
            return []
        refs: list[CheckpointRef] = []
        for entry in self._dir.iterdir():
            match = _NAME.match(entry.name)
            if match is not None and entry.is_file():
                refs.append(CheckpointRef(int(match[1]), float(int(match[2])), entry))
        # Ties on iteration (a resume that re-labelled an index) resolve to the later training time.
        return sorted(refs, key=lambda r: (r.iteration, r.train_seconds))

    @staticmethod
    def integrity_error(ref: CheckpointRef) -> str | None:
        """Why ``ref``'s file is not an intact zip archive, or None when every member passes its CRC-32.

        torch 2.8.0's ``torch.load`` raises ``RuntimeError`` (no central directory), ``EOFError``,
        ``OSError`` or ``KeyError`` on a truncated file depending on where it was cut, and silently loads a
        file whose flipped bytes land in tensor data; the archive check catches all of these alike. An
        ``OSError`` reading the file (an I/O fault, not damage) propagates.
        """
        try:
            with zipfile.ZipFile(ref.path) as archive:
                bad_member = archive.testzip()
        except zipfile.BadZipFile as error:
            return f"BadZipFile: {error}"
        return None if bad_member is None else f"CRC mismatch in member {bad_member}"

    def quarantine(self, ref: CheckpointRef) -> Path:
        """Rename ``ref``'s file to ``<name>.corrupt`` (durably), so ``list()`` and ``latest()`` skip it."""
        require(ref.path.parent == self._dir, f"{ref.path} is not in {self._dir}")
        require(_NAME.match(ref.path.name) is not None, f"{ref.path.name} is not a checkpoint name")
        target = ref.path.with_name(ref.path.name + _CORRUPT_SUFFIX)
        os.replace(ref.path, target)
        _fsync_path(self._dir, _DIR_FLAGS)
        require(ref not in self.list(), f"{ref.path.name} is still listed after quarantine")
        return target

    def quarantined(self) -> list[Path]:
        """Files ``quarantine`` moved aside, sorted by name."""
        if not self._dir.is_dir():
            return []
        return sorted(
            entry
            for entry in self._dir.iterdir()
            if entry.name.endswith(_CORRUPT_SUFFIX) and _NAME.match(entry.name.removesuffix(_CORRUPT_SUFFIX))
        )

    def latest(self) -> CheckpointRef | None:
        refs = self.list()
        return refs[-1] if refs else None

    def at_or_before(self, train_minutes: float) -> CheckpointRef | None:
        limit = train_minutes * 60.0
        eligible = [r for r in self.list() if r.train_seconds <= limit]
        return eligible[-1] if eligible else None

    def crossed(self, mark_minutes: float, interval_minutes: float) -> CheckpointRef | None:
        """The checkpoint that crossed ``mark_minutes``: the earliest with ``mark <= t < mark + interval``.

        A checkpoint is saved at the end of the first iteration at or past its mark, and its stamp is
        rounded to whole seconds, so the 3:00 checkpoint is typically stamped 180-183 s; the window runs up
        to (not including) the next mark's. None when no checkpoint is in the window: the mark is absent.
        This is the one rule by which an evaluation mark, a curve point and a race-video frame pick a
        checkpoint.
        """
        require(interval_minutes > 0.0, f"interval_minutes must be positive, got {interval_minutes}")
        require(mark_minutes >= 0.0, f"mark_minutes must be non-negative, got {mark_minutes}")
        low = mark_minutes * 60.0 - _STAMP_TOLERANCE_S
        high = (mark_minutes + interval_minutes) * 60.0 - _STAMP_TOLERANCE_S
        in_window = [r for r in self.list() if low <= r.train_seconds < high]
        return min(in_window, key=lambda r: (r.train_seconds, r.iteration)) if in_window else None
