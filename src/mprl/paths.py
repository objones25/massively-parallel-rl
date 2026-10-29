"""On-volume layout per run and per sweep."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from mprl.config import RunSpec

VOLUME_ROOT_ENV = "MPRL_VOLUME_ROOT"
DEFAULT_VOLUME_ROOT = "/workspace/mprl"


def volume_root(environ: Mapping[str, str]) -> Path:
    return Path(environ.get(VOLUME_ROOT_ENV, DEFAULT_VOLUME_ROOT))


def sweep_results_dir(root: Path, sweep: str) -> Path:
    return root / "sweeps" / sweep / "results"


@dataclass(frozen=True)
class RunPaths:
    root: Path
    spec: RunSpec

    @property
    def run_dir(self) -> Path:
        return self.root / "sweeps" / self.spec.sweep / self.spec.run_id

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def state_file(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def eval_dir(self) -> Path:
        return self.run_dir / "eval"

    @property
    def videos_dir(self) -> Path:
        return self.run_dir / "videos"

    @property
    def tensorboard_dir(self) -> Path:
        return self.run_dir / "tensorboard"
