"""Per-run ``state.json``: status, W&B run id, config hash, training clock and upload outcomes.

Written atomically (temp file in the same directory, fsync, ``os.replace``), so a process killed
mid-write leaves the previous complete state in place and the temp file is never read.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final, Literal, get_args

from mprl.negative_space import check_finite, require

Status = Literal["running", "complete", "failed"]
_STATUSES: Final = frozenset(get_args(Status))


@dataclass
class RunState:
    run_id: str
    config_hash: str
    wandb_run_id: str
    status: Status
    train_seconds: float
    budget_minutes: float
    git_sha: str
    uploads: dict[str, str] = field(default_factory=dict[str, str])  # artifact -> "ok" or the error text

    def __post_init__(self) -> None:
        require(self.status in _STATUSES, f"status must be one of {sorted(_STATUSES)}, got {self.status!r}")
        require(self.run_id != "", "run_id must be non-empty")
        require(self.wandb_run_id != "", "wandb_run_id must be non-empty")
        check_finite(self.train_seconds, "train_seconds")
        require(self.train_seconds >= 0.0, f"train_seconds must be non-negative, got {self.train_seconds}")
        check_finite(self.budget_minutes, "budget_minutes")
        require(self.budget_minutes > 0.0, f"budget_minutes must be positive, got {self.budget_minutes}")


def load_state(path: Path) -> RunState | None:
    """The stored state, or None if the run never wrote one."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    require(isinstance(data, dict), f"{path} does not hold a JSON object")
    return RunState(**data)


def save_state(path: Path, state: RunState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("w") as f:
            json.dump(asdict(state), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    require(path.is_file(), f"state file {path} missing after replace")
