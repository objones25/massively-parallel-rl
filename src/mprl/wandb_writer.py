"""A resumable Weights & Biases writer for rsl_rl.

rsl_rl's built-in ``WandbLogWriter`` calls ``wandb.init`` without ``id``/``resume``, so every pod restart
would fork a new W&B run. ``ResumableWandbWriter`` pins the run id; on resume it passes
``resume_from="<id>?_step=<iteration>"`` so history after the checkpoint is truncated and the steps of the
resumed run (which starts at ``checkpoint_iteration + 1``) stay monotonic.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import wandb
from rsl_rl.utils.log_writer import LogWriter

from mprl.negative_space import require

WRITER_CLASS_NAME = "mprl.wandb_writer:ResumableWandbWriter"  # "module:Attr" form accepted by resolve_callable


class ResumableWandbWriter(LogWriter):
    """rsl_rl LogWriter that resumes one W&B run across process restarts."""

    def __init__(
        self,
        log_dir: str,
        project: str,
        run_id: str,
        group: str,
        job_type: str = "train",
        resume_from_iteration: int | None = None,
        mode: str = "online",
        entity: str | None = None,
        tags: tuple[str, ...] = (),
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        require(run_id != "", "run_id must be non-empty")
        require(resume_from_iteration is None or resume_from_iteration >= 0, "resume_from_iteration must be >= 0")
        common = {
            "project": project,
            "entity": entity,
            "group": group,
            "job_type": job_type,
            "mode": mode,
            "tags": list(tags),
            "config": {"log_dir": log_dir, **(extra_config or {})},
        }
        if resume_from_iteration is None:
            wandb.init(id=run_id, resume="allow", **common)  # pyright: ignore[reportArgumentType]
        else:
            wandb.init(resume_from=f"{run_id}?_step={resume_from_iteration}", **common)  # pyright: ignore[reportArgumentType]
        self._logged_videos: set[str] = set()

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        wandb.log({tag: scalar_value}, step=global_step)

    def store_config(self, env_cfg: dict[str, Any] | object, train_cfg: dict[str, Any]) -> None:
        env = (
            dataclasses.asdict(env_cfg)
            if dataclasses.is_dataclass(env_cfg) and not isinstance(env_cfg, type)
            else env_cfg
        )
        wandb.config.update({"train_cfg": train_cfg, "env_cfg": env}, allow_val_change=True)

    def save_model(self, model_path: str, it: int) -> None:
        """No-op: checkpoint uploads are owned by the Hugging Face publisher, not W&B."""

    def save_video(self, video: pathlib.Path, it: int) -> None:
        if video.name not in self._logged_videos:
            wandb.log({"video": wandb.Video(str(video), format="mp4")}, step=it)
            self._logged_videos.add(video.name)

    def stop(self) -> None:
        wandb.finish()


def logger_cfg(project: str, run_id: str, group: str, resume_from_iteration: int | None, mode: str) -> dict[str, Any]:
    """The dict for ``agent_cfg["logger"]``; rsl_rl pops ``class_name`` and passes the rest as kwargs."""
    return {
        "class_name": WRITER_CLASS_NAME,
        "project": project,
        "run_id": run_id,
        "group": group,
        "resume_from_iteration": resume_from_iteration,
        "mode": mode,
    }
