"""A resumable Weights & Biases writer for rsl_rl.

rsl_rl's built-in ``WandbLogWriter`` calls ``wandb.init`` without ``id``/``resume``, so every pod restart
would fork a new W&B run. ``ResumableWandbWriter`` pins the run id and always opens it with
``id=<run_id>, resume="allow"``, fresh or resumed (a ``finished`` run reopens too). It does not use
``resume_from`` (W&B Rewind): that is a private preview, and on 2026-10-29 the server answered it with a 400
that killed the long run. Without rewind, W&B keeps the history logged before a crash; a resumed run logs
from the checkpoint iteration on, and W&B ignores (with a warning) any step not above the run's last step, so
after a crash the iterations between the last checkpoint and the crash keep their first values.

Three more behaviors:

* **No ``/time`` series.** rsl-rl-lib 5.4.2's ``Logger.log`` (``rsl_rl/utils/logger.py:227-233``) writes
  ``Train/mean_reward/time`` and ``Train/mean_episode_length/time`` at ``step=int(tot_time)`` (wall seconds)
  for every writer whose type is not its own ``WandbLogWriter``. W&B drops any step lower than the current
  one, so once wall seconds pass the iteration count every later per-iteration metric would be discarded.
  ``add_scalar`` drops tags ending in ``/time`` (a TensorBoard-only wall-time x-axis); the training clock
  replaces it as ``Perf/train_clock_minutes`` (``CLOCK_METRIC``, written by ``mprl.train`` each iteration).
* **Training clock as x-axis.** ``CLOCK_METRIC`` is ``define_metric``-ed and is the default x-axis of
  ``Train/*`` (mean reward, mean episode length), the equal-training-time comparison. Every other chart keeps
  the iteration step; any chart can switch its x-axis to ``Perf/train_clock_minutes`` in the UI.
* **Offline fallback.** An online ``wandb.init`` that fails with an operating error (``CommError``, which
  includes ``AuthenticationError``, the base ``wandb.errors.Error`` of an UNKNOWN wandb-core error, or
  ``TimeoutError``) is retried once in offline mode with the same id, after
  ``wandb.teardown()`` (the failed init leaves the id claimed in wandb's service, and a retry without the
  teardown raises ``ServerResponseError: run ID ... is in use``; seen with wandb 0.31.0 on 2026-10-29), so
  one W&B outage does not abort an unattended sweep. The reason is kept in ``offline_fallback`` (``train``
  records it in ``state.json``) and logged as a warning; ``wandb sync`` uploads the offline run later. A
  ``UsageError`` (a bad argument: a bug) is not caught.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from collections.abc import Sequence
from typing import Any, Final

import wandb
from rsl_rl.utils.log_writer import LogWriter
from wandb.errors import Error as WandbError
from wandb.errors import UsageError

from mprl.negative_space import require

log = logging.getLogger(__name__)

WRITER_CLASS_NAME = "mprl.wandb_writer:ResumableWandbWriter"  # "module:Attr" form accepted by resolve_callable
CLOCK_METRIC: Final = "Perf/train_clock_minutes"
_CLOCK_AXIS_GLOBS: Final = ("Train/*",)
_WALL_TIME_SUFFIX: Final = "/time"
# wandb 0.31 maps wandb-core error codes to Error (UNKNOWN), CommError (incl. AuthenticationError) and UsageError
# (wandb/errors/util.py); every one but UsageError is the outside world, as is the init timeout.
_INIT_OPERATING_ERRORS: Final = (WandbError, TimeoutError)


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
        tags: Sequence[str] = (),
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        require(run_id != "", "run_id must be non-empty")
        require(resume_from_iteration is None or resume_from_iteration >= 0, "resume_from_iteration must be >= 0")
        common = {
            "project": project,
            "entity": entity,
            "group": group,
            "job_type": job_type,
            "tags": list(tags),
            "config": {"log_dir": log_dir, **(extra_config or {})},
        }
        self.offline_fallback: str | None = None
        if resume_from_iteration is not None:
            log.info("resuming W&B run %s from iteration %d", run_id, resume_from_iteration)
        try:
            run = wandb.init(id=run_id, resume="allow", mode=mode, **common)  # pyright: ignore[reportArgumentType]
        except UsageError:
            raise  # a bad argument: a bug, not the outside world
        except _INIT_OPERATING_ERRORS as error:
            if mode != "online":
                raise
            self.offline_fallback = f"{type(error).__name__}: {error}"
            log.warning("wandb.init failed for %s (%s); falling back to offline mode for this run", run_id, error)
            wandb.teardown()  # frees the run id the failed online init still holds in wandb's service
            run = wandb.init(id=run_id, resume="allow", mode="offline", **common)  # pyright: ignore[reportArgumentType]
        self._run = run
        run.define_metric(CLOCK_METRIC)
        for pattern in _CLOCK_AXIS_GLOBS:
            run.define_metric(pattern, step_metric=CLOCK_METRIC)
        self._logged_videos: set[str] = set()

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        if tag.endswith(_WALL_TIME_SUFFIX):
            return  # rsl_rl's wall-time series: a lower step than the iteration's would make W&B drop the rest
        self._run.log({tag: scalar_value}, step=global_step)

    def store_config(self, env_cfg: dict[str, Any] | object, train_cfg: dict[str, Any]) -> None:
        env = (
            dataclasses.asdict(env_cfg)
            if dataclasses.is_dataclass(env_cfg) and not isinstance(env_cfg, type)
            else env_cfg
        )
        self._run.config.update({"train_cfg": train_cfg, "env_cfg": env}, allow_val_change=True)

    def save_model(self, model_path: str, it: int) -> None:
        """No-op: checkpoint uploads are owned by the Hugging Face publisher, not W&B."""

    def save_video(self, video: pathlib.Path, it: int) -> None:
        if video.name not in self._logged_videos:
            self._run.log({"video": wandb.Video(str(video), format="mp4")}, step=it)
            self._logged_videos.add(video.name)

    def stop(self) -> None:
        self._run.finish()


def logger_cfg(
    project: str,
    run_id: str,
    group: str,
    resume_from_iteration: int | None,
    mode: str,
    *,
    tags: Sequence[str] = (),
) -> dict[str, Any]:
    """The dict for ``agent_cfg["logger"]``; rsl_rl pops ``class_name`` and passes the rest as kwargs.

    ``tags`` is stored as a list: the dict is dumped to ``params/agent.yaml``.
    """
    return {
        "class_name": WRITER_CLASS_NAME,
        "project": project,
        "run_id": run_id,
        "group": group,
        "resume_from_iteration": resume_from_iteration,
        "mode": mode,
        "tags": list(tags),
    }
