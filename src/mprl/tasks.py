"""The registered mjlab task configs, with mjlab imported on first use.

mjlab is imported lazily throughout ``mprl`` (here, ``train.built_runner``, the evaluator and preflight's
env check) so that importing ``mprl`` for its pure parts (the CLI's control flow, budgets, run state,
``git_sha``) costs no mjlab import, about 2.7 s, and needs none of the EGL/GL system libraries that
``import mjlab.tasks`` requires on Linux. The fast test tier relies on this.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

from mprl.negative_space import expect, require

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.rl import RslRlOnPolicyRunnerCfg


def _registry() -> ModuleType:
    """``mjlab.tasks.registry`` with every task registered (``import mjlab.tasks`` registers them)."""
    import mjlab.tasks  # noqa: F401  # pyright: ignore[reportUnusedImport]  (registers the tasks)
    from mjlab.tasks import registry

    return registry


def load_task_env_cfg(task_id: str, *, play: bool = False) -> ManagerBasedRlEnvCfg:
    """The registered task's env config (``mjlab.tasks.registry.load_env_cfg``), a fresh copy per call."""
    require(bool(task_id), "task_id must be non-empty")
    return _registry().load_env_cfg(task_id, play=play)


def load_task_rl_cfg(task_id: str) -> RslRlOnPolicyRunnerCfg:
    """The registered task's PPO runner config; a task without one is a programmer error."""
    from mjlab.rl import RslRlOnPolicyRunnerCfg

    require(bool(task_id), "task_id must be non-empty")
    loaded = _registry().load_rl_cfg(task_id)
    return expect(loaded if isinstance(loaded, RslRlOnPolicyRunnerCfg) else None, f"{task_id} is not a PPO task")
