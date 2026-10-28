"""Smoke test: TrainingDiagnostics on the real G1 runner (mjlab + rsl_rl, CPU, 4 envs)."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest
from rsl_rl.utils.log_writer import LogWriter
from support import LONG_S, make_env_runner

from mprl.checkpoints import CheckpointStore
from mprl.clock import TrainingClock, cuda_sync_for
from mprl.diagnostics import for_runner

pytestmark = pytest.mark.smoke


class ScalarRecorder(LogWriter):
    """A real rsl_rl LogWriter that records every scalar with its step."""

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.scalars.append((tag, scalar_value, global_step))


def _module_tree(runner: Any) -> dict[str, list[str]]:
    """The real module and parameter names of actor and critic, as the diagnostics must name them."""
    return {
        "actor": [name for name, _ in runner.alg.get_policy().named_parameters()],
        "critic": [name for name, _ in runner.alg.critic.named_parameters()],
    }


@pytest.fixture(scope="module")
def g1_scalars(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[dict[tuple[str, int], float], dict[str, list[str]]]]:
    """Run 2 real G1 iterations with every_k=1, once per module; yield {(key, step): value} and the module tree."""
    root = tmp_path_factory.mktemp("diag")
    env, runner = make_env_runner(root / "logs", logger={"class_name": ScalarRecorder})
    try:
        tree = _module_tree(runner)
        diag, on_before_update, on_iteration = for_runner(runner, every_k=1)
        diag.attach()
        clock = TrainingClock(sync=cuda_sync_for("cpu"))
        runner.learn_for(
            clock, LONG_S, LONG_S, CheckpointStore(root / "checkpoints"), fresh_start=True, max_iterations=2,
            on_before_update=on_before_update, on_iteration=on_iteration,
        )  # fmt: skip
        diag.detach()
        yield {(k, step): v for k, v, step in runner.logger.writer.scalars}, tree
    finally:
        env.close()


@pytest.mark.parametrize("module", ["actor", "critic"])
def test_gradient_weight_and_activation_metrics(
    g1_scalars: tuple[dict[tuple[str, int], float], dict[str, list[str]]], module: str
) -> None:
    by_key, tree = g1_scalars
    steps = (0, 1)

    assert all(by_key[(f"Diag/grad_norm_preclip/{module}", s)] > 0.0 for s in steps)
    assert all(0.0 < by_key[(f"Diag/clip_coef/{module}", s)] <= 1.0 for s in steps)
    metrics = ("grad_norm", "weight_norm", "update_ratio")
    assert all(by_key[(f"Diag/{m}/{module}/{n}", s)] >= 0.0 for m in metrics for n in tree[module] for s in steps)
    assert all(by_key[(f"Diag/weight_norm/{module}/mlp.0.weight", s)] > 0.0 for s in steps)
    layers = range(3)  # hidden_dims (512, 256, 128)
    assert all((f"Diag/act_mean/{module}/layer{i}", s) in by_key for i in layers for s in steps)
    assert all(0.0 <= by_key[(f"Diag/act_frac_saturated/{module}/layer{i}", s)] <= 1.0 for i in layers for s in steps)
    assert (f"Diag/act_mean/{module}/layer3", 0) not in by_key  # the final Linear has no activation


def test_policy_scalars_finite_and_aligned_with_rsl_rl_steps(
    g1_scalars: tuple[dict[tuple[str, int], float], dict[str, list[str]]],
) -> None:
    by_key, _ = g1_scalars

    assert all(by_key[("Diag/action_std", s)] > 0.0 for s in (0, 1))
    assert all(("Diag/obs_norm_mean_abs", s) in by_key and ("Diag/obs_norm_std_mean", s) in by_key for s in (0, 1))
    assert all(math.isfinite(v) for v in by_key.values())
    assert ("Loss/value", 1) in by_key  # same step axis as rsl_rl's own metrics
