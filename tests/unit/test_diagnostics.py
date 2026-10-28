"""Unit tests for TrainingDiagnostics on a tiny nn.Sequential(Linear, ELU, Linear)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mprl.diagnostics import TrainingDiagnostics
from mprl.negative_space import CheckFailed, check_finite


def _tiny(seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(3, 4), nn.ELU(), nn.Linear(4, 1))


def _backward(actor: nn.Module, critic: nn.Module, x: torch.Tensor) -> None:
    loss = actor(x).pow(2).sum() + critic(x).pow(2).sum()
    loss.backward()


def _grad_norm(model: nn.Module) -> float:
    grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(grads)))


@pytest.fixture
def models() -> tuple[nn.Sequential, nn.Sequential]:
    return _tiny(0), _tiny(1)


def test_grad_norm_matches_manual(models: tuple[nn.Sequential, nn.Sequential]) -> None:
    actor, critic = models
    diag = TrainingDiagnostics(actor, critic, every_k=1, max_grad_norm=1.0)
    diag.attach()
    diag.before_update(0)

    _backward(actor, critic, torch.randn(5, 3))
    metrics = diag.after_update(0)

    assert metrics["Diag/grad_norm_preclip/actor"] == pytest.approx(_grad_norm(actor), rel=1e-6)
    assert metrics["Diag/grad_norm_preclip/critic"] == pytest.approx(_grad_norm(critic), rel=1e-6)
    assert metrics["Diag/grad_norm/actor/0.weight"] == pytest.approx(
        float(torch.linalg.vector_norm(actor[0].weight.grad)), rel=1e-6
    )
    expected_coef = min(1.0, 1.0 / _grad_norm(actor))
    assert metrics["Diag/clip_coef/actor"] == pytest.approx(expected_coef, rel=1e-6)


def test_grad_norm_is_mean_over_backwards(models: tuple[nn.Sequential, nn.Sequential]) -> None:
    actor, critic = models
    diag = TrainingDiagnostics(actor, critic, every_k=1)
    diag.attach()
    diag.before_update(0)
    norms: list[float] = []
    for _ in range(3):
        actor.zero_grad()
        critic.zero_grad()
        _backward(actor, critic, torch.randn(5, 3))
        norms.append(_grad_norm(actor))

    metrics = diag.after_update(0)

    assert metrics["Diag/grad_norm_preclip/actor"] == pytest.approx(sum(norms) / 3, rel=1e-6)


def test_update_ratio_zero_when_weights_unchanged(models: tuple[nn.Sequential, nn.Sequential]) -> None:
    actor, critic = models
    diag = TrainingDiagnostics(actor, critic, every_k=1)
    diag.attach()
    diag.before_update(0)
    _backward(actor, critic, torch.randn(5, 3))  # gradients computed, optimizer never steps

    metrics = diag.after_update(0)

    ratios = {k: v for k, v in metrics.items() if k.startswith("Diag/update_ratio/")}
    assert len(ratios) == 8  # (weight, bias) x 2 layers x (actor, critic)
    assert all(v == pytest.approx(0.0, abs=0.0) for v in ratios.values())


def test_update_ratio_matches_manual(models: tuple[nn.Sequential, nn.Sequential]) -> None:
    actor, critic = models
    diag = TrainingDiagnostics(actor, critic, every_k=1)
    diag.attach()
    diag.before_update(0)
    _backward(actor, critic, torch.randn(5, 3))
    before = actor[0].weight.detach().clone()
    with torch.no_grad():
        actor[0].weight.add_(0.1)

    metrics = diag.after_update(0)

    expected = float(torch.linalg.vector_norm(actor[0].weight.detach() - before) / torch.linalg.vector_norm(before))
    assert metrics["Diag/update_ratio/actor/0.weight"] == pytest.approx(expected, rel=1e-5)
    assert metrics["Diag/weight_norm/actor/0.weight"] == pytest.approx(
        float(torch.linalg.vector_norm(actor[0].weight.detach())), rel=1e-6
    )


def test_saturated_fraction_counts_elu_floor() -> None:
    actor, critic = _tiny(0), _tiny(1)
    with torch.no_grad():
        actor[0].weight.fill_(-10.0)  # positive inputs -> pre-activation << 0 -> ELU output -1
        actor[0].bias.zero_()
    diag = TrainingDiagnostics(actor, critic, every_k=1)

    metrics = diag.activation_stats(torch.ones(6, 3))

    assert metrics["Diag/act_frac_saturated/actor/layer0"] == pytest.approx(1.0)
    assert metrics["Diag/act_mean/actor/layer0"] == pytest.approx(-1.0, abs=1e-3)
    assert metrics["Diag/act_std/actor/layer0"] == pytest.approx(0.0, abs=1e-3)


def test_non_k_iteration_returns_empty(models: tuple[nn.Sequential, nn.Sequential]) -> None:
    actor, critic = models
    diag = TrainingDiagnostics(actor, critic, every_k=4)
    diag.attach()
    diag.before_update(1)
    _backward(actor, critic, torch.randn(5, 3))

    assert diag.after_update(1) == {}


def test_hooks_are_inert_on_non_k_iteration(models: tuple[nn.Sequential, nn.Sequential]) -> None:
    actor, critic = models
    diag = TrainingDiagnostics(actor, critic, every_k=4)
    diag.attach()
    diag.before_update(1)  # not a K-iteration: not armed
    _backward(actor, critic, torch.randn(5, 3))
    diag.before_update(4)  # K-iteration: armed, buffers must start empty
    _backward(actor, critic, torch.randn(5, 3))

    assert diag.after_update(4)["Diag/grad_norm_preclip/actor"] > 0.0


def test_detach_removes_hooks(models: tuple[nn.Sequential, nn.Sequential]) -> None:
    actor, critic = models
    diag = TrainingDiagnostics(actor, critic, every_k=1)
    diag.attach()
    diag.detach()
    diag.before_update(0)
    _backward(actor, critic, torch.randn(5, 3))

    with pytest.raises(CheckFailed, match="no backward"):
        diag.after_update(0)
    assert all(not p._post_accumulate_grad_hooks for p in [*actor.parameters(), *critic.parameters()])


def test_metrics_are_finite(models: tuple[nn.Sequential, nn.Sequential]) -> None:
    actor, critic = models
    diag = TrainingDiagnostics(actor, critic, every_k=1)
    diag.attach()
    diag.before_update(0)
    _backward(actor, critic, torch.randn(5, 3))
    metrics = diag.after_update(0) | diag.activation_stats(torch.randn(5, 3))

    assert len(metrics) > 20
    for key, value in metrics.items():
        check_finite(value, key)
