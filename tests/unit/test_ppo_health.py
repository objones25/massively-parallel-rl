"""Unit tests for the PPO health metrics (approx KL, clip fraction, explained variance) in mprl.diagnostics.

The pure estimators run on hand-written log-probabilities; ``PpoHealth`` runs on a real rsl_rl ``MLPModel``
(1 action, Gaussian, scalar std) over a real ``RolloutStorage`` filled by hand, so every expected value
below is computed by hand from the Gaussian density.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mprl.diagnostics import PpoHealth, approx_kl, clip_fraction, explained_variance, for_runner
from mprl.negative_space import CheckFailed

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
# ratios r = (1.5, 0.5, 1.1, 1.0): (r - 1) - log r = (0.0945349, 0.1931472, 0.0046898, 0.0)
HAND_RATIOS = (1.5, 0.5, 1.1, 1.0)
HAND_KL = (0.5 - math.log(1.5) + (-0.5 + math.log(2.0)) + (0.1 - math.log(1.1))) / 4.0  # 0.0730930


def _log_probs(ratios: tuple[float, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(len(ratios)), torch.log(torch.tensor(ratios))


def test_approx_kl_is_the_low_variance_estimator_on_hand_values() -> None:
    old, new = _log_probs(HAND_RATIOS)

    assert float(approx_kl(old, new)) == pytest.approx(HAND_KL, rel=1e-6)


def test_approx_kl_is_zero_for_an_unchanged_policy() -> None:
    old = torch.tensor([-1.0, -2.5, 0.3])

    assert float(approx_kl(old, old.clone())) == pytest.approx(0.0, abs=1e-12)


def test_clip_fraction_counts_ratios_outside_the_clip_range() -> None:
    old, new = _log_probs(HAND_RATIOS)

    assert float(clip_fraction(old, new, 0.2)) == pytest.approx(0.5)  # 1.5 and 0.5; 1.1 and 1.0 are inside


def test_explained_variance_on_hand_values() -> None:
    returns = torch.tensor([1.0, 2.0, 3.0, 4.0])  # population variance 1.25
    values = torch.tensor([1.0, 2.0, 3.0, 3.0])  # residual (0, 0, 0, 1): population variance 0.1875

    assert explained_variance(values, returns) == pytest.approx(1.0 - 0.1875 / 1.25)


def test_explained_variance_is_undefined_for_constant_returns() -> None:
    assert explained_variance(torch.tensor([0.0, 1.0]), torch.tensor([2.0, 2.0])) is None


def test_estimators_reject_mismatched_shapes() -> None:
    with pytest.raises(CheckFailed, match=r"^new_log_prob shape \(3,\) != old_log_prob shape \(4,\)$"):
        approx_kl(torch.zeros(4), torch.zeros(3))


# --- PpoHealth on a real MLPModel and RolloutStorage ----------------------------------------------------

OBS_DIM = 2


def _obs(num_envs: int, seed: int = 0) -> TensorDict:
    generator = torch.Generator().manual_seed(seed)
    return TensorDict({"policy": torch.randn(num_envs, OBS_DIM, generator=generator)}, batch_size=[num_envs])


def _actor() -> MLPModel:
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    distribution = {"class_name": "rsl_rl.modules.GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
    return MLPModel(_obs(1), obs_groups, "actor", 1, hidden_dims=(4,), distribution_cfg=distribution)


def _mean(actor: MLPModel, obs: TensorDict) -> torch.Tensor:
    with torch.no_grad():
        return actor(obs)  # deterministic output: the Gaussian mean


def _filled_storage(
    actor: MLPModel,
    num_envs: int,
    steps: int,
    *,
    old_log_prob: torch.Tensor,
    values: torch.Tensor | None = None,
    returns: torch.Tensor | None = None,
) -> RolloutStorage:
    """A full rollout: obs per step, action = the actor's mean, the given old log-probs, values and returns."""
    storage = RolloutStorage("rl", num_envs, steps, _obs(num_envs), [1])
    values = torch.zeros(steps, num_envs) if values is None else values
    for t in range(steps):
        obs = _obs(num_envs, seed=t)
        transition = RolloutStorage.Transition()
        transition.observations = obs
        transition.actions = _mean(actor, obs)
        transition.rewards = torch.zeros(num_envs)
        transition.dones = torch.zeros(num_envs)
        transition.values = values[t].unsqueeze(-1)
        transition.actions_log_prob = old_log_prob[t]
        transition.distribution_params = (transition.actions, torch.ones(num_envs, 1))
        storage.add_transition(transition)
    if returns is not None:
        storage.returns.copy_(returns.unsqueeze(-1))
    return storage


def _log_prob_at_mean(std: float) -> float:
    return -math.log(std) - LOG_SQRT_2PI  # log N(mu | mu, std^2)


def _set_std(actor: MLPModel, std: float) -> None:
    with torch.no_grad():
        actor.distribution.std_param.fill_(std)  # type: ignore[union-attr]


def _health(actor: MLPModel, storage: RolloutStorage, *, every_k: int = 1, max_samples: int = 1024) -> PpoHealth:
    return PpoHealth(actor, storage, clip_param=0.2, every_k=every_k, max_samples=max_samples)


def test_unchanged_policy_has_zero_kl_and_clip_fraction() -> None:
    actor = _actor()
    storage = _filled_storage(actor, 3, 2, old_log_prob=torch.full((2, 3), _log_prob_at_mean(1.0)))
    health = _health(actor, storage)
    health.before_update(0)

    metrics = health.after_update(0)

    assert metrics["Diag/approx_kl"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["Diag/clip_fraction"] == pytest.approx(0.0, abs=1e-12)


def test_doubling_the_std_gives_the_hand_computed_kl_and_full_clip_fraction() -> None:
    """Actions at the mean: log r = log N(mu|mu,4) - log N(mu|mu,1) = -log 2, so r = 0.5 everywhere."""
    actor = _actor()
    storage = _filled_storage(actor, 3, 2, old_log_prob=torch.full((2, 3), _log_prob_at_mean(1.0)))
    health = _health(actor, storage)
    health.before_update(0)
    _set_std(actor, 2.0)  # the PPO update

    metrics = health.after_update(0)

    assert metrics["Diag/approx_kl"] == pytest.approx(-0.5 + math.log(2.0), rel=1e-5)
    assert metrics["Diag/clip_fraction"] == pytest.approx(1.0)


def test_explained_variance_uses_the_pre_update_values_and_returns() -> None:
    actor = _actor()
    storage = _filled_storage(
        actor,
        2,
        2,
        old_log_prob=torch.full((2, 2), _log_prob_at_mean(1.0)),
        values=torch.tensor([[1.0, 2.0], [3.0, 3.0]]),
        returns=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    health = _health(actor, storage)
    health.before_update(0)
    storage.values.zero_()  # the next rollout overwrites the storage; the snapshot must not see it
    storage.returns.zero_()

    metrics = health.after_update(0)

    assert metrics["Diag/explained_variance"] == pytest.approx(0.85)


def test_constant_returns_omit_explained_variance() -> None:
    actor = _actor()
    storage = _filled_storage(actor, 2, 2, old_log_prob=torch.full((2, 2), _log_prob_at_mean(1.0)))
    health = _health(actor, storage)
    health.before_update(0)

    metrics = health.after_update(0)

    assert set(metrics) == {"Diag/approx_kl", "Diag/clip_fraction"}


def test_snapshot_is_a_strided_subsample_bounded_by_max_samples() -> None:
    """10 transitions, max 4: stride 3 keeps flat rows 0, 3, 6, 9. Only those rows hold the true old log-prob."""
    actor = _actor()
    flat = torch.full((10,), _log_prob_at_mean(1.0) - 5.0)  # ratio e^5 on every row outside the subsample
    flat[[0, 3, 6, 9]] = _log_prob_at_mean(1.0)
    storage = _filled_storage(actor, 5, 2, old_log_prob=flat.view(2, 5))
    health = _health(actor, storage, max_samples=4)
    health.before_update(0)

    metrics = health.after_update(0)

    assert metrics["Diag/approx_kl"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["Diag/clip_fraction"] == pytest.approx(0.0, abs=1e-12)


def test_non_k_iteration_takes_no_snapshot_and_returns_nothing() -> None:
    actor = _actor()
    storage = _filled_storage(actor, 3, 2, old_log_prob=torch.full((2, 3), _log_prob_at_mean(1.0)))
    health = _health(actor, storage, every_k=10)
    health.before_update(3)

    assert health.after_update(3) == {}
    assert health.snapshot_bytes == 0


def test_metrics_do_not_consume_the_global_rng() -> None:
    """Sampling here would shift every later minibatch permutation and action sample of the training run."""
    actor = _actor()
    storage = _filled_storage(actor, 3, 2, old_log_prob=torch.full((2, 3), _log_prob_at_mean(1.0)))
    health = _health(actor, storage)
    before = torch.get_rng_state()

    health.before_update(0)
    health.after_update(0)

    assert torch.equal(torch.get_rng_state(), before)


def test_snapshot_holds_only_the_actor_observations() -> None:
    """Per kept transition: the actor's 2 obs floats + 1 action + 1 old log-prob; plus values and returns (6 x 2)."""
    actor = _actor()
    storage = _filled_storage(actor, 3, 2, old_log_prob=torch.full((2, 3), _log_prob_at_mean(1.0)))
    health = _health(actor, storage)

    health.before_update(0)

    assert health.snapshot_bytes == 4 * (6 * (OBS_DIM + 1 + 1) + 2 * 6)


# --- a diverging run: metrics must never stop training ---------------------------------------------------


def test_kl_beyond_the_float32_range_is_still_finite() -> None:
    """log r = 100: expm1 overflows float32 above ~88.7; the estimator works in float64."""
    old, new = torch.zeros(2), torch.full((2,), 100.0)

    assert float(approx_kl(old, new)) == pytest.approx(math.expm1(100.0) - 100.0, rel=1e-9)


def test_overflowing_kl_is_omitted_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """log r = 1000 overflows even float64: the key is dropped, the clip fraction (still defined) is kept."""
    actor = _actor()
    storage = _filled_storage(actor, 3, 2, old_log_prob=torch.full((2, 3), _log_prob_at_mean(1.0) - 1000.0))
    health = _health(actor, storage)
    health.before_update(0)

    metrics = health.after_update(0)

    assert metrics == {"Diag/clip_fraction": pytest.approx(1.0)}
    assert "Diag/approx_kl=inf" in caplog.text


def test_non_finite_explained_variance_is_omitted_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    actor = _actor()
    storage = _filled_storage(
        actor,
        2,
        2,
        old_log_prob=torch.full((2, 2), _log_prob_at_mean(1.0)),
        values=torch.tensor([[math.inf, 2.0], [3.0, 3.0]]),
        returns=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    health = _health(actor, storage)
    health.before_update(0)

    metrics = health.after_update(0)

    assert set(metrics) == {"Diag/approx_kl", "Diag/clip_fraction"}
    assert "Diag/explained_variance=" in caplog.text


# --- for_runner wiring ---------------------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.scalars: dict[str, float] = {}

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars[f"{tag}@{step}"] = value


def test_for_runner_uses_the_algorithms_clip_param() -> None:
    """std 1 -> 1.15 with actions at the mean: r = 1/1.15, |r - 1| = 0.13, clipped at 0.1 but not at 0.2."""
    actor = _actor()
    critic = MLPModel(_obs(1), {"actor": ["policy"], "critic": ["policy"]}, "critic", 1, hidden_dims=(4,))
    storage = _filled_storage(actor, 3, 2, old_log_prob=torch.full((2, 3), _log_prob_at_mean(1.0)))
    writer = _Recorder()
    alg = SimpleNamespace(
        get_policy=lambda: actor, critic=critic, max_grad_norm=1.0, storage=storage, clip_param=0.1
    )
    runner = SimpleNamespace(
        alg=alg, logger=SimpleNamespace(writer=writer), env=SimpleNamespace(get_observations=lambda: _obs(3)),
        device="cpu",
    )  # fmt: skip
    diag, on_before_update, on_iteration = for_runner(runner, every_k=1)  # type: ignore[arg-type]  duck-typed
    diag.attach()
    on_before_update(0)
    obs = _obs(3)
    loss = actor.mlp(actor.get_latent(obs)).sum() + critic(obs).sum() + actor.distribution.std_param.sum()  # type: ignore[union-attr]
    loss.backward()  # the update's backward, which the gradient hooks need
    _set_std(actor, 1.15)  # the update's step
    with torch.no_grad():  # update() leaves the distribution over its last minibatch (ppo.py:221)
        actor.distribution.update(actor.mlp(actor.get_latent(obs)))  # type: ignore[union-attr]

    on_iteration(0)

    assert writer.scalars["Diag/clip_fraction@0"] == pytest.approx(1.0)
