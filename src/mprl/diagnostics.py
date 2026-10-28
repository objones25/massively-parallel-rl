"""Per-layer training diagnostics: gradient, weight, update-ratio and activation metrics.

Real module tree (rsl-rl-lib 5.4.2, G1 flat): ``runner.alg.get_policy()`` is the actor
``MLPModel`` and ``runner.alg.critic`` the critic ``MLPModel`` (``get_policy`` is the public
accessor; ``_raw_actor`` is private). Each owns ``obs_normalizer`` (``EmpiricalNormalization``,
buffers only), ``mlp`` (``rsl_rl.modules.MLP``, an ``nn.Sequential`` named ``"0".."6"``:
``Linear, ELU, Linear, ELU, Linear, ELU, Linear``; one ELU *instance* is shared by every hidden
layer) and, for the actor only, ``distribution`` (a ``GaussianDistribution`` whose ``std_param``
is a parameter). Parameters are therefore named ``mlp.0.weight`` ... ``distribution.std_param``
and keyed ``Diag/<metric>/{actor,critic}/<param_name>``. Activation layer ``i`` is the ``i``-th
ELU call in a forward pass.

What runs where (see ``timed_runner`` for the clock):

* ON the clock, only on K-iterations: one post-accumulate-grad hook call per parameter per
  backward during ``PPO.update()``. The body is a squared-norm write into a preallocated tensor,
  with no ``.item()``/``.cpu()``. Non-K iterations pay only a Python bool check per hook call.
  Every run pays this identically per iteration, so the equal-time comparison stays fair.
* OFF the clock: weight snapshot (``on_before_update``), metric assembly, activation forward
  pass and writing (``on_iteration``).

PPO health (``PpoHealth``, K-iterations only, entirely OFF the clock). rsl_rl 5.4.2's ``PPO.update``
returns only the value, surrogate and entropy losses (``rsl_rl/algorithms/ppo.py:331``), so three more
are computed here, keyed ``Diag/approx_kl``, ``Diag/clip_fraction`` and ``Diag/explained_variance``:

* ``on_before_update`` (rollout and ``compute_returns`` done, update not yet run) snapshots from
  ``alg.storage``: the actor's observation groups, ``actions`` and ``actions_log_prob`` (the rollout
  policy's), subsampled to at most ``max_samples`` transitions by a fixed stride over the flattened
  ``(T, N)`` batch (no RNG), plus the full ``values`` and ``returns``. ``update()`` only reads the
  storage and ``clear()`` only rewinds its cursor (``rollout_storage.py:207``), so the snapshot is a
  guard against a later rsl_rl, not a fix for 5.4.2. G1 at 16384 envs x 24 steps: 16384 kept rows x
  (99 actor obs + 29 actions + 1 log-prob) x 4 B = 8.5 MB, plus 2 x 393216 x 4 B = 3.1 MB, on device.
* ``on_iteration`` (after the update) recomputes the log-probs of the snapshot's actions under the
  updated actor without gradients, through ``get_latent`` -> ``mlp`` -> ``distribution.update``:
  ``forward(stochastic_output=True)`` would also *sample*, consuming the global RNG the next
  minibatch permutation draws from. This leaves the actor's distribution over the snapshot, as
  ``update()`` leaves it over its last minibatch; the next ``act()`` replaces it.

These measure the whole update (rollout policy -> final policy), not a per-minibatch mean as SB3 and
CleanRL log; rsl_rl's adaptive learning rate uses an analytic per-minibatch KL it does not return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import torch
from rsl_rl.models import MLPModel
from torch import nn

from mprl.negative_space import check_finite, expect, require

if TYPE_CHECKING:
    from collections.abc import Callable

    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.storage import RolloutStorage
    from tensordict import TensorDict
    from torch.utils.hooks import RemovableHandle

_MAX_BACKWARDS = 256  # PPO backwards per update = num_learning_epochs * num_mini_batches (G1: 5 * 4)
_ELU_SATURATED_BELOW = -0.99  # ELU(alpha=1) floors at -1
_RATIO_EPS = 1e-8  # zero-initialised biases would otherwise divide by zero
_MODULES = ("actor", "critic")
PPO_MAX_SAMPLES: Final = 16_384  # rows kept for the KL / clip-fraction recomputation (memory: module docstring)


class TrainingDiagnostics:
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        every_k: int,
        *,
        max_grad_norm: float = 1.0,
        max_backwards: int = _MAX_BACKWARDS,
    ) -> None:
        require(every_k >= 1, f"every_k must be >= 1, got {every_k}")
        require(max_backwards >= 1, f"max_backwards must be >= 1, got {max_backwards}")
        check_finite(max_grad_norm, "max_grad_norm")
        require(max_grad_norm > 0.0, f"max_grad_norm must be positive, got {max_grad_norm}")
        self._models = {"actor": actor, "critic": critic}
        self._every_k = every_k
        self._max_grad_norm = max_grad_norm
        self._params: list[tuple[str, str, nn.Parameter]] = [  # (module, name, parameter)
            (module, name, p) for module, model in self._models.items() for name, p in model.named_parameters()
        ]
        require(len(self._params) > 0, "actor and critic have no parameters")
        require(
            len({id(p) for _, _, p in self._params}) == len(self._params),
            "actor and critic share a parameter; per-module gradient norms would be ambiguous",
        )
        device = self._params[0][2].device
        self._sq = torch.zeros(len(self._params), max_backwards, device=device)  # sum(grad^2) per backward
        self._counts = [0] * len(self._params)  # Python ints: no device sync
        self._handles: list[RemovableHandle] = []
        self._armed = False
        self._snapshot: list[torch.Tensor] | None = None

    # ---- hooks (ON the clock when armed) ----

    def attach(self) -> None:
        require(not self._handles, "attach() called twice")
        for i, (_, _, p) in enumerate(self._params):
            self._handles.append(p.register_post_accumulate_grad_hook(self._make_hook(i)))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._armed = False

    def _make_hook(self, i: int) -> Callable[[torch.Tensor], None]:
        def hook(p: torch.Tensor) -> None:
            if not self._armed:
                return
            row = self._counts[i]
            require(row < self._sq.shape[1], f"more than {self._sq.shape[1]} backwards in one update")
            grad = expect(p.grad, "post-accumulate-grad hook fired without a gradient")
            self._sq[i, row] = (grad * grad).sum()
            self._counts[i] = row + 1

        return hook

    # ---- per-iteration protocol (OFF the clock) ----

    def before_update(self, it: int) -> None:
        """Arm the gradient hooks and snapshot weights, on K-iterations only."""
        self._armed = it % self._every_k == 0
        if not self._armed:
            self._snapshot = None
            return
        self._sq.zero_()
        self._counts = [0] * len(self._params)
        self._snapshot = [p.detach().clone() for _, _, p in self._params]

    def after_update(self, it: int) -> dict[str, float]:
        """Disarm and return this K-iteration's gradient / weight / update-ratio metrics ({} otherwise)."""
        if not self._armed:
            return {}
        self._armed = False
        snapshot = expect(self._snapshot, "after_update without before_update")
        n_back = self._counts[0]
        require(n_back > 0, f"no backward recorded at iteration {it}: hooks not attached or update did not run")
        require(all(c == n_back for c in self._counts), f"parameters saw different backward counts {self._counts}")
        sq = self._sq[:, :n_back].cpu()  # one sync, off the clock
        metrics: dict[str, float] = {}
        for module in _MODULES:
            rows = [i for i, (m, _, _) in enumerate(self._params) if m == module]
            norms = sq[rows].sum(dim=0).sqrt()  # per-backward global norm, pre-clip
            coef = (self._max_grad_norm / norms.clamp_min(1e-12)).clamp_max(1.0)
            metrics[f"Diag/grad_norm_preclip/{module}"] = float(norms.mean())
            metrics[f"Diag/clip_coef/{module}"] = float(coef.mean())
        with torch.no_grad():
            for i, (module, name, p) in enumerate(self._params):
                before = snapshot[i]
                w_norm = float(torch.linalg.vector_norm(p.detach()))
                delta = float(torch.linalg.vector_norm(p.detach() - before))
                metrics[f"Diag/grad_norm/{module}/{name}"] = float(sq[i].sqrt().mean())
                metrics[f"Diag/weight_norm/{module}/{name}"] = w_norm
                metrics[f"Diag/update_ratio/{module}/{name}"] = delta / (
                    float(torch.linalg.vector_norm(before)) + _RATIO_EPS
                )
        for key, value in metrics.items():
            check_finite(value, key)
        return metrics

    def activation_stats(self, obs: TensorDict | torch.Tensor) -> dict[str, float]:
        """Per-ELU-layer mean, std and saturated fraction of both models, plus policy-level scalars."""
        metrics: dict[str, float] = {}
        with torch.no_grad():
            for module, model in self._models.items():
                outputs: list[torch.Tensor] = []
                handles = [
                    m.register_forward_hook(lambda _m, _i, out, acc=outputs: acc.append(out))
                    for m in model.modules()
                    if isinstance(m, nn.ELU)
                ]
                require(len(handles) > 0, f"{module} has no ELU module to take activation statistics from")
                try:
                    model(obs)
                finally:
                    for handle in handles:
                        handle.remove()
                for layer, out in enumerate(outputs):  # one ELU instance may be called once per layer
                    act = out.float()
                    prefix = f"{module}/layer{layer}"
                    metrics[f"Diag/act_mean/{prefix}"] = float(act.mean())
                    metrics[f"Diag/act_std/{prefix}"] = float(act.std(correction=0))
                    metrics[f"Diag/act_frac_saturated/{prefix}"] = float((act < _ELU_SATURATED_BELOW).float().mean())
            actor = self._models["actor"]
            if isinstance(actor, MLPModel):
                metrics["Diag/action_std"] = float(actor.output_std.mean())
                if actor.obs_normalization:
                    normalizer = actor.obs_normalizer
                    metrics["Diag/obs_norm_mean_abs"] = float(normalizer.mean.abs().mean())  # type: ignore[union-attr]
                    metrics["Diag/obs_norm_std_mean"] = float(normalizer.std.mean())  # type: ignore[union-attr]
        for key, value in metrics.items():
            check_finite(value, key)
        return metrics


def _check_log_probs(old_log_prob: torch.Tensor, new_log_prob: torch.Tensor) -> None:
    require(
        new_log_prob.shape == old_log_prob.shape,
        f"new_log_prob shape {tuple(new_log_prob.shape)} != old_log_prob shape {tuple(old_log_prob.shape)}",
    )
    require(old_log_prob.numel() > 0, "no log-probabilities")


def approx_kl(old_log_prob: torch.Tensor, new_log_prob: torch.Tensor) -> torch.Tensor:
    """KL(old || new) estimated on samples from the old policy: ``mean((r - 1) - log r)``, ``r = p_new / p_old``.

    Schulman's k3 estimator (http://joschu.net/blog/kl-approx.html, as in SB3 and CleanRL): unbiased, and
    non-negative per sample, unlike ``mean(-log r)``. A 0-dim tensor; no device sync.
    """
    _check_log_probs(old_log_prob, new_log_prob)
    log_ratio = new_log_prob - old_log_prob
    return (torch.expm1(log_ratio) - log_ratio).mean()


def clip_fraction(old_log_prob: torch.Tensor, new_log_prob: torch.Tensor, clip_param: float) -> torch.Tensor:
    """Fraction of samples whose ratio left PPO's clip range: ``mean(|r - 1| > clip_param)``. 0-dim, no sync."""
    _check_log_probs(old_log_prob, new_log_prob)
    require(clip_param > 0.0, f"clip_param must be positive, got {clip_param}")
    ratio = torch.exp(new_log_prob - old_log_prob)
    return ((ratio - 1.0).abs() > clip_param).float().mean()


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float | None:
    """``1 - Var(returns - values) / Var(returns)`` (population variances); None when the returns are constant.

    1 is a perfect critic, 0 no better than the mean return, negative worse. One device sync.
    """
    require(values.shape == returns.shape, f"values {tuple(values.shape)} != returns {tuple(returns.shape)}")
    require(returns.numel() > 1, "explained variance needs at least 2 samples")
    var_residual, var_returns = torch.stack(
        [(returns - values).var(correction=0), returns.var(correction=0)]
    ).tolist()
    check_finite(var_returns, "Var(returns)")
    check_finite(var_residual, "Var(returns - values)")
    if var_returns == 0.0:
        return None
    return 1.0 - var_residual / var_returns


@dataclass(frozen=True)
class _RolloutSnapshot:
    observations: TensorDict  # the actor's groups, strided rows
    actions: torch.Tensor  # (rows, action_dim)
    old_log_prob: torch.Tensor  # (rows,)
    values: torch.Tensor  # (T * N,), pre-update critic
    returns: torch.Tensor  # (T * N,)

    def nbytes(self) -> int:
        leaves = [*self.observations.values(True, True), self.actions, self.old_log_prob, self.values, self.returns]
        return sum(leaf.numel() * leaf.element_size() for leaf in leaves)


class PpoHealth:
    """Approx KL, clip fraction and explained variance of each K-iteration's PPO update (module docstring)."""

    def __init__(
        self,
        actor: MLPModel,
        storage: RolloutStorage,
        *,
        clip_param: float,
        every_k: int,
        max_samples: int = PPO_MAX_SAMPLES,
    ) -> None:
        require(every_k >= 1, f"every_k must be >= 1, got {every_k}")
        require(max_samples >= 1, f"max_samples must be >= 1, got {max_samples}")
        check_finite(clip_param, "clip_param")
        require(clip_param > 0.0, f"clip_param must be positive, got {clip_param}")
        require(storage.training_type == "rl", f"PPO health needs RL storage, got {storage.training_type!r}")
        require(not actor.is_recurrent, "PPO health recomputes log-probs without hidden states: MLP actors only")
        require(actor.distribution is not None, "the actor has no action distribution")
        self._actor = actor
        self._storage = storage
        self._clip_param = clip_param
        self._every_k = every_k
        self._max_samples = max_samples
        self._snapshot: _RolloutSnapshot | None = None

    @property
    def snapshot_bytes(self) -> int:
        """Device bytes the current snapshot holds (0 outside a K-iteration)."""
        return 0 if self._snapshot is None else self._snapshot.nbytes()

    def before_update(self, it: int) -> None:
        """Snapshot the rollout (K-iterations only); needs the full storage, after ``compute_returns``."""
        if it % self._every_k != 0:
            self._snapshot = None
            return
        st = self._storage
        require(
            st.step == st.num_transitions_per_env,
            f"storage holds {st.step} of {st.num_transitions_per_env} steps: snapshot only between rollout and update",
        )
        n = st.num_transitions_per_env * st.num_envs
        stride = -(-n // self._max_samples)  # ceil: at most max_samples rows
        rows = torch.arange(0, n, stride, device=st.actions.device)
        require(0 < rows.numel() <= self._max_samples, f"{rows.numel()} rows kept, bound {self._max_samples}")
        observations = st.observations.select(*self._actor.obs_groups).flatten(0, 1)
        self._snapshot = _RolloutSnapshot(  # advanced indexing copies: later storage writes do not reach it
            observations=observations[rows],
            actions=st.actions.flatten(0, 1)[rows],
            old_log_prob=st.actions_log_prob.flatten(0, 1)[rows].squeeze(-1),
            values=st.values.flatten().clone(),
            returns=st.returns.flatten().clone(),
        )

    def after_update(self, it: int) -> dict[str, float]:
        """The three metrics under the updated actor ({} on a non-K iteration). Off the clock; one or two syncs."""
        snapshot = self._snapshot
        self._snapshot = None
        if it % self._every_k != 0:
            return {}
        snapshot = expect(snapshot, f"after_update({it}) without before_update({it})")
        distribution = expect(self._actor.distribution, "the actor lost its action distribution")
        with torch.no_grad():
            # MLPModel.forward minus the sample (mlp_model.py:97-106): no global-RNG draw.
            distribution.update(self._actor.mlp(self._actor.get_latent(snapshot.observations)))
            new_log_prob = self._actor.get_output_log_prob(snapshot.actions)
            kl, clipped = torch.stack(
                [
                    approx_kl(snapshot.old_log_prob, new_log_prob),
                    clip_fraction(snapshot.old_log_prob, new_log_prob, self._clip_param),
                ]
            ).tolist()
        metrics = {"Diag/approx_kl": kl, "Diag/clip_fraction": clipped}
        ev = explained_variance(snapshot.values, snapshot.returns)
        if ev is not None:
            metrics["Diag/explained_variance"] = ev
        for key, value in metrics.items():
            check_finite(value, key)
        require(metrics["Diag/approx_kl"] >= -1e-6, f"approx KL {metrics['Diag/approx_kl']} < 0: k3 is non-negative")
        require(0.0 <= metrics["Diag/clip_fraction"] <= 1.0, f"clip fraction {metrics['Diag/clip_fraction']}")
        return metrics


def for_runner(
    runner: OnPolicyRunner, every_k: int
) -> tuple[TrainingDiagnostics, Callable[[int], None], Callable[[int], None]]:
    """Diagnostics for ``runner`` plus the ``(on_before_update, on_iteration)`` hooks for ``learn_for``.

    The hooks drive both ``TrainingDiagnostics`` and ``PpoHealth`` (same ``every_k``). The caller owns
    ``attach()``/``detach()``. ``on_iteration`` writes through
    ``runner.logger.writer.add_scalar`` at rsl_rl's own step, so W&B steps align.
    """
    alg = runner.alg
    diag = TrainingDiagnostics(alg.get_policy(), alg.critic, every_k, max_grad_norm=alg.max_grad_norm)
    health = PpoHealth(alg.get_policy(), alg.storage, clip_param=alg.clip_param, every_k=every_k)

    def on_before_update(it: int) -> None:
        diag.before_update(it)
        health.before_update(it)

    def on_iteration(it: int) -> None:
        metrics = diag.after_update(it)
        writer = runner.logger.writer
        if not metrics or writer is None:
            health.after_update(it)  # drops the snapshot
            return
        obs = runner.env.get_observations().to(runner.device)  # no env step: off the clock, state untouched
        metrics |= diag.activation_stats(obs)
        metrics |= health.after_update(it)  # last: it re-points the actor's distribution at the snapshot
        for key, value in metrics.items():
            writer.add_scalar(key, value, it)

    return diag, on_before_update, on_iteration
