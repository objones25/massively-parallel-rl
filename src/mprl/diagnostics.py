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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from rsl_rl.models import MLPModel
from torch import nn

from mprl.negative_space import check_finite, expect, require

if TYPE_CHECKING:
    from collections.abc import Callable

    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict
    from torch.utils.hooks import RemovableHandle

_MAX_BACKWARDS = 256  # PPO backwards per update = num_learning_epochs * num_mini_batches (G1: 5 * 4)
_ELU_SATURATED_BELOW = -0.99  # ELU(alpha=1) floors at -1
_RATIO_EPS = 1e-8  # zero-initialised biases would otherwise divide by zero
_MODULES = ("actor", "critic")


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


def for_runner(
    runner: OnPolicyRunner, every_k: int
) -> tuple[TrainingDiagnostics, Callable[[int], None], Callable[[int], None]]:
    """Diagnostics for ``runner`` plus the ``(on_before_update, on_iteration)`` hooks for ``learn_for``.

    The caller owns ``attach()``/``detach()``. ``on_iteration`` writes through
    ``runner.logger.writer.add_scalar`` at rsl_rl's own step, so W&B steps align.
    """
    alg = runner.alg
    diag = TrainingDiagnostics(alg.get_policy(), alg.critic, every_k, max_grad_norm=alg.max_grad_norm)

    def on_iteration(it: int) -> None:
        metrics = diag.after_update(it)
        writer = runner.logger.writer
        if not metrics or writer is None:
            return
        obs = runner.env.get_observations().to(runner.device)  # no env step: off the clock, state untouched
        metrics |= diag.activation_stats(obs)
        for key, value in metrics.items():
            writer.add_scalar(key, value, it)

    return diag, diag.before_update, on_iteration
