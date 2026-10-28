"""GPU-hour cost estimate for a sweep, and the spec's cut order when it exceeds the cap.

Formula (``estimate``), with ``io`` = ``io_overhead_fraction`` and ``e`` = ``eval_seconds_per_checkpoint``:

    runs          = len(env_counts) * len(seeds)
    sweep hours   = runs * budget_minutes / 60 * (1 + io)
    long hours    = max(0, long_run_minutes - long_run_done_minutes) / 60 * (1 + io)
    evaluations   = runs * budget_minutes / eval_every_minutes
                    + max(0, long_run_minutes - long_run_done_minutes) / eval_every_minutes
    eval hours    = evaluations * e / 3600
    total hours   = setup_hours + sweep hours + long hours + eval hours
    total usd     = total hours * price_per_hour

With ``io = e = 0`` and the default sweep this is the spec's table (section 5):
0.25 (throughput) + 4 (debugging) + 15 x 0.75 = 11.25 + 3 = 18.5 h, $33.115 at $1.79/h.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Final

from mprl.errors import BudgetExceeded, MeasurementsCorrupt
from mprl.negative_space import CheckFailed, check_finite, require
from mprl.run_state import write_json_atomic

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mprl.config import SweepSpec

PRICE_PER_HOUR: Final = 1.79  # Secure Cloud A100, read 2026-10-27
SETUP_HOURS: Final = 4.25  # spec table: throughput test 0.25 + debugging/smoke/eval/rendering 4
CAP_USD: Final = 40.0
NEVER_DROPPED: Final = 64  # the contrast the demo exists to show
CUT_ORDER: Final = ("eval_every_6min", "long_run_half", "drop_16384", "two_seeds")


@dataclass(frozen=True)
class Measurements:
    """Throughput-test (and pilot) measurements. ``fits`` and ``steps_per_second`` are keyed by num_envs."""

    eval_seconds_per_checkpoint: float
    io_overhead_fraction: float
    fits: dict[int, bool]
    steps_per_second: dict[int, float]

    def __post_init__(self) -> None:
        for name in ("eval_seconds_per_checkpoint", "io_overhead_fraction"):
            value = getattr(self, name)
            check_finite(value, name)
            require(value >= 0.0, f"{name} must be non-negative, got {value}")
        for n, sps in self.steps_per_second.items():
            check_finite(sps, f"steps_per_second[{n}]")
            require(sps > 0.0, f"steps_per_second[{n}] must be positive, got {sps}")
            require(self.fits.get(n) is True, f"steps_per_second has num_envs={n}, which is not recorded as fitting")


@dataclass(frozen=True)
class CostEstimate:
    lines: tuple[tuple[str, float], ...]  # (item, hours)
    total_hours: float
    total_usd: float

    def describe(self) -> str:
        price = self.total_usd / self.total_hours if self.total_hours > 0 else 0.0
        rows = [f"  {hours:7.3f} h  ${hours * price:7.2f}  {item}" for item, hours in self.lines]
        return "\n".join([*rows, f"  {self.total_hours:7.3f} h  ${self.total_usd:7.2f}  total"])


def save_measurements(path: Path, m: Measurements) -> None:
    write_json_atomic(path, asdict(m))  # JSON object keys become strings


def load_measurements(path: Path) -> Measurements:
    """The measurements in ``path`` (which must exist); ``MeasurementsCorrupt`` naming the path otherwise.

    The file is outside input (a crash mid-write on an older version, a hand edit), so an unreadable or
    out-of-range value is an operating error, not a crash: every parse failure, and a value
    ``Measurements.__post_init__`` rejects, becomes one.
    """
    text = path.read_text()
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise TypeError(f"top level is {type(data).__name__}, not a JSON object")
        return Measurements(
            eval_seconds_per_checkpoint=float(data["eval_seconds_per_checkpoint"]),
            io_overhead_fraction=float(data["io_overhead_fraction"]),
            fits={int(n): bool(ok) for n, ok in data["fits"].items()},
            steps_per_second={int(n): float(sps) for n, sps in data["steps_per_second"].items()},
        )
    except (ValueError, KeyError, TypeError, AttributeError, CheckFailed) as error:  # JSONDecodeError is a ValueError
        raise MeasurementsCorrupt(
            f"{path} is corrupt ({type(error).__name__}: {error}); re-run `mprl throughput` to rewrite it"
        ) from error


def estimate(
    sweep: SweepSpec,
    m: Measurements,
    price_per_hour: float = PRICE_PER_HOUR,
    setup_hours: float = SETUP_HOURS,
    long_run_done_minutes: float = 0.0,
) -> CostEstimate:
    """GPU hours and dollars for ``sweep`` (formula in the module docstring)."""
    check_finite(price_per_hour, "price_per_hour")
    require(price_per_hour > 0.0, f"price_per_hour must be positive, got {price_per_hour}")
    require(setup_hours >= 0.0, f"setup_hours must be non-negative, got {setup_hours}")
    require(long_run_done_minutes >= 0.0, f"long_run_done_minutes must be non-negative, got {long_run_done_minutes}")
    overhead = 1.0 + m.io_overhead_fraction
    runs = len(sweep.env_counts) * len(sweep.seeds)
    long_minutes = max(0.0, sweep.long_run_minutes - long_run_done_minutes)
    evaluations = (runs * sweep.budget_minutes + long_minutes) / sweep.eval_every_minutes
    lines = (
        ("setup: throughput test + debugging, smoke tests, evaluation, rendering", setup_hours),
        (
            f"sweep: {runs} runs x {sweep.budget_minutes:g} min x (1 + io {m.io_overhead_fraction:.3f})",
            runs * sweep.budget_minutes / 60.0 * overhead,
        ),
        (
            f"long run: {long_minutes:g} of {sweep.long_run_minutes:g} min left x (1 + io)",
            long_minutes / 60.0 * overhead,
        ),
        (
            f"evaluations: {evaluations:g} every {sweep.eval_every_minutes:g} min x "
            f"{m.eval_seconds_per_checkpoint:g} s",
            evaluations * m.eval_seconds_per_checkpoint / 3600.0,
        ),
    )
    total_hours = math.fsum(hours for _, hours in lines)
    return CostEstimate(lines, total_hours, total_hours * price_per_hour)


def _eval_every_6min(sweep: SweepSpec) -> SweepSpec:
    return replace(sweep, eval_every_minutes=max(sweep.eval_every_minutes, 6.0))


def _long_run_half(sweep: SweepSpec) -> SweepSpec:
    return replace(sweep, long_run_minutes=sweep.long_run_minutes / 2.0)


def _drop_16384(sweep: SweepSpec) -> SweepSpec:
    return replace(sweep, env_counts=tuple(n for n in sweep.env_counts if n != 16384))


def _two_seeds(sweep: SweepSpec) -> SweepSpec:
    return replace(sweep, seeds=sweep.seeds[:2])


_CUTS: Final[dict[str, Callable[[SweepSpec], SweepSpec]]] = {
    "eval_every_6min": _eval_every_6min,
    "long_run_half": _long_run_half,
    "drop_16384": _drop_16384,
    "two_seeds": _two_seeds,
}


def plan_within_budget(
    sweep: SweepSpec, m: Measurements, cap_usd: float = CAP_USD
) -> tuple[SweepSpec, tuple[str, ...], CostEstimate]:
    """Apply ``CUT_ORDER`` one cut at a time until the estimate is within ``cap_usd``.

    A cut that would not change the sweep (16384 already absent, fewer than 3 seeds) is not applied
    or reported. Raises ``BudgetExceeded`` with the final estimate if every cut is not enough.
    """
    require(tuple(_CUTS) == CUT_ORDER, "cut table out of sync with CUT_ORDER")
    require(NEVER_DROPPED in sweep.env_counts, f"sweep {sweep.name} lacks the {NEVER_DROPPED}-env count")
    est = estimate(sweep, m)
    applied: list[str] = []
    for name in CUT_ORDER:
        if est.total_usd <= cap_usd:
            break
        cut = _CUTS[name](sweep)
        if cut == sweep:
            continue
        sweep = cut
        applied.append(name)
        est = estimate(sweep, m)
    require(NEVER_DROPPED in sweep.env_counts, f"a cut removed the {NEVER_DROPPED}-env count")
    if est.total_usd > cap_usd:
        raise BudgetExceeded(
            f"sweep {sweep.name}: ${est.total_usd:.2f} ({est.total_hours:.2f} h) exceeds the ${cap_usd:.2f} cap "
            f"after cuts {applied}:\n{est.describe()}"
        )
    return sweep, tuple(applied), est
