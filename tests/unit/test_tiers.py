"""The fast tier stays fast: the default -m selects only it, and neither its tests nor mprl's pure modules import mjlab."""

from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

import pytest
from support import DEFAULT_MARKEXPR

REPO = Path(__file__).resolve().parents[2]
LIGHT_MODULES = (
    "mprl.cli", "mprl.sweep", "mprl.results", "mprl.preflight", "mprl.evaluator", "mprl.train",
    "mprl.throughput", "mprl.curves", "mprl.publish", "mprl.clock",
)  # fmt: skip


def test_default_marker_expression_is_the_fast_tier() -> None:
    addopts = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]["addopts"]

    assert addopts[addopts.index("-m") + 1] == DEFAULT_MARKEXPR
    assert not any(opt.startswith("--cov") for opt in addopts)


@pytest.fixture
def fast_tier_session(request: pytest.FixtureRequest) -> None:
    """Skip unless this session collected only the fast tier (the heavy tiers import mjlab themselves)."""
    if request.config.getoption("markexpr") != DEFAULT_MARKEXPR:
        pytest.skip("heavy tiers collected in this session; they import mjlab themselves")


@pytest.mark.usefixtures("fast_tier_session")
def test_fast_tier_session_has_not_imported_mjlab() -> None:
    """Every fast-tier test module is imported by now, and the pure mprl modules here; none may import mjlab.

    A fast-tier test that imports or lazily triggers mjlab before this one runs also fails it. Only
    meaningful when the heavy tiers were not collected (``fast_tier_session``).
    """
    modules = [importlib.import_module(name) for name in LIGHT_MODULES]

    assert len(modules) == len(LIGHT_MODULES)
    assert sorted(m for m in sys.modules if m.split(".")[0] in {"mjlab", "mujoco_warp", "warp"}) == []
