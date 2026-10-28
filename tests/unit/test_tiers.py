"""The fast tier stays fast: the default -m selects only it, and mprl's pure modules import no mjlab."""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Any

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


def test_importing_mprl_modules_imports_no_mjlab(subprocesses: Any) -> None:
    code = (
        "import sys\n"
        f"for name in {LIGHT_MODULES!r}: __import__(name)\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in {'mjlab', 'mujoco_warp', 'warp'}))\n"
    )
    env = {"PATH": os.environ["PATH"], "PYTHONPATH": str(REPO / "src")}

    done = subprocesses.run([sys.executable, "-c", code], env=env, timeout_s=60.0)

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"
