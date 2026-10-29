"""Smoke test: ``scripts/train.sh`` hands its arguments (or the default) to ``uv run --no-sync mprl``.

A fake ``uv`` first on PATH prints the argv it was exec'd with, so nothing is installed or trained.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from support import Subprocesses

pytestmark = pytest.mark.smoke

TRAIN_SH = Path(__file__).resolve().parents[2] / "scripts" / "train.sh"


def _uv_argv(tmp_path: Path, subprocesses: Subprocesses, *args: str) -> str:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text('#!/bin/sh\nprintf "%s\\n" "$*"\n')
    fake_uv.chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "UV_PROJECT_ENVIRONMENT": str(tmp_path / "venv")}
    done = subprocesses.run(["/bin/bash", str(TRAIN_SH), *args], env=env, timeout_s=30.0)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def test_no_arguments_run_the_main_sweep_reusing_the_pilot(tmp_path: Path, subprocesses: Subprocesses) -> None:
    assert _uv_argv(tmp_path, subprocesses) == "run --no-sync mprl sweep --name main --reuse-pilot"


def test_arguments_pass_through_unchanged(tmp_path: Path, subprocesses: Subprocesses) -> None:
    assert _uv_argv(tmp_path, subprocesses, "pilot", "--price-per-hour", "1.79") == (
        "run --no-sync mprl pilot --price-per-hour 1.79"
    )
