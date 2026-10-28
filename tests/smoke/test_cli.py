"""Smoke tests for the ``mprl`` CLI where real work is the behavior, and the installed entry point / bash scripts.

``preflight --only env`` builds the real env and ``throughput`` trains and evaluates for real, so both run
in-process through ``mprl.cli.main``. Only two tests start a process: the installed ``mprl`` entry point
(``uv run mprl --help``) and ``scripts/setup.sh``, each through the ``subprocesses`` fixture (own session,
no stdin, bounded wait, process group killed on timeout) with a minimal environment, so no key from the
developer's shell reaches them. The CLI's control flow is unit-tested in tests/unit/test_cli.py.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest

import mprl.cli as cli
from mprl.budget import load_measurements

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[2]
SUBCOMMANDS = ("setup", "preflight", "throughput", "train-run", "pilot", "sweep", "results")
PROCESS_TIMEOUT_S = 120.0


def _env(**extra: str) -> dict[str, str]:
    # PYTHONPATH: on macOS something sets the UF_HIDDEN flag on the venv's editable-install .pth file,
    # and Python 3.12's site skips hidden .pth files, so the entry point could not import mprl.
    return {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/tmp"),  # noqa: S108
        "PYTHONPATH": str(REPO / "src"),
        "MPRL_ENV_FILE": str(REPO / "tests" / "no-such.env"),  # hermetic: never the developer's .env
        **extra,
    }


def test_installed_entry_point_lists_subcommands(subprocesses: Any) -> None:
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to run the installed entry point"

    done = subprocesses.run(
        [uv, "run", "--no-sync", "--project", str(REPO), "mprl", "--help"], env=_env(), timeout_s=PROCESS_TIMEOUT_S
    )

    assert done.returncode == 0, done.stderr
    assert [name for name in SUBCOMMANDS if name not in done.stdout] == []


_PROC_MOUNTS = Path("/proc/mounts")


@pytest.mark.skipif(
    _PROC_MOUNTS.is_file() and " /workspace " in _PROC_MOUNTS.read_text(), reason="/workspace is mounted here"
)
def test_setup_sh_refuses_without_a_workspace_mount(subprocesses: Any) -> None:
    bash = shutil.which("bash")
    assert bash is not None

    done = subprocesses.run([bash, str(REPO / "scripts" / "setup.sh")], env=_env(), timeout_s=PROCESS_TIMEOUT_S)

    assert done.returncode == 1
    assert "--allow-no-volume" in done.stderr


def test_preflight_cpu_passes_local_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "volume"
    monkeypatch.setenv("MPRL_VOLUME_ROOT", str(root))
    monkeypatch.setenv("MPRL_ENV_FILE", str(tmp_path / "no.env"))

    code = cli.main(["preflight", "--cpu", "--only", "volume", "env"])

    out = capsys.readouterr().out
    assert code == 0, out
    assert "PASS env" in out
    assert "PASS volume" in out
    assert "FAIL" not in out
    assert list(root.iterdir()) == []


def test_throughput_writes_measurements_with_timed_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "volume"
    monkeypatch.setenv("MPRL_VOLUME_ROOT", str(root))
    monkeypatch.setenv("MPRL_ENV_FILE", str(tmp_path / "no.env"))
    (root / "throughput" / "stale").mkdir(parents=True)

    code = cli.main(["throughput", "--counts", "4", "--iterations", "2", "--device", "cpu"])

    assert code == 0
    m = load_measurements(root / "measurements.json")
    assert m.fits == {4: True}
    assert m.steps_per_second[4] > 0
    assert m.eval_seconds_per_checkpoint > 0
    assert not (root / "throughput" / "stale").exists()
