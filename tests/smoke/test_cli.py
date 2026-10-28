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
import time
from pathlib import Path
from typing import Any

import pytest

import mprl.cli as cli
import mprl.throughput as throughput_mod
from mprl.budget import load_measurements
from mprl.checkpoints import CheckpointRef
from mprl.evaluator import EvalConfig
from mprl.throughput import CountMeasurement

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
    measured: list[CountMeasurement] = []
    evaluated: list[tuple[CheckpointRef, EvalConfig, float]] = []  # (checkpoint, config, inner wall seconds)
    real_measure, real_evaluate = throughput_mod.measure_count, throughput_mod.evaluate_checkpoint

    def measure_count(*args: Any, **kwargs: Any) -> CountMeasurement:
        measured.append(real_measure(*args, **kwargs))
        return measured[-1]

    def evaluate_checkpoint(run_id: str, ref: CheckpointRef, cfg: EvalConfig, device: str) -> Any:
        start = time.perf_counter()
        result = real_evaluate(run_id, ref, cfg, device)
        evaluated.append((ref, cfg, time.perf_counter() - start))
        return result

    monkeypatch.setattr(throughput_mod, "measure_count", measure_count)
    monkeypatch.setattr(throughput_mod, "evaluate_checkpoint", evaluate_checkpoint)

    code = cli.main(["throughput", "--counts", "4", "--iterations", "2", "--device", "cpu"])

    assert code == 0
    m = load_measurements(root / "measurements.json")
    assert m.fits == {4: True}
    assert m.steps_per_second == {4: measured[0].steps_per_second}  # what the real training run measured
    (ref, cfg, inner_s) = evaluated[0]
    assert len(evaluated) == 1
    assert ref.path.parent == root / "throughput" / "n00004" / "checkpoints"  # the throughput run's own checkpoint
    assert cfg == EvalConfig()  # the config `mprl results` evaluates with
    assert m.eval_seconds_per_checkpoint >= inner_s  # the timed span encloses the whole evaluation
    assert not (root / "throughput" / "stale").exists()
