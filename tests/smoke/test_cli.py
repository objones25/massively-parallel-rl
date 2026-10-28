"""Smoke tests for the installed ``mprl`` entry point and the bash scripts, run as subprocesses.

Each subprocess gets a minimal environment (PATH, HOME, the volume root) so no key from the
developer's shell reaches it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[2]
SUBCOMMANDS = ("setup", "preflight", "throughput", "train-run", "pilot", "sweep", "results")


def _uv() -> str:
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to run the CLI smoke tests"
    return uv


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


def _mprl(*args: str, env: dict[str, str], timeout: float = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603  fixed argv
        [_uv(), "run", "--no-sync", "--project", str(REPO), "mprl", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        check=False,
    )


def test_cli_help_lists_subcommands(tmp_path: Path) -> None:
    done = _mprl("--help", env=_env())

    assert done.returncode == 0, done.stderr
    assert [name for name in SUBCOMMANDS if name not in done.stdout] == []


def test_train_run_refuses_without_pod_id(tmp_path: Path) -> None:
    done = _mprl(
        "train-run",
        "--sweep",
        "dbg",
        "--num-envs",
        "4",
        "--seed",
        "0",
        "--minutes",
        "3",
        env=_env(MPRL_VOLUME_ROOT=str(tmp_path / "volume"), RUNPOD_API_KEY="rpa_unused"),
    )

    assert done.returncode != 0
    assert "--no-terminate" in done.stderr
    assert not (tmp_path / "volume").exists()


def test_preflight_cpu_passes_local_checks(tmp_path: Path) -> None:
    root = tmp_path / "volume"

    done = _mprl("preflight", "--cpu", "--only", "volume", "env", env=_env(MPRL_VOLUME_ROOT=str(root)))

    assert done.returncode == 0, done.stdout + done.stderr
    assert "PASS env" in done.stdout
    assert "PASS volume" in done.stdout
    assert "FAIL" not in done.stdout
    assert list(root.iterdir()) == []


def test_preflight_exits_nonzero_on_a_failed_check(tmp_path: Path) -> None:
    done = _mprl("preflight", "--cpu", "--only", "wandb", env=_env(MPRL_VOLUME_ROOT=str(tmp_path)))

    assert done.returncode == 1
    assert "FAIL wandb: MissingSecret" in done.stdout


def test_shellcheck_scripts_clean() -> None:
    scripts = sorted(str(p) for p in (REPO / "scripts").glob("*.sh"))
    assert [Path(s).name for s in scripts] == ["setup.sh", "train.sh"]

    done = subprocess.run(  # noqa: S603  fixed argv
        [_uv(), "tool", "run", "--from", "shellcheck-py", "shellcheck", *scripts],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert done.returncode == 0, done.stdout + done.stderr


_PROC_MOUNTS = Path("/proc/mounts")


@pytest.mark.skipif(
    _PROC_MOUNTS.is_file() and " /workspace " in _PROC_MOUNTS.read_text(), reason="/workspace is mounted here"
)
def test_setup_sh_refuses_without_a_workspace_mount(tmp_path: Path) -> None:
    bash = shutil.which("bash")
    assert bash is not None

    done = subprocess.run(  # noqa: S603  fixed argv
        [bash, str(REPO / "scripts" / "setup.sh")], capture_output=True, text=True, env=_env(), timeout=60, check=False
    )

    assert done.returncode == 1
    assert "--allow-no-volume" in done.stderr


def test_throughput_writes_measurements_with_timed_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mprl.cli as cli
    from mprl.budget import load_measurements

    root = tmp_path / "volume"
    monkeypatch.setenv("MPRL_VOLUME_ROOT", str(root))
    (root / "throughput" / "stale").mkdir(parents=True)

    monkeypatch.setenv("MPRL_ENV_FILE", str(tmp_path / "no.env"))

    code = cli.main(["throughput", "--counts", "4", "--iterations", "2", "--device", "cpu"])

    assert code == 0
    m = load_measurements(root / "measurements.json")
    assert m.fits == {4: True}
    assert m.steps_per_second[4] > 0
    assert m.eval_seconds_per_checkpoint > 0
    assert not (root / "throughput" / "stale").exists()


def test_env_file_override_is_loaded(tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text(f"MPRL_VOLUME_ROOT={tmp_path / 'from-env-file'}\n")

    done = _mprl("preflight", "--cpu", "--only", "volume", env={**_env(), "MPRL_ENV_FILE": str(env_file)})

    assert done.returncode == 0, done.stdout + done.stderr
    assert "from-env-file is writable" in done.stdout
