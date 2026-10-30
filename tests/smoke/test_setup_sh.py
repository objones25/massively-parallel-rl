"""Smoke tests: ``scripts/setup.sh`` and ``docker/uv-sync-base.sh`` install into the right environment.

The incident these pin (2026-10-30): a pod from the stock ``runpod/pytorch`` image (no ``/opt/venv``; the
system python has torch; ``UV_CACHE_DIR`` on the volume; no libEGL) made setup.sh run a plain ``uv sync`` into
the repo's ``.venv`` on the 30 GB volume, which re-downloaded torch and the nvidia wheels until the quota ran out.

Nothing is installed: fake ``uv``, system ``python3.12`` and ``apt-get`` come first on PATH and log their
calls, and the script's test-only overrides (``MPRL_MOUNTS_FILE``, ``MPRL_IMAGE_VENV``, ``MPRL_SYSTEM_PYTHON``,
``MPRL_UV_CACHE``, ``MPRL_PROFILE_FILE``) point the pod's fixed paths into ``tmp_path``. Each script runs through
the ``subprocesses`` fixture (no stdin, bounded wait) with a minimal environment.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[2]
PROCESS_TIMEOUT_S = 60.0
STOCK_UV_CACHE = "/workspace/.cache/uv/"  # what the stock runpod/pytorch image sets


def _env(**extra: str) -> dict[str, str]:
    return {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/tmp"),  # noqa: S108
        "MPRL_ENV_FILE": str(REPO / "tests" / "no-such.env"),  # hermetic: never the developer's .env
        **extra,
    }


def _fake_uv(tmp_path: Path) -> tuple[Path, Path]:
    """A ``uv`` that records its arguments, one call per line, and succeeds; returns (bin dir, log).

    ``uv sync`` also appends its UV_* environment to ``uv-env.log``; ``uv venv ... DIR`` creates ``DIR`` with
    the ``pyvenv.cfg`` line uv 0.13.6 writes for ``--system-site-packages``.
    """
    bin_dir, log, env_log = tmp_path / "bin", tmp_path / "uv-calls.log", tmp_path / "uv-env.log"
    bin_dir.mkdir()
    fake = bin_dir / "uv"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {log}\n"
        'if [[ $1 == sync ]]; then printf "UV_PROJECT_ENVIRONMENT=%s UV_CACHE_DIR=%s UV_NO_SYNC=%s\\n" '
        f'"${{UV_PROJECT_ENVIRONMENT-}}" "${{UV_CACHE_DIR-}}" "${{UV_NO_SYNC-}}" >> {env_log}; fi\n'
        'if [[ $1 == venv ]]; then for d; do :; done; mkdir -p "$d"; '
        "printf 'include-system-site-packages = true\\n' > \"$d/pyvenv.cfg\"; fi\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return bin_dir, log


def _base_sync() -> str:
    lines = (REPO / "docker" / "base-provided-packages.txt").read_text(encoding="utf-8").splitlines()
    names = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return " ".join(["sync --frozen", *(f"--no-install-package {name}" for name in names)])


def _venv(path: Path, *, system_site_packages: str = "true") -> Path:
    path.mkdir(parents=True)
    (path / "pyvenv.cfg").write_text(
        f"home = /usr/bin\ninclude-system-site-packages = {system_site_packages}\nversion_info = 3.12.3\n",
        encoding="utf-8",
    )
    return path


@dataclass(frozen=True)
class FakePod:
    """A pod from the stock RunPod image: fake uv, system python, apt-get and a /proc/mounts with /workspace."""

    env: dict[str, str]
    uv_log: Path
    uv_env_log: Path
    apt_log: Path
    image_venv: Path
    python: Path
    cache: Path
    profile: Path


def _fake_pod(tmp_path: Path, *, torch_rc: int = 0, egl_loads: bool = True, apt_rc: int = 0) -> FakePod:
    bin_dir, uv_log = _fake_uv(tmp_path)
    egl_marker, apt_log = tmp_path / "egl-installed", tmp_path / "apt-calls.log"
    if egl_loads:
        egl_marker.write_text("", encoding="utf-8")
    python, apt = bin_dir / "python3.12", bin_dir / "apt-get"
    python.write_text(
        "#!/usr/bin/env bash\n"
        f'case "$*" in *torch*) exit {torch_rc} ;; *libEGL*) [[ -e {egl_marker} ]] ;; *) exit 2 ;; esac\n',
        encoding="utf-8",
    )
    apt.write_text(
        "#!/usr/bin/env bash\n"
        f'printf \'DEBIAN_FRONTEND=%s %s\\n\' "${{DEBIAN_FRONTEND-}}" "$*" >> {apt_log}\n'
        f"if [[ $1 == install ]]; then (({apt_rc} == 0)) || exit {apt_rc}; touch {egl_marker}; fi\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    apt.chmod(0o755)
    mounts = tmp_path / "mounts"
    mounts.write_text("overlay / overlay rw 0 0\nmfs#eu /workspace fuse rw 0 0\n", encoding="utf-8")
    image_venv, cache = tmp_path / "opt" / "venv", tmp_path / "root-cache-uv"
    profile = tmp_path / "profile.d" / "mprl-uv.sh"
    env = _env(
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        MPRL_MOUNTS_FILE=str(mounts),
        MPRL_IMAGE_VENV=str(image_venv),
        MPRL_SYSTEM_PYTHON=str(python),
        MPRL_UV_CACHE=str(cache),
        MPRL_PROFILE_FILE=str(profile),
        UV_CACHE_DIR=STOCK_UV_CACHE,
    )
    return FakePod(env, uv_log, tmp_path / "uv-env.log", apt_log, image_venv, python, cache, profile)


def _bash() -> str:
    bash = shutil.which("bash")
    assert bash is not None, "bash is required"
    return bash


def _setup_sh(subprocesses: Any, env: dict[str, str], *args: str) -> Any:
    return subprocesses.run([_bash(), str(REPO / "scripts" / "setup.sh"), *args], env=env, timeout_s=PROCESS_TIMEOUT_S)


def _uv_sync_base(subprocesses: Any, env: dict[str, str]) -> Any:
    return subprocesses.run([_bash(), str(REPO / "docker" / "uv-sync-base.sh")], env=env, timeout_s=PROCESS_TIMEOUT_S)


# --- --allow-no-volume (local development) -------------------------------------------------------------


def test_setup_sh_skips_base_provided_packages_in_the_image_venv(tmp_path: Path, subprocesses: Any) -> None:
    bin_dir, log = _fake_uv(tmp_path)
    venv = _venv(tmp_path / "opt" / "venv")
    env = _env(PATH=f"{bin_dir}:{os.environ['PATH']}", MPRL_IMAGE_VENV=str(venv), UV_PROJECT_ENVIRONMENT=str(venv))

    done = _setup_sh(subprocesses, env, "--allow-no-volume")

    assert done.returncode == 0, done.stderr
    assert log.read_text(encoding="utf-8").splitlines()[0] == _base_sync()


def test_setup_sh_allow_no_volume_syncs_a_local_venv_plainly(tmp_path: Path, subprocesses: Any) -> None:
    bin_dir, log = _fake_uv(tmp_path)
    env = _env(
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        MPRL_IMAGE_VENV=str(tmp_path / "no-image-venv"),
        UV_PROJECT_ENVIRONMENT=str(tmp_path / "venv"),
    )

    done = _setup_sh(subprocesses, env, "--allow-no-volume")

    assert done.returncode == 0, done.stderr
    assert log.read_text(encoding="utf-8").splitlines() == ["sync --frozen", "run --no-sync mprl setup"]


def test_setup_sh_allow_no_volume_writes_no_profile(tmp_path: Path, subprocesses: Any) -> None:
    pod = _fake_pod(tmp_path)
    env = {**pod.env, "UV_PROJECT_ENVIRONMENT": str(tmp_path / "venv")}

    done = _setup_sh(subprocesses, env, "--allow-no-volume")

    assert done.returncode == 0, done.stderr
    assert not pod.profile.parent.exists()


@pytest.mark.parametrize("spelling", ["trailing-slash", "symlink"])
def test_setup_sh_recognizes_the_image_venv_however_it_is_spelled(
    spelling: str, tmp_path: Path, subprocesses: Any
) -> None:
    pod = _fake_pod(tmp_path)
    _venv(pod.image_venv)
    link = tmp_path / "venv-link"
    link.symlink_to(pod.image_venv)
    given = {"trailing-slash": f"{pod.image_venv}/", "symlink": str(link)}[spelling]

    done = _setup_sh(subprocesses, {**pod.env, "UV_PROJECT_ENVIRONMENT": given}, "--allow-no-volume")

    assert done.returncode == 0, done.stderr
    assert pod.uv_log.read_text(encoding="utf-8").splitlines()[0] == _base_sync()


# --- volume mode on a pod from the stock image -----------------------------------------------------------


def test_setup_sh_on_a_stock_pod_creates_the_image_venv_from_the_system_python(
    tmp_path: Path, subprocesses: Any
) -> None:
    pod = _fake_pod(tmp_path)

    done = _setup_sh(subprocesses, pod.env)

    assert done.returncode == 0, done.stderr
    assert pod.uv_log.read_text(encoding="utf-8").splitlines() == [
        f"venv --system-site-packages --python {pod.python} {pod.image_venv}",
        _base_sync(),
        "run --no-sync mprl setup",
    ]


def test_setup_sh_on_a_stock_pod_syncs_into_the_image_venv_with_the_cache_off_the_volume(
    tmp_path: Path, subprocesses: Any
) -> None:
    pod = _fake_pod(tmp_path)

    done = _setup_sh(subprocesses, pod.env)

    assert done.returncode == 0, done.stderr
    assert pod.uv_env_log.read_text(encoding="utf-8").splitlines() == [
        f"UV_PROJECT_ENVIRONMENT={pod.image_venv} UV_CACHE_DIR={pod.cache} UV_NO_SYNC=1"
    ]


@pytest.mark.parametrize("cache", [STOCK_UV_CACHE, "/workspace", ""], ids=["stock", "volume-root", "unset"])
def test_setup_sh_moves_a_missing_or_volume_uv_cache_to_the_container_disk(
    cache: str, tmp_path: Path, subprocesses: Any
) -> None:
    pod = _fake_pod(tmp_path)

    done = _setup_sh(subprocesses, {**pod.env, "UV_CACHE_DIR": cache})

    assert done.returncode == 0, done.stderr
    assert f"UV_CACHE_DIR={pod.cache} " in pod.uv_env_log.read_text(encoding="utf-8")


def test_setup_sh_keeps_a_uv_cache_outside_the_volume(tmp_path: Path, subprocesses: Any) -> None:
    pod = _fake_pod(tmp_path)
    own = tmp_path / "own-cache"

    done = _setup_sh(subprocesses, {**pod.env, "UV_CACHE_DIR": str(own)})

    assert done.returncode == 0, done.stderr
    assert f"UV_CACHE_DIR={own} " in pod.uv_env_log.read_text(encoding="utf-8")


def test_setup_sh_refuses_a_stock_pod_whose_system_python_has_no_torch(tmp_path: Path, subprocesses: Any) -> None:
    pod = _fake_pod(tmp_path, torch_rc=1)

    done = _setup_sh(subprocesses, pod.env)

    assert done.returncode == 1
    assert "ghcr.io/objones25/massively-parallel-rl:latest" in done.stderr
    assert not pod.uv_log.exists()


def test_setup_sh_refuses_a_project_environment_on_the_volume(tmp_path: Path, subprocesses: Any) -> None:
    pod = _fake_pod(tmp_path)
    env = {**pod.env, "UV_PROJECT_ENVIRONMENT": "/workspace/massively-parallel-rl/.venv/"}

    done = _setup_sh(subprocesses, env)

    assert done.returncode == 1
    assert "/workspace/massively-parallel-rl/.venv is on the volume" in done.stderr
    assert not pod.uv_log.exists()


def test_setup_sh_installs_libegl1_when_it_cannot_be_loaded(tmp_path: Path, subprocesses: Any) -> None:
    pod = _fake_pod(tmp_path, egl_loads=False)

    done = _setup_sh(subprocesses, pod.env)

    assert done.returncode == 0, done.stderr
    assert pod.apt_log.read_text(encoding="utf-8").splitlines() == [
        "DEBIAN_FRONTEND=noninteractive update",
        "DEBIAN_FRONTEND=noninteractive install -y --no-install-recommends libegl1",
    ]


def test_setup_sh_leaves_apt_alone_when_libegl_loads(tmp_path: Path, subprocesses: Any) -> None:
    pod = _fake_pod(tmp_path)

    done = _setup_sh(subprocesses, pod.env)

    assert done.returncode == 0, done.stderr
    assert not pod.apt_log.exists()


def test_setup_sh_dies_when_libegl1_cannot_be_installed(tmp_path: Path, subprocesses: Any) -> None:
    pod = _fake_pod(tmp_path, egl_loads=False, apt_rc=100)

    done = _setup_sh(subprocesses, pod.env)

    assert done.returncode == 1
    assert "apt-get install libegl1 failed" in done.stderr
    assert not pod.uv_log.exists()


def test_setup_sh_writes_the_uv_login_profile(tmp_path: Path, subprocesses: Any) -> None:
    pod = _fake_pod(tmp_path)

    done = _setup_sh(subprocesses, pod.env)

    assert done.returncode == 0, done.stderr
    exports = [line for line in pod.profile.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    assert exports == [
        f"export UV_PROJECT_ENVIRONMENT='{pod.image_venv}'",
        "export UV_NO_SYNC=1",
        f"export UV_CACHE_DIR='{pod.cache}'",
        f"export UV_PYTHON='{pod.python}'",
        "export UV_LINK_MODE=copy",
        f"case \":$PATH:\" in *':{pod.image_venv}/bin:'*) ;; *) PATH='{pod.image_venv}/bin':\"$PATH\"; export PATH ;; esac",
    ]


def test_setup_sh_rewrites_the_profile_idempotently(tmp_path: Path, subprocesses: Any) -> None:
    pod = _fake_pod(tmp_path)
    first = _setup_sh(subprocesses, pod.env)
    written = pod.profile.read_text(encoding="utf-8")

    second = _setup_sh(subprocesses, pod.env)

    assert (first.returncode, second.returncode) == (0, 0), second.stderr
    assert pod.profile.read_text(encoding="utf-8") == written
    assert [p.name for p in pod.profile.parent.iterdir()] == ["mprl-uv.sh"]


def test_login_profile_puts_the_image_venv_first_on_path_once(tmp_path: Path, subprocesses: Any) -> None:
    pod = _fake_pod(tmp_path)
    _setup_sh(subprocesses, pod.env)

    done = subprocesses.run(
        ["/bin/sh", "-c", f'. {pod.profile}; . {pod.profile}; printf "%s\\n" "$PATH" "$UV_NO_SYNC"'],
        env={"PATH": "/usr/bin:/bin"},
        timeout_s=PROCESS_TIMEOUT_S,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [f"{pod.image_venv}/bin:/usr/bin:/bin", "1"]


# --- docker/uv-sync-base.sh ------------------------------------------------------------------------------


def _no_venv(path: Path) -> Path:
    return path


def _isolated_venv(path: Path) -> Path:
    return _venv(path, system_site_packages="false")


@pytest.mark.parametrize(
    ("make_venv", "reason"),
    [(_no_venv, "has no pyvenv.cfg"), (_isolated_venv, "without system site packages")],
    ids=["missing", "isolated"],
)
def test_uv_sync_base_refuses_a_venv_without_system_site_packages(
    make_venv: Callable[[Path], Path], reason: str, tmp_path: Path, subprocesses: Any
) -> None:
    bin_dir, log = _fake_uv(tmp_path)
    venv = make_venv(tmp_path / "venv")

    done = _uv_sync_base(subprocesses, _env(PATH=f"{bin_dir}:{os.environ['PATH']}", UV_PROJECT_ENVIRONMENT=str(venv)))

    assert done.returncode == 1
    assert reason in done.stderr
    assert not log.exists()


def test_uv_sync_base_refuses_without_a_project_environment(tmp_path: Path, subprocesses: Any) -> None:
    bin_dir, log = _fake_uv(tmp_path)

    done = _uv_sync_base(subprocesses, _env(PATH=f"{bin_dir}:{os.environ['PATH']}"))

    assert done.returncode == 1
    assert "UV_PROJECT_ENVIRONMENT is not set" in done.stderr
    assert not log.exists()


def test_uv_sync_base_syncs_a_venv_with_system_site_packages(tmp_path: Path, subprocesses: Any) -> None:
    bin_dir, log = _fake_uv(tmp_path)
    venv = _venv(tmp_path / "venv")

    done = _uv_sync_base(subprocesses, _env(PATH=f"{bin_dir}:{os.environ['PATH']}", UV_PROJECT_ENVIRONMENT=str(venv)))

    assert done.returncode == 0, done.stderr
    assert log.read_text(encoding="utf-8").splitlines() == [_base_sync()]
