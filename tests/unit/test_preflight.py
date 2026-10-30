"""Unit tests for mprl.preflight: the check runner and the checks that need no network or GPU.

The RunPod check runs over a real httpx.Client with httpx.MockTransport (httpx 0.28.1). The W&B and
HF checks are exercised against the real services in tests/network/test_preflight_real.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, create_autospec

import httpx
import pytest
import wandb
from huggingface_hub import HfApi
from wandb.errors import AuthenticationError, CommError

import mprl.preflight as preflight_mod
from mprl.errors import MissingSecret, OwnPodUnknown, PreflightFailed
from mprl.negative_space import CheckFailed, require
from mprl.preflight import (
    CheckResult,
    check_egl,
    check_hf,
    check_python_env,
    check_runpod,
    check_volume,
    check_wandb,
    default_checks,
    run_checks,
)

POD = "abc123podid"


def test_run_checks_reports_failure_detail() -> None:
    def missing() -> str:
        raise MissingSecret("HF_TOKEN is not set")

    results = run_checks([("ok", lambda: "fine"), ("hf", missing)])

    assert results == [
        CheckResult("ok", ok=True, detail="fine"),
        CheckResult("hf", ok=False, detail="MissingSecret: HF_TOKEN is not set"),
    ]


def test_run_checks_does_not_swallow_programmer_errors() -> None:
    def broken() -> str:
        require(False, "a bug in the check")
        return "unreachable"

    with pytest.raises(CheckFailed, match=r"^a bug in the check$"):
        run_checks([("broken", broken)])


@pytest.mark.parametrize(
    "error",
    [
        ImportError("libEGL.so.1: cannot open shared object file"),
        AttributeError("'NoneType' object has no attribute 'eglQueryString'"),
        OSError(122, "Disk quota exceeded"),
    ],
    ids=["import", "attribute", "os"],
)
def test_run_checks_reports_an_environment_error_and_runs_the_rest(error: Exception) -> None:
    def broken_install() -> str:
        raise error

    results = run_checks([("env", broken_install), ("volume", lambda: "fine")])

    assert results == [
        CheckResult("env", ok=False, detail=f"{type(error).__name__}: {error}"),
        CheckResult("volume", ok=True, detail="fine"),
    ]


def test_default_checks_on_the_pod_run_every_check(tmp_path: Path) -> None:
    assert [name for name, _ in default_checks({}, require_gpu=True, repo_dir=tmp_path)] == [
        "cuda",
        "python-env",
        "egl",
        "env",
        "wandb",
        "hf",
        "runpod",
        "volume",
    ]


def test_default_checks_skip_the_pod_only_checks_without_gpu(tmp_path: Path) -> None:
    assert [name for name, _ in default_checks({}, require_gpu=False, repo_dir=tmp_path)] == [
        "env",
        "wandb",
        "hf",
        "volume",
    ]


# --- python-env: the interpreter, torch and the uv cache are off the volume -----------------------------

SYSTEM_TORCH = "/usr/local/lib/python3.12/dist-packages/torch/__init__.py"


def _python_env(tmp_path: Path, env: dict[str, str] | None = None, **overrides: Any) -> str:
    image_venv = tmp_path / "opt" / "venv"
    image_venv.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_dir": tmp_path / "repo",
        "torch_file": SYSTEM_TORCH,
        "prefix": str(image_venv),
        "linux": True,
        "image_venv": image_venv,
    }
    kwargs.update(overrides)
    base = {"MPRL_VOLUME_ROOT": str(tmp_path / "volume"), "UV_NO_SYNC": "1", "UV_CACHE_DIR": "/root/.cache/uv"}
    return check_python_env(base if env is None else env, **kwargs)


def test_check_python_env_passes_the_image_venv_with_the_base_torch(tmp_path: Path) -> None:
    detail = _python_env(tmp_path)

    assert detail == f"python {tmp_path / 'opt' / 'venv'}; torch {SYSTEM_TORCH}"


@pytest.mark.parametrize(
    ("override", "problem"),
    [
        ({"torch_file": "/workspace/massively-parallel-rl/.venv/lib/torch/__init__.py"}, "torch is imported from"),
        ({"prefix": "/workspace/massively-parallel-rl/.venv"}, "the interpreter's prefix"),
    ],
    ids=["torch", "prefix"],
)
def test_check_python_env_fails_on_the_volume(override: dict[str, str], problem: str, tmp_path: Path) -> None:
    with pytest.raises(PreflightFailed, match=f"{problem} /workspace/massively-parallel-rl/.venv"):
        _python_env(tmp_path, **override)


def test_check_python_env_fails_when_torch_is_under_the_volume_root(tmp_path: Path) -> None:
    torch_file = str(tmp_path / "volume" / "venv" / "torch" / "__init__.py")

    with pytest.raises(PreflightFailed, match=r"torch is imported from .*volume/venv/torch"):
        _python_env(tmp_path, torch_file=torch_file)


def test_check_python_env_fails_when_torch_is_in_the_repo_venv(tmp_path: Path) -> None:
    torch_file = str(tmp_path / "repo" / ".venv" / "lib" / "torch" / "__init__.py")

    with pytest.raises(PreflightFailed, match=r"torch is imported from .*repo/\.venv"):
        _python_env(tmp_path, torch_file=torch_file)


def test_check_python_env_fails_with_the_uv_cache_on_the_volume(tmp_path: Path) -> None:
    env = {"UV_NO_SYNC": "1", "UV_CACHE_DIR": "/workspace/.cache/uv/"}

    with pytest.raises(
        PreflightFailed,
        match=r"UV_CACHE_DIR=/workspace/\.cache/uv/ is on the volume; export UV_CACHE_DIR=/root/\.cache/uv",
    ):
        _python_env(tmp_path, env)


def test_check_python_env_fails_when_uv_run_would_sync_the_image_venv(tmp_path: Path) -> None:
    with pytest.raises(PreflightFailed, match=r"UV_NO_SYNC is not set.*export UV_NO_SYNC=1"):
        _python_env(tmp_path, {"UV_CACHE_DIR": "/root/.cache/uv"})


def test_check_python_env_needs_no_uv_no_sync_off_linux(tmp_path: Path) -> None:
    detail = _python_env(tmp_path, {}, linux=False)

    assert detail.startswith("python ")


# --- egl ----------------------------------------------------------------------------------------------


def test_check_egl_fails_with_the_apt_fix_when_libegl_is_missing() -> None:
    def load(name: str) -> object:
        raise OSError(f"{name}: cannot open shared object file: No such file or directory")

    with pytest.raises(PreflightFailed, match=r"libEGL\.so\.1 cannot be loaded.*apt-get install -y libegl1"):
        check_egl(load)


def test_check_egl_loads_libegl() -> None:
    loaded: list[str] = []

    detail = check_egl(lambda name: loaded.append(name))

    assert (detail, loaded) == ("libEGL.so.1 loads", ["libEGL.so.1"])


def test_check_volume_writes_reads_and_deletes(tmp_path: Path) -> None:
    root = tmp_path / "volume" / "mprl"

    detail = check_volume({"MPRL_VOLUME_ROOT": str(root)})

    assert str(root) in detail
    assert list(root.iterdir()) == []


def test_check_volume_fails_when_root_is_not_a_directory(tmp_path: Path) -> None:
    root = tmp_path / "a-file"
    root.write_text("")

    with pytest.raises(PreflightFailed, match="a-file"):
        check_volume({"MPRL_VOLUME_ROOT": str(root)})


def _mounts(tmp_path: Path, *points: str) -> Path:
    mounts = tmp_path / "mounts"
    lines = ["overlay / overlay rw,relatime 0 0", *(f"mfs {point} fuse rw 0 0" for point in points)]
    mounts.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return mounts


def test_check_volume_fails_on_the_container_disk(tmp_path: Path) -> None:
    root = tmp_path / "volume" / "mprl"

    with pytest.raises(PreflightFailed, match=r"is on the container disk \(mount /\)"):
        check_volume({"MPRL_VOLUME_ROOT": str(root)}, mounts=_mounts(tmp_path, "/proc", "/sys"))


def test_check_volume_fails_before_writing_to_the_container_disk(tmp_path: Path) -> None:
    root = tmp_path / "volume" / "mprl"

    with pytest.raises(PreflightFailed, match="container disk"):
        check_volume({"MPRL_VOLUME_ROOT": str(root)}, mounts=_mounts(tmp_path))
    assert not root.exists()


def test_check_volume_passes_under_a_mounted_volume(tmp_path: Path) -> None:
    volume = (tmp_path / "net vol").resolve()
    root = volume / "mprl"

    detail = check_volume({"MPRL_VOLUME_ROOT": str(root)}, mounts=_mounts(tmp_path, str(volume).replace(" ", "\\040")))

    assert detail == f"{root} is writable (on mount {volume})"


def test_check_volume_skips_the_mount_rule_without_a_mounts_file(tmp_path: Path) -> None:
    root = tmp_path / "volume"

    detail = check_volume({"MPRL_VOLUME_ROOT": str(root)}, mounts=tmp_path / "no-such-mounts")

    assert detail == f"{root} is writable"


@pytest.mark.parametrize(
    ("check", "env", "missing"),
    [
        (check_wandb, {}, "WANDB_API_KEY"),
        (check_hf, {"HF_REPO_ID": "u/r"}, "HF_TOKEN"),
        (check_hf, {"HF_TOKEN": "hf_x"}, "HF_REPO_ID"),
    ],
)
def test_service_checks_name_the_missing_key(check, env: dict[str, str], missing: str) -> None:
    with pytest.raises(MissingSecret, match=missing):
        check(env)


def _client(status: int, body: dict[str, str] | None, seen: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_check_runpod_reads_own_pod() -> None:
    seen: list[httpx.Request] = []
    env = {"RUNPOD_API_KEY": "rpa_key", "RUNPOD_POD_ID": POD}

    detail = check_runpod(env, _client(200, {"status": "RUNNING"}, seen))

    assert detail == (
        f"pod {POD} is RUNNING; API key accepted for GET only: self-termination (DELETE) needs a key with "
        '"All" permission, which this check does not prove'
    )
    assert [(r.method, r.url.path, r.headers["Authorization"]) for r in seen] == [
        ("GET", f"/v2/pods/{POD}", "Bearer rpa_key")
    ]


def test_check_runpod_fails_when_pod_is_not_found() -> None:
    env = {"RUNPOD_API_KEY": "rpa_key", "RUNPOD_POD_ID": POD}

    with pytest.raises(PreflightFailed, match="not found"):
        check_runpod(env, _client(404, None, []))


def test_check_runpod_fails_on_rejected_key() -> None:
    env = {"RUNPOD_API_KEY": "rpa_bad", "RUNPOD_POD_ID": POD}

    with pytest.raises(PreflightFailed, match="HTTP 401"):
        check_runpod(env, _client(401, {"error": "unauthorized"}, []))


def test_check_runpod_rejects_a_malformed_pod_id() -> None:
    env = {"RUNPOD_API_KEY": "rpa_key", "RUNPOD_POD_ID": "../../etc"}

    with pytest.raises(OwnPodUnknown, match="malformed"):
        check_runpod(env, _client(200, {"status": "RUNNING"}, []))


def test_check_runpod_needs_the_api_key() -> None:
    with pytest.raises(MissingSecret, match="RUNPOD_API_KEY"):
        check_runpod({"RUNPOD_POD_ID": POD}, _client(200, {"status": "RUNNING"}, []))


@pytest.mark.parametrize(
    "error",
    [
        AuthenticationError("Failed to verify credentials: the service process is busy and did not respond in time"),
        CommError("network unreachable"),
    ],
)
def test_check_wandb_maps_auth_and_network_failures(error: Exception, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight_mod.wandb, "Api", create_autospec(wandb.Api, side_effect=error))

    results = run_checks([("wandb", lambda: check_wandb({"WANDB_API_KEY": "k" * 40}))])

    assert results[0].ok is False
    assert results[0].detail.startswith("PreflightFailed: W&B")


# --- hf: the token can write (huggingface_hub 2.3.0 whoami(): auth.accessToken.role, read in _login.py) ----

HF_ENV = {"HF_TOKEN": "hf_x", "HF_REPO_ID": "someone/mprl"}


def _hf_api(monkeypatch: pytest.MonkeyPatch, auth: dict[str, Any]) -> MagicMock:
    api = create_autospec(HfApi, instance=True)
    api.whoami.return_value = {"type": "user", "name": "someone", "auth": auth}
    monkeypatch.setattr(preflight_mod, "HfApi", create_autospec(HfApi, return_value=api))
    return api


def test_check_hf_fails_for_a_read_only_token(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _hf_api(monkeypatch, {"type": "access_token", "accessToken": {"displayName": "t", "role": "read"}})

    with pytest.raises(PreflightFailed, match="HF_TOKEN is a read token"):
        check_hf(HF_ENV)
    api.create_repo.assert_not_called()


def test_check_hf_passes_a_write_token(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _hf_api(monkeypatch, {"type": "access_token", "accessToken": {"displayName": "t", "role": "write"}})

    detail = check_hf(HF_ENV)

    assert detail == "write token for someone; repo someone/mprl exists"
    api.create_repo.assert_called_once_with("someone/mprl", private=True, repo_type="model", exist_ok=True)


@pytest.mark.parametrize(
    "auth",
    [{"type": "access_token", "accessToken": {"displayName": "t", "role": "fineGrained"}}, {"type": "oauth"}],
    ids=["fine-grained", "no-role"],
)
def test_check_hf_says_when_write_access_is_unverified(auth: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    _hf_api(monkeypatch, auth)

    detail = check_hf(HF_ENV)

    assert detail.startswith("token for someone (role: ")
    assert detail.endswith("write access not verified); repo someone/mprl exists")
