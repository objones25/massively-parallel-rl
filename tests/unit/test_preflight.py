"""Unit tests for mprl.preflight: the check runner and the checks that need no network.

The RunPod check runs over a real httpx.Client with httpx.MockTransport (httpx 0.28.1). The W&B and
HF checks are exercised against the real services in tests/network/test_preflight_real.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import create_autospec

import httpx
import pytest
import wandb
from wandb.errors import AuthenticationError, CommError

import mprl.preflight as preflight_mod
from mprl.errors import MissingSecret, OwnPodUnknown, PreflightFailed
from mprl.negative_space import CheckFailed, require
from mprl.preflight import (
    CheckResult,
    check_hf,
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


def test_default_checks_skip_cuda_and_pod_without_gpu() -> None:
    assert [name for name, _ in default_checks({}, require_gpu=True)] == [
        "cuda",
        "env",
        "wandb",
        "hf",
        "runpod",
        "volume",
    ]
    assert [name for name, _ in default_checks({}, require_gpu=False)] == ["env", "wandb", "hf", "volume"]


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

    assert detail == f"pod {POD} is RUNNING; API key accepted"
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
