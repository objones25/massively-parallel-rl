"""Secrets are never echoed: sentinel values for every key, then everything a command prints or logs.

Each test sets WANDB_API_KEY, HF_TOKEN and RUNPOD_API_KEY to distinctive sentinels, runs a real command
path in-process (``mprl.cli.main``) whose every service call fails the way the real service fails (the
wandb error class preflight observed, ``hf_raise_for_status`` on a real 401 reply, a RunPod 401 over
``httpx.MockTransport``), and asserts no sentinel reaches stdout, stderr or the log. A positive control in
each test proves the sentinels reached the code under test. The W&B run config is checked in
tests/smoke/test_train_run.py (it needs the mjlab task config); train_run's files in tests/unit/test_train_run.py.
"""

from __future__ import annotations

import functools
import getpass
import logging
import re
from collections.abc import Callable
from pathlib import Path
from unittest.mock import create_autospec

import httpx
import pytest
import wandb
from huggingface_hub import HfApi
from wandb.errors import AuthenticationError

import mprl.cli as cli
import mprl.preflight as preflight_mod
import mprl.runpod as runpod_mod

SENTINELS = {
    "WANDB_API_KEY": "SENTINEL-wandb-3f9a1c7e5b",
    "HF_TOKEN": "hf_SENTINEL-hf-8d2e6a4c0f",
    "RUNPOD_API_KEY": "rpa_SENTINEL-runpod-1b7d5f9e3a",
}
POD = "abc123podid"


def _leaked(*texts: str) -> list[str]:
    """Names of the secrets whose sentinel value appears in any of ``texts``."""
    joined = "\n".join(texts)
    return [name for name, value in SENTINELS.items() if value in joined]


def _runpod_rejecting(seen: list[httpx.Request]) -> httpx.Client:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(401, json={"error": "invalid api key"})

    return httpx.Client(transport=httpx.MockTransport(handle))


@pytest.fixture
def services_reject(monkeypatch: pytest.MonkeyPatch, hub_http_error: Callable[[int], Exception]) -> list[httpx.Request]:
    """W&B, Hugging Face and RunPod all reject the key; returns the RunPod requests (to check the key was sent)."""
    rejected = AuthenticationError("API key must be 40 characters long, yours was 25")
    monkeypatch.setattr(preflight_mod.wandb, "Api", create_autospec(wandb.Api, side_effect=rejected))
    api = create_autospec(HfApi, instance=True)
    api.whoami.side_effect = hub_http_error(401)
    monkeypatch.setattr(preflight_mod, "HfApi", create_autospec(HfApi, return_value=api))
    seen: list[httpx.Request] = []
    monkeypatch.setattr(
        preflight_mod, "check_runpod", functools.partial(preflight_mod.check_runpod, client=_runpod_rejecting(seen))
    )
    return seen


@pytest.fixture
def secrets_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> Path:
    for name, value in SENTINELS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("RUNPOD_POD_ID", POD)
    monkeypatch.setenv("HF_REPO_ID", "someone/mprl-test")
    monkeypatch.setenv("MPRL_VOLUME_ROOT", str(tmp_path / "volume"))
    monkeypatch.setenv("MPRL_ENV_FILE", str(tmp_path / "mprl.env"))
    # The volume check's mount rule would fail tmp_path on Linux CI (it is on "/"); it reads no secret.
    monkeypatch.setattr(preflight_mod, "PROC_MOUNTS", tmp_path / "no-proc-mounts")
    caplog.set_level(logging.DEBUG)
    return tmp_path


def test_failing_preflight_never_echoes_secrets(
    secrets_env: Path,
    services_reject: list[httpx.Request],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    code = cli.main(["preflight", "--only", "wandb", "hf", "runpod", "volume"])

    out, err = capsys.readouterr()
    assert code == 1
    verdicts = re.findall(r"^(PASS|FAIL) (\w+): ", out, re.MULTILINE)
    assert verdicts == [("FAIL", "wandb"), ("FAIL", "hf"), ("FAIL", "runpod"), ("PASS", "volume")]
    assert services_reject[0].headers["Authorization"] == f"Bearer {SENTINELS['RUNPOD_API_KEY']}"  # control
    assert _leaked(out, err, caplog.text) == []


def test_setup_never_echoes_the_secrets_it_writes(
    secrets_env: Path,
    services_reject: list[httpx.Request],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    answers = {"HF_REPO_ID": "someone/mprl-test", "MPRL_VOLUME_ROOT": str(secrets_env / "volume")}
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: answers.get(prompt.split(" ")[0], ""))
    monkeypatch.setattr(getpass, "getpass", lambda prompt: SENTINELS[prompt.split(" ")[0]])
    # The env check builds the mjlab env (pinned in the smoke tier); it reads no secret.
    monkeypatch.setattr(preflight_mod, "check_env", lambda device: f"env check skipped on {device}")

    code = cli.main(["setup", "--cpu"])

    out, err = capsys.readouterr()
    written = (secrets_env / "mprl.env").read_text()
    assert code == 1  # W&B and HF reject the keys
    assert _leaked(written) == ["WANDB_API_KEY", "HF_TOKEN", "RUNPOD_API_KEY"]  # control: they were written
    assert _leaked(out, err, caplog.text) == []


def test_cli_error_with_failed_termination_never_echoes_secrets(
    secrets_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """train-run without HF_REPO_ID: MissingSecret, then the real terminator's DELETE is refused (401)."""
    monkeypatch.delenv("HF_REPO_ID")
    seen: list[httpx.Request] = []
    real_own_terminator = runpod_mod.own_terminator
    monkeypatch.setattr(
        cli, "own_terminator", lambda environ, _client: real_own_terminator(environ, _runpod_rejecting(seen))
    )

    code = cli.main(["train-run", "--sweep", "dbg", "--num-envs", "64", "--seed", "0", "--minutes", "3"])

    out, err = capsys.readouterr()
    assert code == 1
    assert "MissingSecret: HF_REPO_ID is not set" in err
    assert "pod termination also failed: TerminateFailed: GET of pod abc123podid failed with HTTP 401" in err
    assert seen[0].headers["Authorization"] == f"Bearer {SENTINELS['RUNPOD_API_KEY']}"  # control
    assert _leaked(out, err, caplog.text) == []
