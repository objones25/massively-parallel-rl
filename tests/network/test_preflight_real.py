"""The preflight's service checks against the real W&B, Hugging Face and RunPod APIs.

Each test skips unless its keys are in the environment. The HF check creates ``HF_REPO_ID`` (private)
if it does not exist. The RunPod check only reads, and needs the pod id, so it runs on the pod only.
"""

from __future__ import annotations

import os

import pytest

from mprl.preflight import check_hf, check_runpod, check_wandb

pytestmark = pytest.mark.network

# Network tests keep WANDB_/HF_/RUNPOD_ (tests/conftest.py _clean_env); RUNPOD_POD_ID may be exported by hand.
_ENV = dict(os.environ)


def _require(*names: str) -> dict[str, str]:
    missing = [n for n in names if not _ENV.get(n)]
    if missing:
        pytest.skip(f"needs {', '.join(missing)}")
    return {n: _ENV[n] for n in names}


def test_wandb_check_accepts_real_key() -> None:
    assert "logged in as" in check_wandb(_require("WANDB_API_KEY"))


def test_hf_check_accepts_real_token() -> None:
    env = _require("HF_TOKEN", "HF_REPO_ID")
    assert env["HF_REPO_ID"] in check_hf(env)


def test_runpod_check_reads_own_pod() -> None:
    assert "API key accepted" in check_runpod(_require("RUNPOD_API_KEY", "RUNPOD_POD_ID"))
