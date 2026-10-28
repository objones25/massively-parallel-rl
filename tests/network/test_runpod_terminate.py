"""Network test: terminate a throwaway RunPod pod for real and confirm with a read.

Run only by the user, once, against a pod they created for this purpose (cheapest Secure Cloud GPU).
The only pod this test ever touches is the one named in MPRL_THROWAWAY_POD_ID. See CLAUDE.md.
"""

from __future__ import annotations

import os

import httpx
import pytest

from mprl.runpod import RUNPOD_REST_BASE, PodTerminator

pytestmark = pytest.mark.network

_POD_ID = os.environ.get("MPRL_THROWAWAY_POD_ID", "")
_API_KEY = os.environ.get("RUNPOD_API_KEY", "")


def _terminate_then_read() -> httpx.Response:
    with httpx.Client(timeout=30.0) as client:
        PodTerminator(client, _API_KEY, _POD_ID, base_url=RUNPOD_REST_BASE).terminate()
        return client.get(f"{RUNPOD_REST_BASE}/pods/{_POD_ID}", headers={"Authorization": f"Bearer {_API_KEY}"})


@pytest.mark.skipif(
    not (_POD_ID and _API_KEY),
    reason="needs MPRL_THROWAWAY_POD_ID (a throwaway pod you created) and RUNPOD_API_KEY",
)
def test_terminate_real_pod() -> None:
    read = _terminate_then_read()
    assert read.status_code == httpx.codes.NOT_FOUND or read.json()["status"] == "TERMINATED"
