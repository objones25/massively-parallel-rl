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


def _read(client: httpx.Client) -> httpx.Response:
    return client.get(f"{RUNPOD_REST_BASE}/pods/{_POD_ID}", headers={"Authorization": f"Bearer {_API_KEY}"})


def _read_before_terminate_then_read() -> tuple[httpx.Response, httpx.Response]:
    with httpx.Client(timeout=30.0) as client:
        before = _read(client)
        if before.status_code != httpx.codes.OK or before.json()["status"] == "TERMINATED":
            return before, before
        PodTerminator(client, _API_KEY, _POD_ID, base_url=RUNPOD_REST_BASE).terminate()
        return before, _read(client)


@pytest.mark.skipif(
    not (_POD_ID and _API_KEY),
    reason="needs MPRL_THROWAWAY_POD_ID (a throwaway pod you created) and RUNPOD_API_KEY",
)
def test_terminate_real_pod() -> None:
    before, read = _read_before_terminate_then_read()
    # A mistyped, stale or already-terminated id would pass vacuously (404 before and after).
    assert before.status_code == httpx.codes.OK, f"pod {_POD_ID} not found before terminating: check the id"
    assert before.json()["status"] != "TERMINATED", f"pod {_POD_ID} was already terminated"
    assert read.status_code == httpx.codes.NOT_FOUND or read.json()["status"] == "TERMINATED"
