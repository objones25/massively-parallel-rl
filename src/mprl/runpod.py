"""RunPod self-termination: find our own pod id, delete the pod, confirm with a read.

The network volume (all results) survives pod deletion. Callers terminate only after uploads
finished or were recorded as failed.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import httpx

from mprl.errors import OwnPodUnknown, TerminateFailed
from mprl.negative_space import bounded, require

# Verified 2026-10-27 against RunPod's REST v2 docs:
#   DELETE {base}/pods/{id}: 204 no body; 401/403/404/409/429 errors
#     https://docs.runpod.io/api-reference-v2/pods/terminate-a-pod
#   GET {base}/pods/{id}: 200 JSON with `status` (PROVISIONING, STARTING, RUNNING, EXITED, ERROR,
#     TERMINATED); a terminated pod is gone from listPods, so 404 is expected
#     https://docs.runpod.io/api-reference-v2/pods/get-a-pod
# Auth is `Authorization: Bearer <key>` only.
RUNPOD_REST_BASE: Final[str] = "https://api.runpod.io/v2"

_POD_ID_VAR = "RUNPOD_POD_ID"
_POD_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_OK = 200
_NO_CONTENT = 204
_NOT_FOUND = 404
_TOO_MANY_REQUESTS = 429
_SERVER_ERROR_FLOOR = 500
_EXCERPT_CHARS = 200


def own_pod_id(environ: Mapping[str, str], proc1_environ: Path = Path("/proc/1/environ")) -> str:
    """The id of the pod we run on.

    RunPod sets RUNPOD_POD_ID for the container's PID 1 only; SSH login shells do not get it, so
    fall back to the NUL-separated KEY=VALUE entries of /proc/1/environ.
    """
    value = environ.get(_POD_ID_VAR)
    if value:
        return value
    try:
        raw = proc1_environ.read_bytes()
    except OSError as error:
        raise OwnPodUnknown(f"{_POD_ID_VAR} is not set and {proc1_environ} is unreadable: {error}") from error
    prefix = f"{_POD_ID_VAR}=".encode()
    for entry in raw.split(b"\x00"):
        if entry.startswith(prefix) and len(entry) > len(prefix):
            return entry[len(prefix) :].decode()
    raise OwnPodUnknown(f"{_POD_ID_VAR} is not set in the environment or in {proc1_environ}")


class PodTerminator:
    def __init__(
        self,
        client: httpx.Client,
        api_key: str,
        pod_id: str,
        *,
        base_url: str,
        max_attempts: int = 5,
        base_delay_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        require(bool(api_key), "api_key must not be empty")
        require(_POD_ID_PATTERN.fullmatch(pod_id) is not None, f"pod_id must match [A-Za-z0-9_-]+, got {pod_id!r}")
        require(max_attempts >= 1, f"max_attempts must be at least 1, got {max_attempts}")
        require(base_delay_s >= 0, f"base_delay_s must be non-negative, got {base_delay_s}")
        self._client = client
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._pod_id = pod_id
        self._url = f"{base_url.rstrip('/')}/pods/{pod_id}"
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._sleep = sleep

    def terminate(self) -> None:
        """Read the pod, delete it, then confirm with a read that it is gone (404 or status TERMINATED).

        The pre-read makes a mis-resolved pod id loud: 404 there means we never saw the pod, so a later
        404 would prove nothing. An already TERMINATED pod is success (idempotent) and sends no DELETE.
        """
        before = self._read_status()
        if before is None:
            raise TerminateFailed(f"pod {self._pod_id} not found — refusing to report success")
        if before == "TERMINATED":
            return
        deleted = self._send("DELETE")
        if deleted.status_code == _NOT_FOUND:
            # Possibly already terminated; only the read can tell.
            status = self._read_status()
            if status is not None and status != "TERMINATED":
                raise TerminateFailed(f"DELETE of pod {self._pod_id} returned 404 but the pod is {status}")
            return
        if deleted.status_code not in (_OK, _NO_CONTENT):
            raise self._failure("DELETE", deleted)
        status = None
        for attempt in bounded(range(self._max_attempts), self._max_attempts, "termination confirmation"):
            status = self._read_status()
            if status is None or status == "TERMINATED":
                return
            if attempt < self._max_attempts - 1:
                self._sleep(self._delay(attempt + 1))
        raise TerminateFailed(f"pod {self._pod_id} still {status} after {self._max_attempts} reads following DELETE")

    def _delay(self, retry: int) -> float:
        return self._base_delay_s * 2 ** (retry - 1)

    def _read_status(self) -> str | None:
        """The pod's status, or None when the API says it no longer exists (404)."""
        response = self._send("GET")
        if response.status_code == _NOT_FOUND:
            return None
        if response.status_code != _OK:
            raise self._failure("GET", response)
        try:
            return str(response.json()["status"])
        except (ValueError, KeyError, TypeError) as error:
            raise TerminateFailed(
                f"GET of pod {self._pod_id} returned no status: {response.text[:_EXCERPT_CHARS]}"
            ) from error

    def _failure(self, method: str, response: httpx.Response) -> TerminateFailed:
        return TerminateFailed(
            f"{method} of pod {self._pod_id} failed with HTTP {response.status_code}: {response.text[:_EXCERPT_CHARS]}"
        )

    def _send(self, method: str) -> httpx.Response:
        """One request, retrying 5xx, 429 and transport errors; returns the first non-transient reply."""
        last = "no attempt"
        for attempt in bounded(range(self._max_attempts), self._max_attempts, f"{method} attempts"):
            if attempt > 0:
                self._sleep(self._delay(attempt))
            try:
                response = self._client.request(method, self._url, headers=self._headers)
            except httpx.TransportError as error:
                last = f"{type(error).__name__}: {error}"
                continue
            if response.status_code != _TOO_MANY_REQUESTS and response.status_code < _SERVER_ERROR_FLOOR:
                return response
            last = f"HTTP {response.status_code}: {response.text[:_EXCERPT_CHARS]}"
        raise TerminateFailed(f"{method} of pod {self._pod_id} failed after {self._max_attempts} attempts: {last}")
