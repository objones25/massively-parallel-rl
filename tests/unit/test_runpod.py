"""Unit tests for mprl.runpod, using a real httpx.Client over httpx.MockTransport (httpx 0.28.1).

`MockTransport(handler)` takes a callable `(httpx.Request) -> httpx.Response` and may raise
`httpx.TransportError` subclasses such as `httpx.ConnectError` to simulate a dead connection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest

from mprl.errors import OwnPodUnknown, TerminateFailed
from mprl.runpod import RUNPOD_REST_BASE, PodTerminator, own_pod_id

POD = "abc123podid"
KEY = "rpa_secret_key_value"
BASE = "https://api.test/v2"


@dataclass
class Script:
    """A scripted RunPod: replies per method (the last repeats), records every request."""

    delete: list[httpx.Response | Exception]
    get: list[httpx.Response | Exception]
    seen: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        queue = self.delete if request.method == "DELETE" else self.get
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    @property
    def methods(self) -> list[str]:
        return [request.method for request in self.seen]


def _terminator(script: Script, sleeps: list[float], attempts: int = 3) -> PodTerminator:
    client = httpx.Client(transport=httpx.MockTransport(script))
    return PodTerminator(client, KEY, POD, base_url=BASE, max_attempts=attempts, sleep=sleeps.append)


def _reply(status: int, text: str = "") -> httpx.Response:
    return httpx.Response(status, text=text)


def _status(name: str) -> httpx.Response:
    return httpx.Response(200, json={"id": POD, "status": name})


def _refused() -> httpx.ConnectError:
    return httpx.ConnectError("refused", request=httpx.Request("DELETE", BASE))


def test_pod_id_from_env(tmp_path: Path) -> None:
    assert own_pod_id({"RUNPOD_POD_ID": POD}, tmp_path / "missing") == POD


def test_pod_id_from_proc1_when_env_missing(tmp_path: Path) -> None:
    environ = tmp_path / "environ"
    environ.write_bytes(b"PATH=/usr/bin\x00RUNPOD_POD_ID=" + POD.encode() + b"\x00HOME=/root\x00")
    assert own_pod_id({}, environ) == POD


def test_pod_id_missing_raises(tmp_path: Path) -> None:
    environ = tmp_path / "environ"
    environ.write_bytes(b"PATH=/usr/bin\x00")
    with pytest.raises(OwnPodUnknown, match="RUNPOD_POD_ID"):
        own_pod_id({}, environ)
    with pytest.raises(OwnPodUnknown, match="RUNPOD_POD_ID"):
        own_pod_id({}, tmp_path / "does-not-exist")


def test_rest_base_is_v2() -> None:
    assert RUNPOD_REST_BASE == "https://api.runpod.io/v2"


def test_terminate_confirms_with_read() -> None:
    script = Script([_reply(204)], [_status("TERMINATED")])

    _terminator(script, []).terminate()

    bearer = f"Bearer {KEY}"
    assert [(r.method, r.url.path, r.headers["authorization"]) for r in script.seen] == [
        ("DELETE", f"/v2/pods/{POD}", bearer),
        ("GET", f"/v2/pods/{POD}", bearer),
    ]


def test_terminate_accepts_404_on_read() -> None:
    script = Script([_reply(204)], [_reply(404)])
    _terminator(script, []).terminate()
    assert script.methods == ["DELETE", "GET"]


def test_terminate_polls_until_terminated() -> None:
    script = Script([_reply(204)], [_status("RUNNING"), _status("EXITED"), _status("TERMINATED")])
    sleeps: list[float] = []
    _terminator(script, sleeps, attempts=5).terminate()
    assert (script.methods, sleeps) == (["DELETE", "GET", "GET", "GET"], [2.0, 4.0])


def test_terminate_retries_5xx_then_raises() -> None:
    script = Script([_reply(503, text="upstream down")], [_reply(404)])
    sleeps: list[float] = []
    with pytest.raises(TerminateFailed, match="503") as info:
        _terminator(script, sleeps).terminate()

    assert script.methods == ["DELETE"] * 3
    assert sleeps == [2.0, 4.0]
    assert KEY not in str(info.value)


def test_terminate_retries_connection_error_then_succeeds() -> None:
    script = Script([_refused(), _reply(204)], [_reply(429), _reply(404)])
    _terminator(script, []).terminate()
    assert script.methods == ["DELETE", "DELETE", "GET", "GET"]


@pytest.mark.parametrize("status", [401, 403, 409])
def test_terminate_does_not_retry_client_errors(status: int) -> None:
    script = Script([_reply(status, text="nope")], [_reply(404)])
    with pytest.raises(TerminateFailed, match=str(status)):
        _terminator(script, []).terminate()
    assert script.methods == ["DELETE"]


def test_delete_404_succeeds_when_read_also_404() -> None:
    script = Script([_reply(404)], [_reply(404)])
    _terminator(script, []).terminate()
    assert script.methods == ["DELETE", "GET"]


def test_delete_404_fails_when_pod_still_exists() -> None:
    script = Script([_reply(404)], [_status("RUNNING")])
    with pytest.raises(TerminateFailed, match="RUNNING"):
        _terminator(script, []).terminate()
    assert script.methods == ["DELETE", "GET"]


def test_never_confirmed_terminated_raises() -> None:
    script = Script([_reply(204)], [_status("RUNNING")])
    with pytest.raises(TerminateFailed, match="RUNNING"):
        _terminator(script, []).terminate()
    assert script.methods == ["DELETE", "GET", "GET", "GET"]
