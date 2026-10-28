"""Unit tests for HubPublisher retry, using create_autospec(HfApi) raising the real exception classes.

Exception classes observed against the real Hub / huggingface_hub 2.3.0 (httpx2 transport):
- HTTP errors surface as `huggingface_hub.errors.HfHubHTTPError` (a subclass of `httpx2.HTTPError`
  and `OSError`); the status is `error.response.status_code`. A real 401 (invalid token on
  `create_repo`) raised exactly `HfHubHTTPError`.
- A refused connection (`HfApi(endpoint="http://127.0.0.1:1")`) raised `httpx2.ConnectError`
  (a `httpx2.NetworkError` / `httpx2.TransportError`), unwrapped.

A live 5xx or 429 cannot be provoked on demand, so `_http_error` builds one the way the library does:
`hf_raise_for_status(httpx2.Response(status, request=httpx2.Request(...)))`, which yields the same
`HfHubHTTPError` class and `.response.status_code` a server reply would.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import create_autospec

import httpx2
import pytest
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import hf_raise_for_status

from mprl.errors import UploadFailed
from mprl.hub import HubPublisher

REPO = "someone/mprl-test"
COMMIT_URL = "https://huggingface.co/someone/mprl-test/commit/abc123"


def _http_error(status: int) -> HfHubHTTPError:
    request = httpx2.Request("POST", f"https://huggingface.co/api/models/{REPO}/commit/main")
    try:
        hf_raise_for_status(httpx2.Response(status, request=request))
    except HfHubHTTPError as error:
        return error
    raise AssertionError(f"status {status} did not raise")


def _publisher(api: HfApi, sleeps: list[float], **kwargs: float) -> HubPublisher:
    return HubPublisher(api, REPO, sleep=sleeps.append, **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def api() -> HfApi:
    mock = create_autospec(HfApi, instance=True)
    mock.upload_folder.return_value.commit_url = COMMIT_URL
    return mock


def test_retries_then_succeeds(api: HfApi, tmp_path: Path) -> None:
    api.upload_folder.side_effect = [_http_error(503), _http_error(503), api.upload_folder.return_value]  # type: ignore[attr-defined]
    sleeps: list[float] = []

    url = _publisher(api, sleeps).upload_folder(tmp_path, "ckpt", "msg")

    assert url == COMMIT_URL
    assert sleeps == [2.0, 4.0]
    assert api.upload_folder.call_count == 3  # type: ignore[attr-defined]


def test_retries_then_raises_upload_failed(api: HfApi, tmp_path: Path) -> None:
    api.upload_folder.side_effect = _http_error(503)  # type: ignore[attr-defined]
    sleeps: list[float] = []

    with pytest.raises(UploadFailed) as raised:
        _publisher(api, sleeps, max_attempts=5).upload_folder(tmp_path, "ckpt", "msg")

    assert api.upload_folder.call_count == 5  # type: ignore[attr-defined]
    assert sleeps == [2.0, 4.0, 8.0, 16.0]
    assert isinstance(raised.value.__cause__, HfHubHTTPError)


def test_client_error_not_retried(api: HfApi, tmp_path: Path) -> None:
    api.upload_folder.side_effect = _http_error(401)  # type: ignore[attr-defined]
    sleeps: list[float] = []

    with pytest.raises(HfHubHTTPError):
        _publisher(api, sleeps).upload_folder(tmp_path, "ckpt", "msg")

    assert api.upload_folder.call_count == 1  # type: ignore[attr-defined]
    assert sleeps == []


def test_rate_limit_is_retried(api: HfApi, tmp_path: Path) -> None:
    api.upload_folder.side_effect = [_http_error(429), api.upload_folder.return_value]  # type: ignore[attr-defined]
    sleeps: list[float] = []

    assert _publisher(api, sleeps).upload_folder(tmp_path, "ckpt", "msg") == COMMIT_URL
    assert sleeps == [2.0]


def test_connection_error_is_retried(api: HfApi) -> None:
    api.create_repo.side_effect = [httpx2.ConnectError("connection refused"), None]  # type: ignore[attr-defined]
    sleeps: list[float] = []

    _publisher(api, sleeps).ensure_repo()

    assert api.create_repo.call_count == 2  # type: ignore[attr-defined]
    api.create_repo.assert_called_with(REPO, private=True, repo_type="model", exist_ok=True)  # type: ignore[attr-defined]


def test_upload_forwards_arguments(api: HfApi, tmp_path: Path) -> None:
    _publisher(api, []).upload_folder(tmp_path, "ckpt", "msg")

    api.upload_folder.assert_called_once_with(  # type: ignore[attr-defined]
        repo_id=REPO, folder_path=tmp_path, path_in_repo="ckpt", commit_message="msg", repo_type="model"
    )
