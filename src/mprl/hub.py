"""Hugging Face publisher with bounded retry.

Retries only what is transient: server 5xx / 429 (`HfHubHTTPError`) and connection errors
(`httpx2.TransportError`, e.g. `ConnectError`). Any other error, including 4xx, propagates at once.

`UPLOAD_ERRORS` is what a caller records instead of raising: `UploadFailed` (retries exhausted),
`HfHubHTTPError` (a 4xx; also an `OSError`), `OSError` (local I/O, staging) and `HFValidationError` (a
malformed repo id from `@validate_hf_hub_args`; a `ValueError` subclass in huggingface_hub 2.3.0).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Final, TypeVar

import httpx2
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, HFValidationError

from mprl.errors import UploadFailed
from mprl.negative_space import bounded, require

T = TypeVar("T")

UPLOAD_ERRORS: Final = (UploadFailed, HfHubHTTPError, OSError, HFValidationError)

_TOO_MANY_REQUESTS = 429
_SERVER_ERROR_FLOOR = 500


def _is_transient(error: Exception) -> bool:
    if isinstance(error, httpx2.TransportError):
        return True
    if isinstance(error, HfHubHTTPError):
        status = error.response.status_code
        return status == _TOO_MANY_REQUESTS or status >= _SERVER_ERROR_FLOOR
    return False


class HubPublisher:
    def __init__(
        self,
        api: HfApi,
        repo_id: str,
        *,
        private: bool = True,
        max_attempts: int = 5,
        base_delay_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        require(max_attempts >= 1, f"max_attempts must be at least 1, got {max_attempts}")
        require(base_delay_s >= 0, f"base_delay_s must be non-negative, got {base_delay_s}")
        self._api = api
        self._repo_id = repo_id
        self._private = private
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._sleep = sleep

    def ensure_repo(self) -> None:
        self._with_retry(
            "create_repo",
            lambda: self._api.create_repo(self._repo_id, private=self._private, repo_type="model", exist_ok=True),
        )

    def upload_folder(self, local: Path, path_in_repo: str, message: str, *, delete_patterns: str | None = None) -> str:
        """Upload `local` to `path_in_repo`; return the commit URL.

        `delete_patterns` (relative to `path_in_repo`) deletes the matching remote files that this commit does not
        upload again, so the remote folder mirrors `local` for those files.
        """
        require(local.is_dir(), f"upload source is not a directory: {local}")
        info = self._with_retry(
            "upload_folder",
            lambda: self._api.upload_folder(
                repo_id=self._repo_id,
                folder_path=local,
                path_in_repo=path_in_repo,
                commit_message=message,
                repo_type="model",
                delete_patterns=delete_patterns,
            ),
        )
        return str(info.commit_url)

    def _with_retry(self, what: str, call: Callable[[], T]) -> T:
        last: Exception | None = None
        for attempt in bounded(range(self._max_attempts), self._max_attempts, f"{what} attempts"):
            if attempt > 0:
                self._sleep(self._base_delay_s * 2 ** (attempt - 1))
            try:
                return call()
            except Exception as error:
                if not _is_transient(error):
                    raise
                last = error
        raise UploadFailed(f"{what} to {self._repo_id} failed after {self._max_attempts} attempts: {last}") from last
