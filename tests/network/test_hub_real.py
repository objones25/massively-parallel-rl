"""Network test: HubPublisher against the real Hugging Face Hub (private throwaway repo)."""

from __future__ import annotations

from pathlib import Path

import pytest
from huggingface_hub import HfApi

from mprl.hub import HubPublisher

pytestmark = pytest.mark.network


def test_ensure_repo_is_idempotent_and_upload_lists_file(tmp_path: Path) -> None:
    api = HfApi()
    user = api.whoami()["name"]
    publisher = HubPublisher(api, f"{user}/mprl-smoke-test", private=True)
    publisher.ensure_repo()
    publisher.ensure_repo()

    (tmp_path / "hello.txt").write_text("hello\n")
    url = publisher.upload_folder(tmp_path, "smoke", "smoke test upload")

    assert url.startswith("https://huggingface.co/")
    assert "smoke/hello.txt" in api.list_repo_files(f"{user}/mprl-smoke-test")
