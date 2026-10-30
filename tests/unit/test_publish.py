"""Unit tests for mprl.publish.upload_run: every upload failure is recorded as text, never raised.

The Hub is a ``create_autospec(HfApi)`` double behind the real ``HubPublisher``. In huggingface_hub 2.3.0
``HfApi.upload_folder`` is ``@validate_hf_hub_args``: a malformed ``repo_id`` (e.g. a bad ``HF_REPO_ID``)
raises ``huggingface_hub.errors.HFValidationError``, a ``ValueError`` subclass (MRO checked), which is not
transient, so ``HubPublisher`` re-raises it at once.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import create_autospec

from huggingface_hub import HfApi
from huggingface_hub.errors import HFValidationError

from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.config import RunSpec
from mprl.hub import HubPublisher
from mprl.paths import RunPaths
from mprl.publish import upload_run


def _finished_run(root: Path) -> tuple[RunPaths, CheckpointRef]:
    paths = RunPaths(root, RunSpec("main", 64, 0, 45.0))
    final = CheckpointStore(paths.checkpoints_dir).save_atomic(lambda p: p.write_bytes(b"w"), 7, 2701.0)
    paths.state_file.write_text("{}")
    return paths, final


def test_upload_run_records_a_hub_validation_error(tmp_path: Path) -> None:
    api = create_autospec(HfApi, instance=True)
    api.upload_folder.side_effect = HFValidationError("Repo id must use alphanumeric chars: 'bad repo'.")
    paths, final = _finished_run(tmp_path)

    outcome = upload_run(HubPublisher(api, "bad repo", sleep=lambda _s: None), paths, final)

    assert outcome == "HFValidationError: Repo id must use alphanumeric chars: 'bad repo'."
    assert api.upload_folder.call_count == 1  # not transient: no retry
    assert not (paths.run_dir / "upload").exists()  # staging removed either way
