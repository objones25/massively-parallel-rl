"""Network test: resuming from an iteration truncates W&B history after that step. Needs WANDB_API_KEY."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
import wandb

from mprl.wandb_writer import ResumableWandbWriter

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(not os.environ.get("WANDB_API_KEY"), reason="WANDB_API_KEY not set; real W&B needed"),
]

PROJECT = "mprl-tests"
TAG = "Test/value"


def test_resume_from_truncates_history(tmp_path: Path) -> None:
    run_id = f"resume-{uuid.uuid4().hex[:8]}"
    first = ResumableWandbWriter(str(tmp_path), PROJECT, run_id, "resume-test")
    for step in range(6):
        first.add_scalar(TAG, float(step), step)
    first.stop()

    second = ResumableWandbWriter(str(tmp_path), PROJECT, run_id, "resume-test", resume_from_iteration=3)
    second.add_scalar(TAG, 99.0, 4)
    second.stop()

    api = wandb.Api()
    run = api.run(f"{api.default_entity}/{PROJECT}/{run_id}")
    history = {int(row["_step"]): row[TAG] for row in run.scan_history(keys=["_step", TAG])}
    assert history[4] == pytest.approx(99.0)
    assert 5 not in history
