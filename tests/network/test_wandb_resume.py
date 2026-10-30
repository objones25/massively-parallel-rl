"""Network tests against real W&B: resuming a finished run and the offline fallback. Need WANDB_API_KEY.

These pin server behavior a test double cannot: on 2026-10-29 ``wandb.init(resume_from=...)`` passed every
autospecced unit test but the server answered 400 (Rewind is a private preview), and the offline retry then
failed with "run ID ... is in use". Run this tier before every paid session (docs/runbook.md).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
import wandb

from mprl.wandb_writer import CLOCK_METRIC, ResumableWandbWriter

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(not os.environ.get("WANDB_API_KEY"), reason="WANDB_API_KEY not set; real W&B needed"),
]

PROJECT = "mprl-tests"
TAG = "Test/value"


@pytest.fixture
def wandb_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    return tmp_path


def test_resume_continues_a_finished_run_after_its_last_step(wandb_dir: Path) -> None:
    run_id = f"resume-{uuid.uuid4().hex[:8]}"
    first = ResumableWandbWriter(str(wandb_dir), PROJECT, run_id, "resume-test")
    for step in range(6):
        first.add_scalar(TAG, float(step), step)
    first.stop()

    second = ResumableWandbWriter(str(wandb_dir), PROJECT, run_id, "resume-test", resume_from_iteration=5)
    second.add_scalar(TAG, 99.0, 6)
    second.stop()

    api = wandb.Api()
    run = api.run(f"{api.default_entity}/{PROJECT}/{run_id}")
    history = {int(row["_step"]): row[TAG] for row in run.scan_history(keys=["_step", TAG])}
    assert (second.offline_fallback, sorted(history), history[6]) == (None, list(range(7)), pytest.approx(99.0))


def test_resume_after_a_crash_between_checkpoints_keeps_logging_past_the_crash(wandb_dir: Path) -> None:
    """History holds steps 0..5, the checkpoint is at 3: W&B ignores the resumed 4 and 5 (first values kept) and
    takes 6. The clock metric (the Train/* x-axis) follows the same rule."""
    run_id = f"crash-{uuid.uuid4().hex[:8]}"
    first = ResumableWandbWriter(str(wandb_dir), PROJECT, run_id, "resume-test")
    for step in range(6):
        first.add_scalar(TAG, float(step), step)
        first.add_scalar(CLOCK_METRIC, float(step), step)
    first.stop()

    second = ResumableWandbWriter(str(wandb_dir), PROJECT, run_id, "resume-test", resume_from_iteration=3)
    for step in range(4, 7):  # iterations 4 and 5 were logged before the crash
        second.add_scalar(TAG, 100.0 + step, step)
        second.add_scalar(CLOCK_METRIC, 3.0 + step / 10, step)
    second.stop()

    api = wandb.Api()
    run = api.run(f"{api.default_entity}/{PROJECT}/{run_id}")
    rows = {int(r["_step"]): (r[TAG], r[CLOCK_METRIC]) for r in run.scan_history(keys=["_step", TAG, CLOCK_METRIC])}
    assert rows == {
        **{step: (pytest.approx(float(step)), pytest.approx(float(step))) for step in range(6)},
        6: (pytest.approx(106.0), pytest.approx(3.6)),
    }


def test_failed_online_init_falls_back_to_a_working_offline_run(wandb_dir: Path) -> None:
    """A nonexistent entity fails the real run upsert with CommError after wandb's service has claimed the id."""
    run_id = f"fallback-{uuid.uuid4().hex[:8]}"
    writer = ResumableWandbWriter(
        str(wandb_dir), PROJECT, run_id, "fallback-test", entity=f"no-such-entity-{uuid.uuid4().hex[:8]}"
    )
    writer.add_scalar(TAG, 1.0, 0)
    writer.stop()

    assert writer.offline_fallback is not None
    assert writer.offline_fallback.startswith("CommError: ")
    assert len(list(wandb_dir.glob(f"wandb/offline-run-*-{run_id}"))) == 1
