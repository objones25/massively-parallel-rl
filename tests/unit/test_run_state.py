from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from mprl.negative_space import CheckFailed
from mprl.run_state import RunState, load_state, save_state


def _state(**overrides: object) -> RunState:
    base = RunState(
        run_id="main-n00064-s0",
        config_hash="ab" * 32,
        wandb_run_id="main-n00064-s0-1a2b3c",
        status="running",
        train_seconds=12.5,
        budget_minutes=45.0,
        git_sha="0123abcd",
        uploads={},
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def test_missing_file_loads_as_none(tmp_path: Path) -> None:
    assert load_state(tmp_path / "state.json") is None


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "run" / "state.json"
    state = _state(status="complete", uploads={"runs/main/main-n00064-s0": "UploadFailed: 503"})

    save_state(path, state)

    assert load_state(path) == state
    assert json.loads(path.read_text()) == asdict(state)


def test_save_overwrites_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    save_state(path, _state(train_seconds=1.0))
    save_state(path, _state(train_seconds=2.0))

    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
    loaded = load_state(path)
    assert loaded is not None
    assert loaded.train_seconds == pytest.approx(2.0)


def test_partial_temp_file_is_ignored(tmp_path: Path) -> None:
    """A write killed before os.replace leaves only the temp file; the last complete state wins."""
    path = tmp_path / "state.json"
    save_state(path, _state(train_seconds=3.0))
    (tmp_path / ".state.json.tmp").write_text('{"run_id": "main-n00064-s0", "stat')

    loaded = load_state(path)
    assert loaded is not None
    assert loaded.train_seconds == pytest.approx(3.0)


def test_status_is_validated() -> None:
    with pytest.raises(CheckFailed, match="status"):
        _state(status="done")


def test_status_is_validated_on_load(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps(asdict(_state()) | {"status": "finished"}))
    with pytest.raises(CheckFailed, match="finished"):
        load_state(path)


def test_negative_train_seconds_rejected() -> None:
    with pytest.raises(CheckFailed, match="train_seconds"):
        _state(train_seconds=-1.0)
