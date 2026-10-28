from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mprl.checkpoints import CheckpointRef, CheckpointStore
from mprl.negative_space import CheckFailed


def _write_ok(path: Path) -> None:
    path.write_bytes(b"weights")


def test_save_atomic_returns_ref_with_minutes(tmp_path: Path):
    store = CheckpointStore(tmp_path)
    ref = store.save_atomic(_write_ok, 3, 180.0)
    assert ref == CheckpointRef(3, 180.0, tmp_path / "model_it0000003_s000180.pt")
    assert ref.path.read_bytes() == b"weights"
    assert ref.train_seconds / 60.0 == pytest.approx(3.0)


def test_write_receives_temp_path_in_same_directory(tmp_path: Path):
    seen: list[Path] = []

    def write(path: Path) -> None:
        seen.append(path)
        path.write_bytes(b"x")

    store = CheckpointStore(tmp_path)
    ref = store.save_atomic(write, 1, 10.0)
    assert seen == [tmp_path / ".model_it0000001_s000010.pt.tmp"]
    assert not seen[0].exists()
    assert ref.path.exists()


def test_list_sorted_by_iteration(tmp_path: Path):
    store = CheckpointStore(tmp_path)
    store.save_atomic(_write_ok, 9, 540.0)
    store.save_atomic(_write_ok, 3, 180.0)
    store.save_atomic(_write_ok, 6, 360.0)
    assert [r.iteration for r in store.list()] == [3, 6, 9]
    latest = store.latest()
    assert latest is not None
    assert latest.iteration == 9


def test_list_breaks_iteration_ties_by_train_seconds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = CheckpointStore(tmp_path)
    store.save_atomic(_write_ok, 5, 10.0)
    store.save_atomic(_write_ok, 5, 20.0)
    real_iterdir = Path.iterdir
    # Directory order is not guaranteed; present the newer file first so only the sort key decides.
    monkeypatch.setattr(Path, "iterdir", lambda self: iter(sorted(real_iterdir(self), reverse=True)))

    assert [(r.iteration, r.train_seconds) for r in store.list()] == [(5, 10.0), (5, 20.0)]
    latest = store.latest()
    assert latest is not None
    assert latest.train_seconds == pytest.approx(20.0)


def test_empty_store(tmp_path: Path):
    store = CheckpointStore(tmp_path / "missing")
    assert store.list() == []
    assert store.latest() is None
    assert store.at_or_before(10.0) is None


def test_partial_temp_file_is_ignored(tmp_path: Path):
    store = CheckpointStore(tmp_path)
    store.save_atomic(_write_ok, 1, 60.0)
    (tmp_path / ".model_it0000003_s000180.pt.tmp").write_bytes(b"partial")
    (tmp_path / "model_it0000004_s000240.pt.tmp").write_bytes(b"partial")
    (tmp_path / "model_it4_s240.pt").write_bytes(b"badname")
    assert [r.iteration for r in store.list()] == [1]


def test_failed_write_leaves_no_file(tmp_path: Path):
    store = CheckpointStore(tmp_path)

    def write(path: Path) -> None:
        path.write_bytes(b"half")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        store.save_atomic(write, 2, 120.0)
    assert list(tmp_path.iterdir()) == []


def test_at_or_before_picks_latest_not_after(tmp_path: Path):
    store = CheckpointStore(tmp_path)
    store.save_atomic(_write_ok, 3, 180.0)
    store.save_atomic(_write_ok, 6, 360.0)
    store.save_atomic(_write_ok, 9, 540.0)
    ref = store.at_or_before(7.0)
    assert ref is not None
    assert ref.train_seconds == pytest.approx(360.0)
    assert store.at_or_before(2.0) is None


@given(
    seconds=st.sets(st.integers(min_value=0, max_value=100_000), max_size=8),
    minutes=st.floats(min_value=0.0, max_value=2_000.0),
)
def test_at_or_before_never_exceeds_query(seconds: set[int], minutes: float):
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        store = CheckpointStore(Path(d))
        refs = [store.save_atomic(_write_ok, s, float(s)) for s in seconds]
        assert len(refs) == len(seconds)
        ref = store.at_or_before(minutes)
        expected = max((x for x in seconds if x <= minutes * 60.0), default=None)
        assert getattr(ref, "train_seconds", None) == expected


def test_largest_representable_seconds_is_saved_and_listed(tmp_path: Path):
    store = CheckpointStore(tmp_path)
    ref = store.save_atomic(_write_ok, 1, 999_999.4)
    assert store.list() == [ref]


def test_seconds_that_round_past_six_digits_are_rejected(tmp_path: Path):
    store = CheckpointStore(tmp_path)
    with pytest.raises(CheckFailed, match="train_seconds"):
        store.save_atomic(_write_ok, 1, 999_999.5)
    assert list(tmp_path.iterdir()) == []
