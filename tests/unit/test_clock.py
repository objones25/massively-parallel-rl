from __future__ import annotations

import pytest
import torch

from mprl.clock import TrainingClock, cuda_sync_for
from mprl.negative_space import CheckFailed


def _fake_now(values: list[float]):
    it = iter(values)
    return lambda: next(it)


def test_measure_adds_only_inside_interval():
    now = _fake_now([0.0, 10.0, 15.0, 18.0])
    now()  # the 0.0 read happens outside any measure
    clock = TrainingClock(now=now)
    with clock.measure():
        pass
    assert clock.elapsed_s == pytest.approx(5.0)
    assert clock.elapsed_minutes == pytest.approx(5.0 / 60.0)


def test_sync_called_at_both_ends():
    calls: list[int] = []
    clock = TrainingClock(now=_fake_now([1.0, 2.0]), sync=lambda: calls.append(1))
    with clock.measure():
        assert len(calls) == 1
    assert len(calls) == 2


def test_nested_measure_crashes():
    clock = TrainingClock(now=_fake_now([0.0, 1.0, 2.0, 3.0]))
    with pytest.raises(CheckFailed, match="nested"), clock.measure(), clock.measure():
        pass


def test_backwards_time_crashes():
    clock = TrainingClock(now=_fake_now([5.0, 4.0]))
    with pytest.raises(CheckFailed, match="negative"), clock.measure():
        pass


def test_clock_round_trips_state():
    clock = TrainingClock(now=_fake_now([0.0, 7.5]))
    with clock.measure():
        pass
    restored = TrainingClock.from_state_dict(clock.state_dict())
    assert restored.elapsed_s == pytest.approx(7.5)
    assert clock.state_dict() == {"elapsed_s": 7.5}


def test_resumed_clock_continues():
    clock = TrainingClock.from_state_dict({"elapsed_s": 120.0}, now=_fake_now([0.0, 5.0]))
    with clock.measure():
        pass
    assert clock.elapsed_s == pytest.approx(125.0)


def test_measure_resets_after_exception():
    clock = TrainingClock(now=_fake_now([0.0, 2.0, 3.0, 4.0]))
    with pytest.raises(RuntimeError, match="boom"), clock.measure():
        raise RuntimeError("boom")
    with clock.measure():
        pass
    assert clock.elapsed_s == pytest.approx(3.0)


def test_cuda_sync_for_cpu_is_noop():
    assert cuda_sync_for("cpu")() is None


def test_cuda_sync_for_cuda_is_torch_synchronize():
    assert cuda_sync_for("cuda:0") is torch.cuda.synchronize
