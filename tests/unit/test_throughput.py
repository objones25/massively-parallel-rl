"""Unit tests for mprl.throughput.measure_throughput's bookkeeping, with ``measure_count`` doubled.

The real measurement and the timed evaluation on a throughput checkpoint are pinned end to end by
tests/smoke/test_cli.py::test_throughput_writes_measurements_with_timed_eval.

``torch.OutOfMemoryError`` is the class CUDA raises (verified in torch 2.14.1:
``torch.cuda.OutOfMemoryError is torch.OutOfMemoryError``, a ``RuntimeError`` subclass).
"""

from __future__ import annotations

import weakref
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest
import torch

import mprl.throughput as throughput_mod
from mprl.evaluator import EvalConfig
from mprl.negative_space import CheckFailed
from mprl.throughput import CountMeasurement, measure_eval_seconds, measure_throughput


def test_oom_count_recorded_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fits(num_envs: int, *args: Any, **kwargs: Any) -> CountMeasurement:
        return CountMeasurement(steps_per_second=1234.0, io_overhead_fraction=0.05)

    def oom(num_envs: int, *args: Any, **kwargs: Any) -> CountMeasurement:
        raise torch.OutOfMemoryError(f"CUDA out of memory at num_envs={num_envs}. Tried to allocate 2.00 GiB")

    behaviors = iter([fits, oom])  # a measurement for 4, OOM for 8

    def measure(num_envs: int, *args: Any, **kwargs: Any) -> CountMeasurement:
        return next(behaviors)(num_envs, *args, **kwargs)

    double = create_autospec(throughput_mod.measure_count, side_effect=measure)
    monkeypatch.setattr(throughput_mod, "measure_count", double)

    m = measure_throughput((4, 8), 2, "cpu", work_dir=tmp_path, eval_seconds_per_checkpoint=1.5)

    assert m.fits == {4: True, 8: False}
    assert m.steps_per_second == {4: pytest.approx(1234.0)}
    assert m.io_overhead_fraction == pytest.approx(0.05)
    assert m.eval_seconds_per_checkpoint == pytest.approx(1.5)
    assert [c.args[0] for c in double.call_args_list] == [4, 8]
    assert double.call_args.args[3] == tmp_path / "n00008"  # each count in its own scratch dir


def test_non_oom_error_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(num_envs: int, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("not an allocation failure")

    monkeypatch.setattr(throughput_mod, "measure_count", create_autospec(throughput_mod.measure_count, side_effect=broken))

    with pytest.raises(RuntimeError, match="not an allocation failure"):
        measure_throughput((4,), 2, "cpu", work_dir=tmp_path, eval_seconds_per_checkpoint=0.0)


class _Payload:
    """Stands in for the runner and rollout tensors a failing measurement holds."""


def test_memory_released_after_oom_frame_is_gone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    refs: list[weakref.ref[_Payload]] = []
    alive_at_release: list[bool] = []

    def oom(num_envs: int, *args: Any, **kwargs: Any) -> Any:
        payload = _Payload()  # a local of the failing frame, pinned by the traceback while it lives
        refs.append(weakref.ref(payload))
        raise torch.OutOfMemoryError(f"CUDA out of memory at num_envs={num_envs}")

    monkeypatch.setattr(throughput_mod, "measure_count", create_autospec(throughput_mod.measure_count, side_effect=oom))
    monkeypatch.setattr(
        throughput_mod, "release_device_memory", lambda _device: alive_at_release.append(refs[-1]() is not None)
    )

    m = measure_throughput((4, 8), 2, "cpu", work_dir=tmp_path, eval_seconds_per_checkpoint=0.0)

    assert m.fits == {4: False, 8: False}
    assert alive_at_release == [False, False]


def test_eval_seconds_needs_a_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(CheckFailed, match=r"^no throughput checkpoint for num_envs=4 under "):
        measure_eval_seconds(tmp_path, 4, "cpu", EvalConfig())
