"""Smoke test for mprl.throughput.measure_throughput against real mjlab on CPU.

``torch.OutOfMemoryError`` is the class CUDA raises (verified in torch 2.14.1:
``torch.cuda.OutOfMemoryError is torch.OutOfMemoryError``, a ``RuntimeError`` subclass).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

import mprl.throughput as throughput_mod
from mprl.throughput import measure_throughput

pytestmark = pytest.mark.smoke


def test_oom_count_recorded_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def oom(num_envs: int, *args: Any, **kwargs: Any) -> Any:
        raise torch.OutOfMemoryError(f"CUDA out of memory at num_envs={num_envs}. Tried to allocate 2.00 GiB")

    behaviors = iter([throughput_mod.measure_count, oom])  # the real measurement for 4, OOM for 8
    seen: list[int] = []

    def measure(num_envs: int, *args: Any, **kwargs: Any) -> Any:
        seen.append(num_envs)
        return next(behaviors)(num_envs, *args, **kwargs)

    monkeypatch.setattr(throughput_mod, "measure_count", measure)

    m = measure_throughput((4, 8), 2, "cpu", work_dir=tmp_path, eval_seconds_per_checkpoint=1.5)

    assert m.fits == {4: True, 8: False}
    assert m.steps_per_second[4] > 0
    assert 8 not in m.steps_per_second
    assert m.io_overhead_fraction > 0
    assert m.eval_seconds_per_checkpoint == pytest.approx(1.5)
    assert seen == [4, 8]


def test_non_oom_error_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(num_envs: int, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("not an allocation failure")

    monkeypatch.setattr(throughput_mod, "measure_count", broken)

    with pytest.raises(RuntimeError, match="not an allocation failure"):
        measure_throughput((4,), 2, "cpu", work_dir=tmp_path, eval_seconds_per_checkpoint=0.0)
