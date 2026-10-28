import pytest

from mprl.config import TASK_ID, RunSpec, SweepSpec, batch_arithmetic


def test_runs_cover_every_count_and_seed() -> None:
    runs = SweepSpec("main").runs()
    assert len(runs) == 15
    assert (runs[0].num_envs, runs[0].seed) == (64, 0)
    assert (runs[-1].num_envs, runs[-1].seed) == (16384, 2)


def test_runs_carry_sweep_settings() -> None:
    run = SweepSpec("main", budget_minutes=10.0, checkpoint_every_minutes=2.0).runs()[0]
    assert (run.sweep, run.budget_minutes, run.checkpoint_every_minutes, run.task_id) == (
        "main",
        10.0,
        2.0,
        TASK_ID,
    )


def test_long_run_uses_long_settings() -> None:
    run = SweepSpec("main").long_run()
    assert run.sweep == "main-long"
    assert (run.num_envs, run.seed, run.budget_minutes) == (4096, 0, 180.0)


def test_run_id_is_zero_padded() -> None:
    assert RunSpec("main", 64, 1, 45.0).run_id == "main-n00064-s1"


def test_config_hash_ignores_budget() -> None:
    assert RunSpec("main", 64, 0, 45.0).config_hash() == RunSpec("main", 64, 0, 180.0).config_hash()


def test_config_hash_changes_with_num_envs() -> None:
    assert RunSpec("main", 64, 0, 45.0).config_hash() != RunSpec("main", 256, 0, 45.0).config_hash()


def test_run_spec_rejects_bad_values() -> None:
    with pytest.raises(AssertionError):
        RunSpec("main", 0, 0, 45.0)
    with pytest.raises(AssertionError):
        RunSpec("main", 64, 0, 0.0)
    with pytest.raises(AssertionError):
        RunSpec("main", 64, 0, 2.0, checkpoint_every_minutes=3.0)
    with pytest.raises(AssertionError):
        RunSpec("main", 64, 0, 45.0, checkpoint_every_minutes=0.0)


def test_batch_arithmetic_64_envs() -> None:
    b = batch_arithmetic(64)
    assert (b.batch_size, b.minibatch_size) == (1536, 384)


def test_batch_arithmetic_16384_envs() -> None:
    b = batch_arithmetic(16384)
    assert (b.batch_size, b.minibatch_size) == (393216, 98304)


def test_batch_arithmetic_rejects_indivisible() -> None:
    with pytest.raises(ValueError, match="divisible"):
        batch_arithmetic(10, 3, 4)


def test_batch_arithmetic_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        batch_arithmetic(0)
