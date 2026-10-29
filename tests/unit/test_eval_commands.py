from __future__ import annotations

import math
from pathlib import Path

import pytest

from mprl.evaluator import EVAL_COMMANDS, EvalConfig, EvalResult, command_tensor, read_eval_csv, write_eval_csv

STAGE0_LIN_X = 1.0  # curriculum command_vel stage 0: lin_vel_x (-1.0, 1.0), ang_vel_z (-0.5, 0.5)
STAGE0_LIN_Y = 1.0  # lin_vel_y is not staged; its range is (-1.0, 1.0) throughout
STAGE0_ANG_Z = 0.5


@pytest.mark.parametrize("command", EVAL_COMMANDS)
def test_eval_commands_inside_stage0_ranges(command: tuple[float, float, float]) -> None:
    lin_x, lin_y, ang_z = command
    assert abs(lin_x) <= STAGE0_LIN_X
    assert abs(lin_y) <= STAGE0_LIN_Y
    assert abs(ang_z) <= STAGE0_ANG_Z


def test_eval_commands_are_the_specified_six() -> None:
    assert EVAL_COMMANDS == (
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.0, -0.5),
    )
    assert EvalConfig() == EvalConfig(EVAL_COMMANDS, (0, 1), 4, 10.0)


def test_command_tensor_repeats_each_command_per_env_block() -> None:
    cfg = EvalConfig(commands=((0.5, 0.0, 0.0), (0.0, 0.0, -0.5)), envs_per_command=2)

    rows = command_tensor(cfg, "cpu").tolist()

    assert rows == [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, -0.5], [0.0, 0.0, -0.5]]


def test_eval_csv_round_trip(tmp_path: Path) -> None:
    results = [
        EvalResult("main-n00064-s0", 12, 3.0, 0.1234567890123, 0.5, 0.75, 7.25, 0.875),
        EvalResult("main-n04096-s2", 1, 0.0166, 1.0 / 3.0, 2.0e-9, 0.0, 10.0, 0.0),
    ]
    path = tmp_path / "eval.csv"

    write_eval_csv(results, path)

    assert read_eval_csv(path) == results
    assert path.read_text().splitlines()[0] == (
        "run_id,iteration,train_minutes,vel_error_xy,vel_error_yaw,no_fall_fraction,"
        "mean_time_to_fall_s,mean_tracking_reward"
    )


def test_eval_csv_round_trips_nan(tmp_path: Path) -> None:
    path = tmp_path / "eval.csv"

    write_eval_csv([EvalResult("r", 0, 0.0, math.nan, math.nan, 0.0, 0.0, math.nan)], path)

    (back,) = read_eval_csv(path)
    assert math.isnan(back.vel_error_xy)
    assert math.isnan(back.vel_error_yaw)
    assert math.isnan(back.mean_tracking_reward)
    assert (back.run_id, back.iteration, back.no_fall_fraction) == ("r", 0, 0.0)
