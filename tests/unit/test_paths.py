from pathlib import Path

from mprl.config import RunSpec
from mprl.paths import RunPaths, sweep_results_dir, volume_root


def test_run_paths_layout() -> None:
    root = Path("/vol")
    paths = RunPaths(root, RunSpec("main", 64, 1, 45.0))
    run_dir = root / "sweeps" / "main" / "main-n00064-s1"
    assert paths.run_dir == run_dir
    assert paths.checkpoints_dir == run_dir / "checkpoints"
    assert paths.state_file == run_dir / "state.json"
    assert paths.eval_dir == run_dir / "eval"
    assert paths.videos_dir == run_dir / "videos"
    assert paths.tensorboard_dir == run_dir / "tensorboard"


def test_sweep_results_dir() -> None:
    assert sweep_results_dir(Path("/vol"), "main") == Path("/vol/sweeps/main/results")


def test_volume_root_defaults_to_workspace() -> None:
    assert volume_root({}) == Path("/workspace/mprl")


def test_volume_root_from_env() -> None:
    assert volume_root({"MPRL_VOLUME_ROOT": "/data/x"}) == Path("/data/x")
