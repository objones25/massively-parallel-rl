"""Root conftest.py: determinism, isolation, tiers and timeouts for every test below tests/.

What it buys you:
  * determinism  : stdlib, NumPy and (once imported) torch global RNGs reseeded before every test;
                   ``torch_determinism`` for the tests that compare runs bit for bit
  * isolation    : outbound network calls fail loudly (loopback and Unix sockets allowed)
  * hygiene      : cwd and app environment variables cannot leak between tests
  * hypothesis   : dev / ci / nightly profiles, selected with HYPOTHESIS_PROFILE
  * tiers        : tests under smoke/, network/, gpu/ are marked by directory; smoke tests get a
                   longer pytest-timeout than the fast default tier
"""

from __future__ import annotations

import importlib.util
import ipaddress
import os
import random
import socket
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

# One fixed seed for the whole suite. Any value works; it must never change.
SEED = 20260826

# Environment variable prefixes owned by the project under test.
APP_ENV_PREFIXES = ("MPRL_", "WANDB_", "HF_", "RUNPOD_")

MARKERS = (
    "smoke: real mjlab / rsl_rl / wandb offline / ffmpeg / subprocesses on CPU; deselected by default (pytest -m smoke)",
    "network: real W&B / Hugging Face / RunPod calls; needs keys; deselected by default",
    "gpu: requires CUDA; deselected by default (pytest -m gpu on the pod)",
)

DIRECTORY_MARKERS = ("smoke", "network", "gpu")

# Per-test pytest-timeout for the heavy tiers; the fast tier uses `timeout` from pyproject. It covers
# fixture setup too, and a session fixture (one real G1 training) runs inside its first test.
HEAVY_TIMEOUT_S = 600


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _deterministic_rngs() -> None:
    """Reseed the global RNGs before each test, so a test reproduces when run alone.

    Covers random.*, the legacy np.random.* functions and torch's global generator when torch is
    already imported (it is never imported just for this). It does NOT reach an unseeded
    np.random.default_rng() or random.Random() created by the code under test.
    """
    random.seed(SEED)
    if importlib.util.find_spec("numpy") is not None:
        import numpy as np

        np.random.seed(SEED)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(SEED)


@pytest.fixture
def torch_determinism(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Deterministic torch kernels for a test that compares two runs bit for bit.

    Not autouse: production training does not run under ``use_deterministic_algorithms``, so only the
    tests whose assertion is bitwise equality opt in. ``warn_only=True``: an op without a deterministic
    kernel warns (an error under filterwarnings=error) instead of raising mid-op.
    """
    torch = pytest.importorskip("torch")
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:  # needed before the first CUDA context
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
        monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------
# Isolation
# --------------------------------------------------------------------------
class NetworkBlockedError(RuntimeError):
    """Raised when a unit test tries to reach a non-loopback host."""


def _is_local(host: object) -> bool:
    if not isinstance(host, str):
        return False
    if host in {"localhost", ""}:
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make outbound network access fail immediately in unit tests.

    Blocks at the Python socket layer, which also catches calls made deep inside
    dependencies. Loopback and Unix-domain sockets stay allowed so local test
    servers and multiprocessing keep working. Known gaps: subprocesses (curl, git)
    and C extensions that open sockets without the Python socket module.

    Tests marked network are exempt: they exercise the real services.
    """
    if request.node.get_closest_marker("network"):
        return

    def blocked(target: object) -> NetworkBlockedError:
        return NetworkBlockedError(
            f"Test tried to reach {target!r}. Stub the boundary with monkeypatch, "
            "or mark the test @pytest.mark.network."
        )

    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self: socket.socket, address: object) -> None:
        if self.family == getattr(socket, "AF_UNIX", None) or (isinstance(address, tuple) and _is_local(address[0])):
            return real_connect(self, address)
        raise blocked(address)

    def guarded_getaddrinfo(host: object, *args: object, **kwargs: object):
        if _is_local(host):
            return real_getaddrinfo(host, *args, **kwargs)
        raise blocked(host)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


@pytest.fixture(autouse=True)
def _isolated_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run each test in its own empty directory; relative writes land somewhere disposable."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove app configuration from the environment.

    PYTHONHASHSEED is set for child processes only. The running interpreter fixed
    its hash seed at startup; to pin it for this process, set it before launching
    pytest (CI does this in the workflow env).

    IMAGEIO_FFMPEG_NO_PREVENT_SIGINT: imageio-ffmpeg (``_utils._popen_kwargs``, read on every call) then
    drops its ``preexec_fn=os.setpgrp``, so no Python code runs in the child between fork and exec of
    ffmpeg; its own comment cites a hang on fork() (imageio-ffmpeg #58).
    """
    for key in list(os.environ):
        if key.startswith(APP_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYTHONHASHSEED", str(SEED))
    monkeypatch.setenv("IMAGEIO_FFMPEG_NO_PREVENT_SIGINT", "1")


# --------------------------------------------------------------------------
# Hypothesis profiles
# --------------------------------------------------------------------------
if importlib.util.find_spec("hypothesis") is not None:
    from hypothesis import settings

    settings.register_profile("dev", max_examples=100)
    settings.register_profile("ci", max_examples=300, deadline=None, print_blob=True)
    settings.register_profile("nightly", max_examples=5_000, deadline=None, print_blob=True)
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


# --------------------------------------------------------------------------
# Markers and reporting
# --------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    """Register markers in code too, so this conftest works before pyproject is updated."""
    for line in MARKERS:
        config.addinivalue_line("markers", line)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark tests by directory (the default -m deselection cannot be forgotten); heavy tiers get a longer timeout."""
    for item in items:
        parts = {p.name for p in Path(str(item.path)).parents}
        for name in DIRECTORY_MARKERS:
            if name in parts:
                item.add_marker(getattr(pytest.mark, name))
        heavy = any(item.get_closest_marker(name) for name in DIRECTORY_MARKERS)
        if heavy and item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(HEAVY_TIMEOUT_S))


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Print the seed in the run header; it is what lets you reproduce a CI failure."""
    return [f"pytest-expert: seed={SEED} python={sys.version.split()[0]}"]
