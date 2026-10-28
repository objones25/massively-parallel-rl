"""Root conftest.py: put this at the top of the tests/ tree.

pytest imports conftest.py automatically, and fixtures defined here are visible
to every test below this directory with no import statement.

What it buys you:
  * determinism  : stdlib and NumPy global RNGs reseeded before every test
  * isolation    : outbound network calls fail loudly (loopback and Unix sockets allowed)
  * hygiene      : cwd and app environment variables cannot leak between tests
  * hypothesis   : dev / ci / nightly / crosshair profiles, selected with --hypothesis-profile
  * markers      : tests under smoke/, network/, gpu/ are marked by directory

Verified against pytest 9.1.1 and Hypothesis 6.168 (October 2026).
"""

from __future__ import annotations

import importlib.util
import ipaddress
import os
import random
import socket
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

# One fixed seed for the whole suite. Any value works; it must never change.
SEED = 20260826

# Environment variable prefixes owned by the project under test. Edit these.
APP_ENV_PREFIXES = ("MPRL_", "WANDB_", "HF_", "RUNPOD_")

MARKERS = (
    "smoke: real libraries (mjlab, rsl_rl) on CPU; runs by default",
    "network: real W&B / Hugging Face / RunPod calls; needs keys; deselected by default",
    "gpu: requires CUDA; deselected by default",
)

DIRECTORY_MARKERS = ("smoke", "network", "gpu")


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _deterministic_rngs() -> None:
    """Reseed the global RNGs before each test, so a test reproduces when run alone.

    This covers random.* and the legacy np.random.* functions. It does NOT reach
    an unseeded np.random.default_rng() or random.Random() created by the code
    under test; those need a seed passed in explicitly.
    """
    random.seed(SEED)
    if importlib.util.find_spec("numpy") is not None:
        import numpy as np

        np.random.seed(SEED)


@pytest.fixture
def rng() -> random.Random:
    """An explicit, independent RNG that nothing else in the process can perturb."""
    return random.Random(SEED)


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
    """
    for key in list(os.environ):
        if key.startswith(APP_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYTHONHASHSEED", str(SEED))


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------
@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """A controllable clock, so tests never sleep.

    Patches time.time/monotonic/perf_counter/sleep on the time module. Code that did
    `from time import time` keeps the real function, and datetime.now() is not
    affected. For those, inject a clock or use the time-machine package.

        def test_timeout(frozen_clock):
            start_job()
            frozen_clock[0] += 3600
            assert job_is_expired()
    """
    import time as _time

    now = [1_000_000.0]
    monkeypatch.setattr(_time, "time", lambda: now[0])
    monkeypatch.setattr(_time, "monotonic", lambda: now[0])
    monkeypatch.setattr(_time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(_time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    return now


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
    """Mark tests by directory, so the default -m deselection cannot be forgotten."""
    for item in items:
        parts = {p.name for p in Path(str(item.path)).parents}
        for name in DIRECTORY_MARKERS:
            if name in parts:
                item.add_marker(getattr(pytest.mark, name))


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Print the seed in the run header; it is what lets you reproduce a CI failure."""
    return [f"pytest-expert: seed={SEED} python={sys.version.split()[0]}"]


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _torch_determinism(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make PyTorch ops reproducible for the duration of each test.

    Four separate sources of nondeterminism, each needing its own switch:

      torch.manual_seed          seeds CPU and all CUDA devices
      cudnn.benchmark = False    stops cuDNN re-picking a conv algorithm per run
      cudnn.deterministic = True forces the deterministic conv algorithm
      use_deterministic_algorithms(True)
                                 raises on any op with no deterministic
                                 implementation, so nondeterminism becomes a
                                 loud error rather than a slow drift

    warn_only=True is the pragmatic default: ops without a deterministic
    kernel warn instead of erroring. Flip to warn_only=False when chasing an
    actual reproducibility bug -- note that under filterwarnings=error a warn
    will itself fail the test, which is usually what you want.

    Determinism is slower. That is the trade you are making on purpose.
    """
    torch = pytest.importorskip("torch")

    # CUBLAS needs this set before the first CUDA context to be deterministic.
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(SEED)
    torch.manual_seed(SEED)
    try:
        import numpy as np

        np.random.seed(SEED)
    except ImportError:
        pass

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------
# Devices
# --------------------------------------------------------------------------
@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request: pytest.FixtureRequest):
    """Run the test on CPU, and again on CUDA when the gpu marker is selected.

    A parametrized fixture: every test that requests `device` runs once per
    param. The cuda case carries the gpu marker, so the default
    `-m "not gpu"` selection skips it on a laptop and CI picks it up on a
    GPU runner with `-m gpu`.
    """
    torch = pytest.importorskip("torch")
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine")
    return torch.device(request.param)


@pytest.fixture
def seeded_generator():
    """A torch.Generator with fixed state, for reproducible DataLoader shuffling.

    Pass to DataLoader(generator=...) alongside worker_init_fn so multi-worker
    loading is reproducible too -- workers otherwise get their own RNG state.
    """
    torch = pytest.importorskip("torch")
    g = torch.Generator()
    g.manual_seed(SEED)
    return g


@pytest.fixture
def seed_worker() -> Callable[[int], None]:
    """worker_init_fn for DataLoader. Straight from the PyTorch docs.

        DataLoader(ds, num_workers=2, worker_init_fn=seed_worker, generator=g)
    """
    torch = pytest.importorskip("torch")

    def _seed_worker(_worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        try:
            import numpy as np

            np.random.seed(worker_seed)
        except ImportError:
            pass

    return _seed_worker


# --------------------------------------------------------------------------
# Tiny fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def tiny_batch(device):
    """A batch small enough that a test costs milliseconds.

    Shape (4, 3, 8, 8): batch 4, 3 channels, 8x8 spatial. Big enough to catch
    a broadcasting bug, small enough that a full forward+backward is instant.
    Never load a real dataset in a unit test -- that is an integration test.
    """
    torch = pytest.importorskip("torch")
    x = torch.randn(4, 3, 8, 8, device=device)
    y = torch.randint(0, 2, (4,), device=device)
    return x, y


@pytest.fixture
def tiny_model(device):
    """A 2-layer model with a handful of parameters.

    Use this to test the *plumbing* -- shapes, dtypes, gradient flow, the
    training step -- not model quality. Testing a real architecture's accuracy
    is a benchmark, not a unit test, and belongs behind @pytest.mark.slow.
    """
    pytest.importorskip("torch")
    from torch import nn

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    ).to(device)
    return model


@pytest.fixture
def assert_close():
    """torch.testing.assert_close with the tolerances stated explicitly.

    Never compare float tensors with ==. assert_close checks shape, dtype and
    device as well as values, and its failure message shows the largest
    absolute and relative deviation and where it occurred.

    rtol/atol defaults here are the float32 defaults; tighten for float64,
    loosen for float16/bfloat16 (roughly rtol=1e-2 for half precision).
    """
    torch = pytest.importorskip("torch")

    def _assert_close(actual, expected, *, rtol: float = 1.3e-6, atol: float = 1e-5) -> None:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    return _assert_close
