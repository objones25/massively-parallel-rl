"""Setup checks: each check returns a detail line or raises.

``run_checks`` turns an operating error (``MprlOperatingError``: a missing key, a rejected credential, an
unwritable volume) into ``ok=False`` with the error as the detail. A broken install raises ``ImportError``,
``AttributeError`` (PyOpenGL without libEGL: ``'NoneType' object has no attribute 'eglQueryString'``) or
``OSError`` (a full volume) from deep inside a library; those also become a FAIL line, named by type and
message, so the remaining checks still run. Anything else, a programmer error (``CheckFailed``) included,
propagates. Each service check converts that service's documented failure exceptions into
``PreflightFailed``:

- W&B (wandb 0.31.0): ``wandb.Api(api_key=..., timeout=...)`` verifies the key while it is built and
  raises ``wandb.errors.AuthenticationError`` (a ``CommError``, so a ``wandb.errors.Error``) both for a bad
  key and when the server is unreachable (observed 2026-10-28 with a bogus key, a closed localhost port
  and an unresolvable host; the outage case after ``timeout`` seconds). ``.viewer`` documents
  ``WandbApiFailedError`` and ``ValueError``. All of these become ``PreflightFailed``.
- Hugging Face (huggingface_hub 2.3.0): ``HfHubHTTPError`` (a bad token: 401, observed) and
  ``httpx2.TransportError``; ``HubPublisher.ensure_repo`` raises ``UploadFailed`` after its retries.
  ``whoami()`` returns the token's role at ``auth.accessToken.role`` (the library itself reads it there in
  ``_login.py``; the Hub docs name the roles read, write and fine-grained). A ``read`` token fails; a
  fine-grained token's per-repo permissions are not exposed in a documented shape, so for it (and for a
  reply with no role) the detail says write access is not verified.
- RunPod: ``PodTerminator.status`` raises ``TerminateFailed`` for a non-200/404 reply or exhausted retries.
"""

from __future__ import annotations

import ctypes
import logging
import re
import secrets
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
import httpx2
import torch
import wandb
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from wandb.errors import Error as WandbError
from wandb.sdk.lib.service.service_connection import WandbApiFailedError

from mprl.config import TASK_ID
from mprl.errors import MissingSecret, MprlOperatingError, PreflightFailed, TerminateFailed
from mprl.hub import HubPublisher
from mprl.negative_space import bounded, check_finite, expect, require
from mprl.paths import volume_root
from mprl.runpod import own_terminator
from mprl.tasks import load_task_env_cfg

log = logging.getLogger(__name__)

Check = tuple[str, Callable[[], str]]

_ENV_CHECK_NUM_ENVS: Final = 2
_ENV_CHECK_STEPS: Final = 4
_GPU_DEVICE: Final = "cuda:0"
_RUNPOD_TIMEOUT_S: Final = 30.0
_WANDB_TIMEOUT_S: Final = 30
_VOLUME_MOUNT: Final = Path("/workspace")  # where RunPod mounts the network volume
IMAGE_VENV: Final = Path("/opt/venv")  # the image's environment (docker/Dockerfile)
CONTAINER_UV_CACHE: Final = "/root/.cache/uv"
PROC_MOUNTS: Final = Path("/proc/mounts")
_MAX_MOUNTS: Final = 10_000
_MOUNT_ESCAPE: Final = re.compile(r"\\([0-7]{3})")  # /proc/mounts escapes space, tab, newline and \ as \ooo
_EGL_LIBRARY: Final = "libEGL.so.1"
_ENVIRONMENT_ERRORS: Final = (ImportError, AttributeError, OSError)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_checks(checks: Sequence[Check]) -> list[CheckResult]:
    results: list[CheckResult] = []
    for name, check in bounded(checks, len(checks), "preflight checks"):
        try:
            detail = check()
        except (MprlOperatingError, *_ENVIRONMENT_ERRORS) as error:
            log.warning("preflight %s failed: %s: %s", name, type(error).__name__, error)
            results.append(CheckResult(name, ok=False, detail=f"{type(error).__name__}: {error}"))
            continue
        results.append(CheckResult(name, ok=True, detail=detail))
    require(len(results) == len(checks), "a check produced no result")
    return results


def _under(path: Path, parent: Path) -> bool:  # nsp: ignore NSP001 pure predicate
    """``path`` is ``parent`` or below it, both with symlinks resolved (neither has to exist)."""
    return path.resolve().is_relative_to(parent.resolve())


def check_python_env(
    env: Mapping[str, str], *, repo_dir: Path, torch_file: str, prefix: str, linux: bool, image_venv: Path
) -> str:
    """Python, torch and the uv cache live on the container disk, and ``uv run`` cannot sync the image venv.

    A venv or cache on the volume re-downloads torch and the nvidia wheels onto the volume's quota (which
    ``df`` does not show); that filled it on 2026-10-30, mid-install.
    """
    require(bool(torch_file), "torch has no __file__")
    torch_path, prefix_path = Path(torch_file), Path(prefix)
    problems: list[str] = []
    places = (volume_root(env), _VOLUME_MOUNT, repo_dir / ".venv")
    holding = [place for place in places if _under(torch_path, place)]
    if holding:
        problems.append(f"torch is imported from {torch_path.parent} (under {holding[0]}), not the base image's")
    if _under(prefix_path, _VOLUME_MOUNT):
        problems.append(f"the interpreter's prefix {prefix_path} is on the volume")
    cache = env.get("UV_CACHE_DIR")
    if cache and _under(Path(cache), _VOLUME_MOUNT):
        problems.append(f"UV_CACHE_DIR={cache} is on the volume; export UV_CACHE_DIR={CONTAINER_UV_CACHE}")
    if linux and image_venv.is_dir() and not env.get("UV_NO_SYNC"):
        problems.append(f"UV_NO_SYNC is not set, so `uv run` would sync {image_venv}; export UV_NO_SYNC=1")
    if problems:
        raise PreflightFailed(
            "; ".join(problems) + f". Re-run scripts/setup.sh, then `. /etc/profile.d/mprl-uv.sh` (venv {image_venv})"
        )
    return f"python {prefix_path}; torch {torch_path}"


def check_egl(load: Callable[[str], Any] = ctypes.CDLL) -> str:
    """libEGL.so.1 loads (mjlab sets MUJOCO_GL=egl; PyOpenGL needs it at ``import mjlab.tasks``)."""
    try:
        load(_EGL_LIBRARY)
    except OSError as error:  # ctypes.CDLL raises OSError when the library cannot be found or loaded
        raise PreflightFailed(
            f"{_EGL_LIBRARY} cannot be loaded ({error}); run: apt-get update && apt-get install -y libegl1"
        ) from error
    return f"{_EGL_LIBRARY} loads"


def _secret(env: Mapping[str, str], name: str) -> str:
    value = env.get(name)
    if not value:
        raise MissingSecret(f"{name} is not set; run scripts/setup.sh")
    return value


def check_cuda(device: str) -> str:
    """PyTorch and Warp (MuJoCo Warp's kernel runtime) both see ``device`` and can allocate on it."""
    import warp as wp  # imported here: initializing Warp prints its banner

    if not torch.cuda.is_available():
        raise PreflightFailed("torch.cuda.is_available() is False")
    if not wp.is_cuda_available():
        raise PreflightFailed("Warp sees no CUDA device")
    probe = wp.zeros(1, dtype=wp.float32, device=device).numpy()
    if probe.tolist() != [0.0]:
        raise PreflightFailed(f"a Warp array on {device} read back {probe.tolist()}")
    return f"{torch.cuda.get_device_name(device)}; Warp {wp.config.version} allocates on {device}"


def check_env(device: str) -> str:
    """The G1 task with 2 envs builds on ``device`` (MuJoCo Warp) and steps with zero actions."""
    from mjlab.envs import ManagerBasedRlEnv  # lazy: see mprl.tasks

    cfg = load_task_env_cfg(TASK_ID)
    cfg.scene.num_envs = _ENV_CHECK_NUM_ENVS
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
        env.reset()
        actions = torch.zeros(_ENV_CHECK_NUM_ENVS, env.action_manager.total_action_dim, device=device)
        for _ in bounded(range(_ENV_CHECK_STEPS), _ENV_CHECK_STEPS, "env check steps"):
            _obs, reward, *_ = env.step(actions)
            check_finite(reward, "reward")
    finally:
        env.close()
    return f"{TASK_ID} with {_ENV_CHECK_NUM_ENVS} envs stepped {_ENV_CHECK_STEPS} times on {device}"


def check_wandb(env: Mapping[str, str]) -> str:
    key = _secret(env, "WANDB_API_KEY")
    try:
        viewer = wandb.Api(api_key=key, timeout=_WANDB_TIMEOUT_S).viewer
        username, entity = viewer.username, viewer.entity
    except (WandbError, WandbApiFailedError, ValueError) as error:
        raise PreflightFailed(f"W&B rejected WANDB_API_KEY: {type(error).__name__}: {error}") from error
    target = env.get("WANDB_ENTITY") or entity
    return f"logged in as {username}; runs go to {target}/{env.get('WANDB_PROJECT') or '(default project)'}"


def check_hf(env: Mapping[str, str]) -> str:
    token = _secret(env, "HF_TOKEN")
    repo_id = _secret(env, "HF_REPO_ID")
    api = HfApi(token=token)
    try:
        info = api.whoami()
        user, role = info["name"], ((info.get("auth") or {}).get("accessToken") or {}).get("role")
        if role == "read":
            raise PreflightFailed(
                f"HF_TOKEN is a read token for {user}; uploads need a write token: https://huggingface.co/settings/tokens"
            )
        HubPublisher(api, repo_id, private=True).ensure_repo()
    except (HfHubHTTPError, httpx2.TransportError) as error:
        raise PreflightFailed(f"Hugging Face: {type(error).__name__}: {error}") from error
    if role == "write":
        return f"write token for {user}; repo {repo_id} exists"
    return f"token for {user} (role: {role or 'none reported'}; write access not verified); repo {repo_id} exists"


def check_runpod(env: Mapping[str, str], client: httpx.Client | None = None) -> str:
    """The own pod id resolves and a GET of that pod with RUNPOD_API_KEY returns 200.

    A GET is all a check may do (a DELETE would end the pod), so a Read Only key passes; the detail says so.
    """
    if client is None:
        with httpx.Client(timeout=_RUNPOD_TIMEOUT_S) as owned:
            return check_runpod(env, owned)
    terminator = own_terminator(env, client)
    try:
        status = terminator.status()
    except TerminateFailed as error:
        raise PreflightFailed(str(error)) from error
    if status is None:
        raise PreflightFailed(f"pod {terminator.pod_id} not found (HTTP 404); is RUNPOD_POD_ID this pod's id?")
    return (
        f"pod {terminator.pod_id} is {status}; API key accepted for GET only: self-termination (DELETE) needs a key "
        'with "All" permission, which this check does not prove'
    )


def _mount_points(mounts: Path) -> list[Path]:
    """The mount points (second field) in a ``/proc/mounts``-format file, octal escapes decoded."""
    lines = mounts.read_text(encoding="utf-8").splitlines()
    points: list[Path] = []
    for line in bounded(lines, _MAX_MOUNTS, f"{mounts} lines"):
        fields = line.split()
        if len(fields) >= 2:
            points.append(Path(_MOUNT_ESCAPE.sub(lambda m: chr(int(m.group(1), 8)), fields[1])))
    require(len(points) <= len(lines), "more mount points than lines")
    return points


def _volume_mount(root: Path, mounts: Path) -> Path:
    """The mount holding ``root`` (the deepest mount point above it); ``PreflightFailed`` if that is ``/``."""
    resolved = root.resolve()
    holding = [point for point in _mount_points(mounts) if resolved.is_relative_to(point)]
    mount = max(holding, key=lambda point: len(point.parts), default=Path("/"))
    if mount == Path("/"):
        raise PreflightFailed(
            f"{root} is on the container disk (mount /), which is lost with the pod; set MPRL_VOLUME_ROOT under "
            f"{_VOLUME_MOUNT} (the network volume)"
        )
    return mount


def check_volume(env: Mapping[str, str], mounts: Path | None = None) -> str:
    """MPRL_VOLUME_ROOT is on a mount other than ``/`` (when ``mounts`` exists), and is writable.

    ``mounts`` is a ``/proc/mounts``-format file; ``None`` or a missing file (macOS) skips the mount rule.
    The mount is checked first, so a root on the container disk is never created.
    """
    root = volume_root(env)
    mount = _volume_mount(root, mounts) if mounts is not None and mounts.is_file() else None
    probe = root / f".preflight-{secrets.token_hex(8)}"
    token = secrets.token_hex(16)
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe.write_text(token)
        read = probe.read_text()
        probe.unlink()
    except OSError as error:
        raise PreflightFailed(f"{root} is not writable: {type(error).__name__}: {error}") from error
    if read != token:
        raise PreflightFailed(f"{probe} read back different bytes than were written")
    return f"{root} is writable" if mount is None else f"{root} is writable (on mount {mount})"


def default_checks(env: Mapping[str, str], require_gpu: bool, *, repo_dir: Path) -> list[Check]:
    """Every check in order. Without ``require_gpu`` (local testing) the pod-only checks are skipped (CUDA,
    python-env, EGL, RunPod, and the volume's mount rule) and the env runs on CPU."""
    device = _GPU_DEVICE if require_gpu else "cpu"
    torch_file = expect(torch.__file__, "torch was imported without a __file__")
    checks: list[Check] = []
    if require_gpu:
        checks += [
            ("cuda", lambda: check_cuda(device)),
            (
                "python-env",
                lambda: check_python_env(
                    env,
                    repo_dir=repo_dir,
                    torch_file=torch_file,
                    prefix=sys.prefix,
                    linux=sys.platform.startswith("linux"),
                    image_venv=IMAGE_VENV,
                ),
            ),
            ("egl", check_egl),
        ]
    checks += [
        ("env", lambda: check_env(device)),
        ("wandb", lambda: check_wandb(env)),
        ("hf", lambda: check_hf(env)),
    ]
    if require_gpu:
        checks.append(("runpod", lambda: check_runpod(env)))
    mounts = PROC_MOUNTS if require_gpu else None
    checks.append(("volume", lambda: check_volume(env, mounts)))
    require(len({name for name, _ in checks}) == len(checks), "duplicate check name")
    return checks
