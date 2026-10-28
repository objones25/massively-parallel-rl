"""Setup checks: each check returns a detail line or raises.

``run_checks`` turns an operating error (``MprlOperatingError``: a missing key, a rejected credential, an
unwritable volume) into ``ok=False`` with the error as the detail; anything else, a programmer error
included, propagates. Each service check converts that service's documented failure exceptions into
``PreflightFailed``:

- W&B (wandb 0.31.0): ``wandb.Api(api_key=...).viewer`` raises ``wandb.errors.Error`` subclasses
  (``AuthenticationError`` for a bad key, observed), ``WandbApiFailedError`` or ``ValueError``
  (documented on ``Api.viewer``).
- Hugging Face (huggingface_hub 2.3.0): ``HfHubHTTPError`` (a bad token: 401, observed) and
  ``httpx2.TransportError``; ``HubPublisher.ensure_repo`` raises ``UploadFailed`` after its retries.
- RunPod: ``PodTerminator.status`` raises ``TerminateFailed`` for a non-200/404 reply or exhausted retries.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import httpx
import httpx2
import mjlab.tasks  # noqa: F401  # pyright: ignore[reportUnusedImport]  (registers the tasks)
import torch
import wandb
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from wandb.errors import Error as WandbError
from wandb.sdk.lib.service.service_connection import WandbApiFailedError

from mprl.config import TASK_ID
from mprl.errors import MissingSecret, MprlOperatingError, PreflightFailed, TerminateFailed
from mprl.hub import HubPublisher
from mprl.negative_space import bounded, check_finite
from mprl.paths import volume_root
from mprl.runpod import own_terminator

log = logging.getLogger(__name__)

Check = tuple[str, Callable[[], str]]

_ENV_CHECK_NUM_ENVS: Final = 2
_ENV_CHECK_STEPS: Final = 4
_GPU_DEVICE: Final = "cuda:0"
_RUNPOD_TIMEOUT_S: Final = 30.0


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
        except MprlOperatingError as error:
            log.warning("preflight %s failed: %s", name, error)
            results.append(CheckResult(name, ok=False, detail=f"{type(error).__name__}: {error}"))
            continue
        results.append(CheckResult(name, ok=True, detail=detail))
    return results


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
    cfg = load_env_cfg(TASK_ID)
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
        viewer = wandb.Api(api_key=key).viewer
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
        user = api.whoami()["name"]
        HubPublisher(api, repo_id, private=True).ensure_repo()
    except (HfHubHTTPError, httpx2.TransportError) as error:
        raise PreflightFailed(f"Hugging Face: {type(error).__name__}: {error}") from error
    return f"token for {user}; repo {repo_id} exists"


def check_runpod(env: Mapping[str, str], client: httpx.Client | None = None) -> str:
    """The own pod id resolves and a GET of that pod with RUNPOD_API_KEY returns 200."""
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
    return f"pod {terminator.pod_id} is {status}; API key accepted"


def check_volume(env: Mapping[str, str]) -> str:
    """Write, read back and delete a temp file under MPRL_VOLUME_ROOT."""
    root = volume_root(env)
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
    return f"{root} is writable"


def default_checks(env: Mapping[str, str], require_gpu: bool) -> list[Check]:
    """Every check in order; without ``require_gpu`` the CUDA and pod checks are skipped and the env runs on CPU."""
    device = _GPU_DEVICE if require_gpu else "cpu"
    checks: list[Check] = []
    if require_gpu:
        checks.append(("cuda", lambda: check_cuda(device)))
    checks += [
        ("env", lambda: check_env(device)),
        ("wandb", lambda: check_wandb(env)),
        ("hf", lambda: check_hf(env)),
    ]
    if require_gpu:
        checks.append(("runpod", lambda: check_runpod(env)))
    checks.append(("volume", lambda: check_volume(env)))
    return checks
