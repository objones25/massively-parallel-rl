"""Build-time check of the training image (run by docker/Dockerfile with /opt/venv/bin/python; fails the build).

Usage: python docker/verify_image.py BASE_TORCH_FILE

BASE_TORCH_FILE is ``torch.__file__`` as the base image's own interpreter (/usr/bin/python3.12) reports it.
Checks that /opt/venv imports exactly that torch (2.8.0, from the base's system site-packages), that none of
docker/base-provided-packages.txt was installed into /opt/venv, and that the training stack imports.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
import sys
import sysconfig
from pathlib import Path

TORCH_VERSION = "2.8.0"
TORCHVISION_VERSION = "0.23.0"
STACK = ("mjlab", "mjlab.tasks", "mujoco", "mujoco_warp", "warp", "rsl_rl", "wandb", "huggingface_hub", "mprl")
LIST_FILE = Path(__file__).resolve().parent / "base-provided-packages.txt"


def fail(message: str) -> None:
    raise SystemExit(f"verify_image: FAIL: {message}")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def base_provided() -> set[str]:
    lines = LIST_FILE.read_text(encoding="utf-8").splitlines()
    return {normalize(line.strip()) for line in lines if line.strip() and not line.startswith("#")}


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        fail("usage: verify_image.py BASE_TORCH_FILE")
    base_torch_file = Path(argv[0]).resolve()
    venv = Path(sys.prefix).resolve()
    if venv == Path(sys.base_prefix).resolve():
        fail(f"not running inside a virtual environment (sys.prefix={sys.prefix})")

    import torch
    import torchvision

    torch_file = Path(torch.__file__).resolve()
    if not torch.__version__.startswith(TORCH_VERSION):
        fail(f"torch {torch.__version__}, expected {TORCH_VERSION}")
    if not torchvision.__version__.startswith(TORCHVISION_VERSION):
        fail(f"torchvision {torchvision.__version__}, expected {TORCHVISION_VERSION}")
    if torch_file.is_relative_to(venv):
        fail(f"torch imported from the venv ({torch_file}), not from the base image")
    if torch_file != base_torch_file:
        fail(f"torch imported from {torch_file}, but the base interpreter's torch is {base_torch_file}")

    venv_site = Path(sysconfig.get_paths()["purelib"])
    in_venv = {normalize(d.metadata["Name"]) for d in importlib.metadata.distributions(path=[str(venv_site)])}
    duplicated = sorted(in_venv & base_provided())
    if duplicated:
        fail(f"base-provided packages were installed into {venv_site}: {duplicated}")

    for module in STACK:
        importlib.import_module(module)

    print(f"verify_image: OK torch {torch.__version__} from {torch_file}")
    print(
        f"verify_image: OK torchvision {torchvision.__version__}; none of {len(base_provided())} base packages in the venv"
    )
    print(f"verify_image: OK imported {', '.join(STACK)}")


if __name__ == "__main__":
    main(sys.argv[1:])
