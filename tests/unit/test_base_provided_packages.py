"""docker/base-provided-packages.txt lists exactly the uv.lock packages the Docker base image already ships.

The image (docker/Dockerfile) and the pod's scripts/setup.sh both pass every name in that file to
`uv sync --no-install-package`, so torch and its CUDA wheels come from the base instead of being
downloaded again. A name missing from the file is a multi-GB re-download; a stale name hides a typo.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIST_FILE = REPO / "docker" / "base-provided-packages.txt"
# The base installs torch, torchvision and torchaudio from the cu128 index; pip pulls triton and the
# nvidia-* CUDA runtime wheels with torch. torchaudio is not in our lock, so it is not listed.
BASE_PROVIDED = re.compile(r"^(torch|torchvision|triton|nvidia-.*)$")


def _listed() -> list[str]:
    lines = LIST_FILE.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def _locked_names() -> set[str]:
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    return {package["name"] for package in lock["package"]}


def test_list_names_every_locked_torch_and_nvidia_package() -> None:
    expected = sorted(name for name in _locked_names() if BASE_PROVIDED.match(name))
    assert sorted(_listed()) == expected


def test_list_has_no_duplicates() -> None:
    listed = _listed()
    assert len(listed) == len(set(listed))


def test_lock_pins_torch_to_the_base_image_version() -> None:
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    versions = sorted({p["version"] for p in lock["package"] if p["name"] in {"torch", "torchvision"}})
    assert versions == ["0.23.0", "0.23.0+cu128", "2.8.0", "2.8.0+cu128"]
