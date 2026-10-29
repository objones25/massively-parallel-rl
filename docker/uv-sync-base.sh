#!/usr/bin/env bash
#
# uv-sync-base.sh: `uv sync --frozen` that skips every package named in base-provided-packages.txt
# (next to this script): torch, torchvision, triton and the nvidia-* CUDA wheels, which the Docker base
# image already ships in its system site-packages. The image's /opt/venv is created with
# --system-site-packages, so it imports the base's copies; a plain `uv sync` would download them again.
# The one place that turns the list into uv arguments: docker/Dockerfile and scripts/setup.sh both call it.
#
# Usage: docker/uv-sync-base.sh [extra `uv sync` arguments, e.g. --no-install-project or --dry-run]
# Requires: bash 3.2+, uv. Run from the project root (uv finds pyproject.toml from the working directory).
# Exit status: 1 when the list cannot be read; otherwise that of `uv sync`.
# bash-target: 3.2

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true # bash 4.4+: $(...) inherits set -e

readonly SCRIPT_NAME=${0##*/}

die() {
  printf '%s: error: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

main() {
  local here list line
  local -a skip=()
  here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || die "cannot find this script's directory"
  list=$here/base-provided-packages.txt
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line//[[:space:]]/}
    if [[ -n $line && $line != \#* ]]; then skip+=(--no-install-package "$line"); fi
  done <"$list" || die "cannot read $list"
  ((${#skip[@]} > 0)) || die "$list names no packages"
  exec uv sync --frozen ${skip[@]+"${skip[@]}"} "$@"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
