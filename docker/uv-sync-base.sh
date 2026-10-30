#!/usr/bin/env bash
#
# uv-sync-base.sh: `uv sync --frozen` that skips every package named in base-provided-packages.txt
# (next to this script): torch, torchvision, triton and the nvidia-* CUDA wheels, which the Docker base
# image already ships in its system site-packages. The image's /opt/venv is created with
# --system-site-packages, so it imports the base's copies; a plain `uv sync` would download them again.
# The one place that turns the list into uv arguments: docker/Dockerfile and scripts/setup.sh both call it.
#
# It refuses unless UV_PROJECT_ENVIRONMENT names an existing venv whose pyvenv.cfg has
# `include-system-site-packages = true` (docker/Dockerfile and scripts/setup.sh create /opt/venv first).
#
# Usage: docker/uv-sync-base.sh [extra `uv sync` arguments, e.g. --no-install-project or --dry-run]
# Requires: bash 3.2+, uv, grep. Run from the project root (uv finds pyproject.toml from the working directory).
# Exit status: 1 when the environment is not such a venv or the list cannot be read; otherwise that of `uv sync`.
# bash-target: 3.2

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true # bash 4.4+: $(...) inherits set -e

readonly SCRIPT_NAME=${0##*/}

die() {
  printf '%s: error: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

# The environment must already be a venv that sees the system site-packages: if it were missing, uv would
# create one WITHOUT them ("If an environment is not present at the provided path, uv will create it"), and
# skipping the base's packages would leave it with no torch at all.
require_system_site_venv() {
  local env cfg
  env=${UV_PROJECT_ENVIRONMENT:-}
  [[ -n $env ]] || die "UV_PROJECT_ENVIRONMENT is not set; point it at a venv made with: uv venv --system-site-packages <dir>"
  cfg=$env/pyvenv.cfg
  [[ -f $cfg ]] || die "$env has no pyvenv.cfg; create it first with: uv venv --system-site-packages $env"
  grep -Eq '^[[:space:]]*include-system-site-packages[[:space:]]*=[[:space:]]*true[[:space:]]*$' "$cfg" \
    || die "$env is a venv without system site packages ($cfg); recreate it with: uv venv --system-site-packages $env"
}

main() {
  local here list line
  local -a skip=()
  require_system_site_venv
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
