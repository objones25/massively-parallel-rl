#!/usr/bin/env bash
#
# train.sh: run an `mprl` subcommand in the image's environment; with no arguments, the main sweep
# reusing the pilot (`mprl sweep --name main --reuse-pilot`; the reuse is refused, with the reason, unless
# the pilot is complete on the same code). The CLI loads .env itself, resumes every run from its latest
# checkpoint on the volume, and terminates the pod when training ends unless --no-terminate is given.
#
# Usage: scripts/train.sh [mprl subcommand and options]   e.g. scripts/train.sh pilot
# Requires: bash 3.2+, uv; run scripts/setup.sh once first.
# Exit status: that of `mprl`.
# bash-target: 3.2

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true # bash 4.4+: $(...) inherits set -e

readonly SCRIPT_NAME=${0##*/}
readonly IMAGE_VENV=/opt/venv # the image's baked environment (never on the volume)

die() {
  printf '%s: error: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

main() {
  local repo
  if [[ $# -eq 0 ]]; then set -- sweep --name main --reuse-pilot; fi
  repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || die "cannot find the repo root"
  cd -- "$repo" || die "cannot cd to $repo"
  if [[ -z ${UV_PROJECT_ENVIRONMENT:-} && -d $IMAGE_VENV ]]; then
    export UV_PROJECT_ENVIRONMENT=$IMAGE_VENV
  fi
  exec uv run --no-sync mprl "$@"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
