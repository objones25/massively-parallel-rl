#!/usr/bin/env bash
#
# setup.sh: configure this pod for training. Refuses unless /workspace (the RunPod network volume)
# is mounted, installs the project into the image's environment, then runs `mprl setup`, which asks
# for the API keys, writes .env (mode 0600) in the repo root and runs the preflight checks.
#
# Usage: scripts/setup.sh [--allow-no-volume] [mprl setup options, e.g. --cpu]
#   --allow-no-volume   skip the /workspace mount check (local testing only)
# Requires: bash 3.2+, uv, awk, and /proc/mounts (Linux) unless --allow-no-volume.
# Exit status: 1 when /workspace is not mounted or uv sync fails; otherwise that of `mprl setup`.
# bash-target: 3.2

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true # bash 4.4+: $(...) inherits set -e

readonly SCRIPT_NAME=${0##*/}
readonly VOLUME=/workspace
readonly IMAGE_VENV=/opt/venv # the image's baked environment (never on the volume)

die() {
  printf '%s: error: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

# True when $1 is a mount point listed in /proc/mounts.
is_mounted() {
  [[ -r /proc/mounts ]] || return 1
  awk -v point="$1" '$2 == point { found = 1 } END { exit !found }' /proc/mounts || return 1
}

main() {
  local allow_no_volume=0 arg repo
  local -a forward=()
  for arg in "$@"; do
    case $arg in
      --allow-no-volume) allow_no_volume=1 ;;
      *) forward+=("$arg") ;;
    esac
  done
  # shellcheck disable=SC2310 # is_mounted is a predicate that handles its own failures (|| return 1)
  if ((allow_no_volume == 0)) && ! is_mounted "$VOLUME"; then
    die "$VOLUME is not mounted: create the pod with the network volume at $VOLUME (--allow-no-volume skips this check for local testing)"
  fi
  repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || die "cannot find the repo root"
  cd -- "$repo" || die "cannot cd to $repo"
  if [[ -z ${UV_PROJECT_ENVIRONMENT:-} && -d $IMAGE_VENV ]]; then
    export UV_PROJECT_ENVIRONMENT=$IMAGE_VENV
  fi
  uv sync --frozen || die "uv sync --frozen failed"
  exec uv run --no-sync mprl setup ${forward[@]+"${forward[@]}"}
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
