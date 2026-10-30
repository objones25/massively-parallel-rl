#!/usr/bin/env bash
#
# setup.sh: configure this pod for training. Refuses unless /workspace (the RunPod network volume)
# is mounted, installs the project into the image's environment, then runs `mprl setup`, which asks
# for the API keys, writes .env (mode 0600) in the repo root and runs the preflight checks.
#
# Volume mode (the default, on a pod) never installs anything onto the volume:
#   - the environment is /opt/venv (container disk). On a pod from the stock runpod/pytorch image, which has
#     no /opt/venv, it is created with `uv venv --system-site-packages` from /usr/bin/python3.12, whose system
#     site-packages hold torch; without torch there it refuses and names the project image. A
#     UV_PROJECT_ENVIRONMENT under /workspace is refused.
#   - the sync always goes through docker/uv-sync-base.sh, which skips every package in
#     docker/base-provided-packages.txt (torch, torchvision, triton, nvidia-*): the venv reuses the base's
#     copies, and a plain `uv sync` would download them again.
#   - UV_CACHE_DIR, when unset or on /workspace (the stock image sets /workspace/.cache/uv/), becomes
#     /root/.cache/uv: the volume's quota is not what `df` shows, and a cache there filled it.
#   - libEGL.so.1 (MuJoCo's headless rendering) is installed with apt-get when it cannot be loaded.
#   - /etc/profile.d/mprl-uv.sh (rewritten atomically) exports the same UV_* settings and PATH, so later
#     SSH / tmux login shells use /opt/venv and `uv run` never syncs.
# --allow-no-volume (local development) skips all of that: /opt/venv if present goes through
# docker/uv-sync-base.sh, any other environment gets a plain `uv sync --frozen`.
#
# Usage: scripts/setup.sh [--allow-no-volume] [mprl setup options, e.g. --cpu]
#   --allow-no-volume   skip the /workspace mount check and the pod-only steps (local testing only)
# Test-only overrides of the pod's fixed paths: MPRL_MOUNTS_FILE (/proc/mounts), MPRL_IMAGE_VENV (/opt/venv),
#   MPRL_SYSTEM_PYTHON (/usr/bin/python3.12), MPRL_UV_CACHE (/root/.cache/uv),
#   MPRL_PROFILE_FILE (/etc/profile.d/mprl-uv.sh).
# Requires: bash 3.2+, uv, awk, mktemp; on a pod also apt-get (root) when libEGL is missing.
# Exit status: 1 when /workspace is not mounted, the environment is unusable or an install step fails;
#   otherwise that of `mprl setup`.
# bash-target: 3.2

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true # bash 4.4+: $(...) inherits set -e

readonly SCRIPT_NAME=${0##*/}
readonly VOLUME=/workspace
readonly PROJECT_IMAGE=ghcr.io/objones25/massively-parallel-rl:latest
readonly MOUNTS_FILE=${MPRL_MOUNTS_FILE:-/proc/mounts}
readonly IMAGE_VENV=${MPRL_IMAGE_VENV:-/opt/venv} # the image's environment (never on the volume)
readonly SYSTEM_PYTHON=${MPRL_SYSTEM_PYTHON:-/usr/bin/python3.12}
readonly CONTAINER_UV_CACHE=${MPRL_UV_CACHE:-/root/.cache/uv}
readonly PROFILE_FILE=${MPRL_PROFILE_FILE:-/etc/profile.d/mprl-uv.sh}

profile_tmp=""
create_venv=0

die() {
  printf '%s: error: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

log() {
  printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
}

cleanup() {
  if [[ -n $profile_tmp ]]; then rm -f -- "$profile_tmp" || true; fi
}
trap cleanup EXIT

# True when $1 is a mount point listed in the mounts file.
is_mounted() {
  [[ -r $MOUNTS_FILE ]] || return 1
  awk -v point="$1" '$2 == point { found = 1 } END { exit !found }' "$MOUNTS_FILE" || return 1
}

# Prints $1 without trailing slashes and, when it is an existing directory, with symlinks resolved.
normalize_dir() {
  local path=$1
  path=${path%"${path##*[!/]}"}
  if [[ -z $path ]]; then path=/; fi
  if [[ -d $path ]]; then path=$(cd -P -- "$path" && pwd -P) || return 1; fi
  printf '%s\n' "$path"
}

# True when the normalized path $1 is /workspace or below it.
on_volume() {
  [[ $1 == "$VOLUME" || $1 == "$VOLUME"/* ]]
}

# Paths written into the login profile must need no quoting beyond single quotes.
require_plain_path() {
  [[ $1 =~ ^/[A-Za-z0-9_./-]*$ ]] || die "$2 must be an absolute path of [A-Za-z0-9_./-], got: $1"
}

system_python_has_torch() {
  [[ -x $SYSTEM_PYTHON ]] || return 1
  "$SYSTEM_PYTHON" -c 'import torch' >/dev/null 2>&1 || return 1
}

egl_loads() {
  "$SYSTEM_PYTHON" -c 'import ctypes; ctypes.CDLL("libEGL.so.1")' >/dev/null 2>&1 || return 1
}

# Volume mode: UV_PROJECT_ENVIRONMENT (exported, normalized) is a venv off the volume; create_venv=1 when the
# image venv is missing and the system python can seed it.
choose_volume_env() {
  local env image
  # shellcheck disable=SC2310 # normalize_dir / on_volume handle their own failures (|| return 1)
  image=$(normalize_dir "$IMAGE_VENV") || die "cannot resolve $IMAGE_VENV"
  if [[ -n ${UV_PROJECT_ENVIRONMENT:-} ]]; then
    # shellcheck disable=SC2310 # normalize_dir / on_volume handle their own failures (|| return 1)
    env=$(normalize_dir "$UV_PROJECT_ENVIRONMENT") || die "cannot resolve UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
    # shellcheck disable=SC2310 # normalize_dir / on_volume handle their own failures (|| return 1)
    if on_volume "$env"; then
      die "UV_PROJECT_ENVIRONMENT=$env is on the volume; a venv there re-downloads torch onto its quota. Unset it to use $image"
    fi
  else
    env=$image
  fi
  # shellcheck disable=SC2310 # system_python_has_torch is a predicate that handles its own failures
  if [[ $env == "$image" && ! -e $env ]]; then
    if ! system_python_has_torch; then
      die "$env does not exist and $SYSTEM_PYTHON cannot import torch; create the pod from the project image $PROJECT_IMAGE (or RunPod's PyTorch 2.8.0 image)"
    fi
    create_venv=1
  fi
  export UV_PROJECT_ENVIRONMENT=$env
}

# Volume mode: create the image venv chosen above, after every check that can refuse has passed.
create_image_venv() {
  ((create_venv == 1)) || return 0
  log "creating $UV_PROJECT_ENVIRONMENT from $SYSTEM_PYTHON with its system site-packages (the base's torch)"
  uv venv --system-site-packages --python "$SYSTEM_PYTHON" "$UV_PROJECT_ENVIRONMENT" \
    || die "uv venv $UV_PROJECT_ENVIRONMENT failed"
}

# Volume mode: the uv cache lives on the container disk.
choose_volume_cache() {
  local cache=""
  if [[ -n ${UV_CACHE_DIR:-} ]]; then
    # shellcheck disable=SC2310 # normalize_dir / on_volume handle their own failures (|| return 1)
    cache=$(normalize_dir "$UV_CACHE_DIR") || die "cannot resolve UV_CACHE_DIR=$UV_CACHE_DIR"
  fi
  # shellcheck disable=SC2310 # on_volume is a predicate
  if [[ -z $cache ]] || on_volume "$cache"; then
    log "UV_CACHE_DIR=${UV_CACHE_DIR:-(unset)} -> $CONTAINER_UV_CACHE (off the volume)"
    cache=$CONTAINER_UV_CACHE
  fi
  export UV_CACHE_DIR=$cache
}

# Volume mode: libEGL.so.1 loads, installing libegl1 when it does not.
ensure_egl() {
  if [[ ! -x $SYSTEM_PYTHON ]]; then
    log "warning: $SYSTEM_PYTHON is missing; not checking libEGL.so.1 (the preflight egl check will)"
    return 0
  fi
  # shellcheck disable=SC2310 # egl_loads is a predicate that handles its own failures
  if egl_loads; then return 0; fi
  log "libEGL.so.1 cannot be loaded; installing libegl1"
  DEBIAN_FRONTEND=noninteractive apt-get update || die "apt-get update failed (apt-get install libegl1 failed)"
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libegl1 \
    || die "apt-get install libegl1 failed; mjlab cannot import without libEGL.so.1"
  # shellcheck disable=SC2310 # egl_loads is a predicate that handles its own failures
  egl_loads || die "libEGL.so.1 still cannot be loaded after apt-get install libegl1"
}

# Volume mode: login shells get the same uv settings (written to a temp file, then renamed over the old one).
write_profile() {
  local dir=${PROFILE_FILE%/*}
  require_plain_path "$UV_PROJECT_ENVIRONMENT" UV_PROJECT_ENVIRONMENT
  require_plain_path "$UV_CACHE_DIR" UV_CACHE_DIR
  require_plain_path "$UV_PYTHON" UV_PYTHON
  mkdir -p -- "$dir" || die "cannot create $dir"
  profile_tmp=$(mktemp "$PROFILE_FILE.XXXXXX") || die "cannot create a temp file next to $PROFILE_FILE"
  {
    printf '%s\n' "# Written by massively-parallel-rl scripts/setup.sh: login shells use $UV_PROJECT_ENVIRONMENT, never sync."
    printf "export UV_PROJECT_ENVIRONMENT='%s'\n" "$UV_PROJECT_ENVIRONMENT"
    printf '%s\n' "export UV_NO_SYNC=1"
    printf "export UV_CACHE_DIR='%s'\n" "$UV_CACHE_DIR"
    printf "export UV_PYTHON='%s'\n" "$UV_PYTHON"
    printf '%s\n' "export UV_LINK_MODE=copy"
    # shellcheck disable=SC2016 # $PATH is expanded by the login shell, not here
    printf 'case ":$PATH:" in *'"':%s/bin:'"'*) ;; *) PATH='"'%s/bin'"':"$PATH"; export PATH ;; esac\n' \
      "$UV_PROJECT_ENVIRONMENT" "$UV_PROJECT_ENVIRONMENT"
  } >"$profile_tmp" || die "cannot write $profile_tmp"
  chmod 0644 "$profile_tmp" || die "cannot chmod $profile_tmp"
  mv -f -- "$profile_tmp" "$PROFILE_FILE" || die "cannot move $profile_tmp to $PROFILE_FILE"
  profile_tmp=""
  log "wrote $PROFILE_FILE; in this shell run: . $PROFILE_FILE"
}

main() {
  local allow_no_volume=0 arg repo image
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
  if ((allow_no_volume == 0)); then
    choose_volume_env
    choose_volume_cache
    export UV_NO_SYNC=1 UV_LINK_MODE=copy UV_PYTHON=${UV_PYTHON:-$SYSTEM_PYTHON}
    ensure_egl
    create_image_venv
    docker/uv-sync-base.sh || die "docker/uv-sync-base.sh (uv sync --frozen without the base's packages) failed"
    write_profile
  else
    # shellcheck disable=SC2310 # normalize_dir / on_volume handle their own failures (|| return 1)
    image=$(normalize_dir "$IMAGE_VENV") || die "cannot resolve $IMAGE_VENV"
    if [[ -n ${UV_PROJECT_ENVIRONMENT:-} ]]; then
      # shellcheck disable=SC2310 # normalize_dir / on_volume handle their own failures (|| return 1)
      UV_PROJECT_ENVIRONMENT=$(normalize_dir "$UV_PROJECT_ENVIRONMENT") || die "cannot resolve UV_PROJECT_ENVIRONMENT"
      export UV_PROJECT_ENVIRONMENT
    elif [[ -d $image ]]; then
      export UV_PROJECT_ENVIRONMENT=$image
    fi
    if [[ ${UV_PROJECT_ENVIRONMENT:-} == "$image" ]]; then
      docker/uv-sync-base.sh || die "docker/uv-sync-base.sh (uv sync --frozen without the base's packages) failed"
    else
      uv sync --frozen || die "uv sync --frozen failed"
    fi
  fi
  exec uv run --no-sync mprl setup ${forward[@]+"${forward[@]}"}
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
