# syntax=docker/dockerfile:1
# check=error=true
#
# Training image for RunPod pods: RunPod's own PyTorch 2.8.0 image plus only what it lacks.
#
#   docker build --platform linux/amd64 -f docker/Dockerfile \
#     --build-arg MPRL_IMAGE_DIGEST=<git sha> -t ghcr.io/objones25/massively-parallel-rl:<sha> .
#
# Base: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, the image behind RunPod's default "PyTorch 2.8.0" pod
# template, so A100 hosts usually have it cached and a pod pulls only our layers. Pinned by its index digest
# (docker buildx imagetools inspect, 2026-10-28). It ships Ubuntu 24.04, CUDA 12.8.1 (NVIDIA_REQUIRE_CUDA
# cuda>=12.8: the pod's minCudaVersion), Python 3.12 at /usr/bin/python3.12 with torch 2.8.0, torchvision 0.23.0
# (cu128 wheels) and their nvidia-* CUDA wheels in its system site-packages, uv at /bin/uv, ffmpeg and libgl1.
# pyproject.toml pins torch/torchvision to exactly those versions; /opt/venv is created with
# --system-site-packages and docker/uv-sync-base.sh skips them (docker/base-provided-packages.txt), so nothing
# from the base is downloaded again. RunPod's ENTRYPOINT (/opt/nvidia/nvidia_entrypoint.sh) and CMD (/start.sh:
# sshd, Jupyter, nginx) are inherited unchanged. The image runs as root because /start.sh needs root for sshd
# and the pod's /workspace volume.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# EGL for MuJoCo headless rendering (mjlab sets MUJOCO_GL=egl; PyOpenGL loads libEGL at `import mjlab.tasks`).
# Only the packages the base lacks are installed; the build log names which ones that was. The NVIDIA EGL
# vendor file lets libglvnd find the driver's libEGL_nvidia.so.0, which the container runtime injects when the
# "graphics" driver capability is requested (NVIDIA_DRIVER_CAPABILITIES below).
RUN <<'EOF'
set -euo pipefail
wanted=(libegl1 libgl1 libglib2.0-0t64)
missing=()
for pkg in "${wanted[@]}"; do
  if [[ $(dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null || true) != "install ok installed" ]]; then missing+=("$pkg"); fi
done
echo "system libraries already in the base: ${wanted[*]}; missing: ${missing[*]:-none}"
if ((${#missing[@]} > 0)); then
  apt-get update
  apt-get install -y --no-install-recommends "${missing[@]}"
  rm -rf /var/lib/apt/lists/*
fi
vendor=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
if [[ ! -e $vendor ]]; then
  mkdir -p "${vendor%/*}"
  printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' >"$vendor"
fi
EOF

# The environment lives outside /workspace (the network volume mount would hide baked files). uv uses the
# base's Python 3.12 and never downloads another. The base sets UV_CACHE_DIR to the volume for pods; builds
# use a cache mount instead.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PATH=/opt/venv/bin:$PATH

WORKDIR /opt/mprl

# Third-party dependencies from the lockfile only, WITH the dev group (the pod runs `uv run pytest -m gpu`),
# minus everything the base already provides.
COPY pyproject.toml uv.lock ./
COPY docker/uv-sync-base.sh docker/base-provided-packages.txt docker/
RUN --mount=type=cache,target=/tmp/uv-cache <<'EOF'
set -euo pipefail
uv --version
uv venv --system-site-packages /opt/venv
UV_CACHE_DIR=/tmp/uv-cache docker/uv-sync-base.sh --no-install-project
EOF

# Then the source and the project itself.
COPY . .
RUN --mount=type=cache,target=/tmp/uv-cache UV_CACHE_DIR=/tmp/uv-cache docker/uv-sync-base.sh

# Fail the build unless /opt/venv runs the base's torch 2.8.0, holds none of the base's packages, and imports
# the training stack; and unless the pod's scripts/setup.sh sync (docker/uv-sync-base.sh) would change nothing.
# The plain `uv sync` dry run is printed only to show what it would have re-downloaded.
RUN --mount=type=cache,target=/tmp/uv-cache <<'EOF'
set -euo pipefail
base_torch=$(/usr/bin/python3.12 -c 'import torch; print(torch.__file__)')
/opt/venv/bin/python docker/verify_image.py "$base_torch"
echo "--- plain 'uv sync --frozen --dry-run' (what setup.sh avoids):"
UV_CACHE_DIR=/tmp/uv-cache uv sync --frozen --dry-run 2>&1 | sed 's/^/    /'
plan=$(UV_CACHE_DIR=/tmp/uv-cache docker/uv-sync-base.sh --dry-run 2>&1)
printf '%s\n' "--- docker/uv-sync-base.sh --dry-run (what setup.sh runs):" "$plan"
if grep -Eq '^ [-+~] ' <<<"$plan"; then
  echo "verify: FAIL: the pod's setup.sh sync would change /opt/venv" >&2
  exit 1
fi
EOF

# Stamped last so a new value only invalidates this layer. CI passes the git commit sha: the pushed image
# digest does not exist until after the build, so the sha is the identity the image can carry.
ARG MPRL_IMAGE_DIGEST=unknown
ENV MPRL_IMAGE_DIGEST=${MPRL_IMAGE_DIGEST}
