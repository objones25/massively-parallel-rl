# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Greenfield. As of 2026-10-27 the repo contains only two reference PDFs; no code, environment, or
algorithm has been chosen yet. The **Commands** and **Architecture** sections below are placeholders
to be filled in as decisions are made during brainstorming — update them in the same change that
introduces the thing they describe. Do not document commands that do not exist yet.

## Goal

A demo of **massively parallel reinforcement learning** (thousands of environments stepped in
lockstep on one GPU) for the user's final-project proposal in *Introduction to Robot Learning*
(CS291K). The task is legged locomotion ("learning to walk"), following the approach of Rudin et al.
2021 on **mjlab + rsl_rl** (legacy Isaac Gym was rejected: Python 3.8, PyTorch 1.10, manual
NVIDIA download). The showcase is the **parallelism itself**: the same robot trained at different
numbers of parallel environments, so the audience sees how fast GPU-parallel RL trains a robot.
Package management is **uv** (including inside Docker).

Reference material in the repo root:

- `2109.11978v3.pdf` — Rudin et al. 2021, *Learning to Walk in Minutes Using Massively Parallel
  Deep RL* (legged_gym); the method this project follows.
- `Empirical_Study_of_RL_Scaling.pdf` — Zheng et al. 2026, *An Empirical Study of RL Scaling Laws*
  (arXiv 2610.33420).
- `RL Scaling Laws References and Massively Parallel Training.pdf` — the user's survey of that
  paper's references and of recent massively-parallel RL work.

## Hard requirements

- **Stack:** PyTorch, Docker, pytest, Weights & Biases. Open-source templates/simulators/libraries
  are welcome; no cheap workarounds or stubs standing in for real functionality.
- **Compute budget:** full training on a single **RunPod A100 for under $40 total**. Any config
  choice (num_envs, steps, model size) must come with the wall-clock × hourly-price arithmetic
  showing it fits.
- **Setup script:** after `git clone`, one script prompts for API keys (W&B, Hugging Face, etc.),
  writes `.env` from a committed `.env.example`, and builds/starts the Docker container that
  installs everything. `.env` is gitignored.
- **Train script:** a second script launches training with:
  - **Resume** from the latest checkpoint automatically.
  - **Checkpointing** throughout, written to the **RunPod persistent volume** (survives pod
    restarts), not the container disk.
  - **Automatic upload of weights to the user's Hugging Face account.**
  - **W&B observability:** loss terms, learning rate, gradient norms (global and per-layer),
    activation statistics, attention maps/scores if the model has attention, plus the RL-specific
    metrics best practice calls for (returns, episode length, entropy, KL, clip fraction,
    explained variance, value error, env steps/s, etc.).
- **Video:** if the task has a visual component (it will for a robot), produce rollout videos of
  the policy over the course of training (logged to W&B and kept as artifacts for the demo).

## How to work in this repo

- **No assumptions about APIs.** Libraries in this space (Isaac Lab, mjlab, MuJoCo/MJX/Warp,
  rsl_rl, Brax/Playground, W&B, huggingface_hub, RunPod) change often. Before writing code against
  any library, fetch current docs via Context7 (`ctx7` CLI or the Context7 MCP tools) and/or
  introspect the installed version. Never quote a default, signature, or benchmark number from
  memory.
- **TDD.** Write the failing test first, watch it fail, then implement. Applies to bug fixes too.
- **Skills to use:** `pytest-expert` (test layout, strict config, `audit_tests.py`, mutation
  checks), `pytorch` (training-loop and checkpoint conventions), `massively-parallel-rl`
  (num_envs/batch arithmetic, truncation vs termination, on-device loops, seeds/IQM), and
  `negative-space-programming` (copy its `negative_space.py` helpers into the project; use
  `require`/`expect`/`check_shape`/`check_finite` instead of bare `assert`; bound every loop and
  retry; programmer errors crash, operating errors raise typed exceptions and are handled).
- **DRY and SOLID.** One source of truth for config; components (env, policy, algorithm, logger,
  checkpointer, uploader) behind small interfaces so they can be tested and swapped independently.
- RL correctness rules that are easy to get silently wrong and must be tested:
  bootstrap truncated (time-limit) steps rather than treating them as terminal; keep rollout and
  update on the GPU (no per-step `.item()`/`.cpu()`); state the batch arithmetic
  (N envs × T steps, minibatch size/count, total frames) whenever it changes.

## Commands

Python 3.12, uv only (never pip). torch is pinned to `==2.8.0` and torchvision to `==0.23.0` on every platform, the exact
versions the Docker base ships (see **Docker image**): the cu128 wheel on Linux (`[tool.uv.sources]`, explicit
`pytorch-cu128` index) and the default PyPI wheel on macOS. Bump them only together with the base image tag.

```
uv sync                                              # install (also installs the dev group)
uv run pytest                                        # fast tier only (~10 s): no mjlab import, no training, no coverage
uv run pytest tests/unit/test_clock.py::test_name -v # one test
uv run pytest -m smoke -p no:randomly                # heavy tier (~2 min CPU): real mjlab/rsl_rl, training, W&B offline, ffmpeg
uv run pytest -m smoke tests/smoke/test_render.py::test_render_rollout_cpu   # one smoke test: still needs -m smoke
uv run pytest -m network                             # real W&B / HF / RunPod (needs keys in the environment)
uv run pytest -m gpu                                 # CUDA tests (on the pod; skip cleanly without CUDA)
uv run pytest -p randomly -n auto                    # random order, parallel
uv run pytest -m "not network and not gpu" --cov --cov-branch --cov-report=xml --cov-fail-under=80   # CI: fast + smoke, coverage floor
# (the 80 % floor is for that fast + smoke run; bare `pytest --cov` on the fast tier alone falls below it)
uv tool run --from shellcheck-py shellcheck scripts/*.sh                                             # lint the bash scripts

uv run python scripts/audit_tests.py tests/          # must print "clean: no findings"
uv run python scripts/audit_negative_space.py src/ --select NSP002,NSP003,NSP005,NSP006,NSP007   # must exit 0
uv run ruff check .                                  # lint (PT, bandit, bugbear, ...)
uv run pyright src                                   # strict on src/mprl; third-party unknown-type reports relaxed (see pyproject)
```

Real-pod termination test (`tests/network/test_runpod_terminate.py`, never in CI). Create the cheapest Secure
Cloud GPU pod yourself in the RunPod console, then run it once. The test deletes that pod and confirms with a read
that it is gone (404 or `TERMINATED`); it touches no other pod and skips if either variable is missing:

```
MPRL_THROWAWAY_POD_ID=<pod id> RUNPOD_API_KEY=<key> uv run pytest tests/network/test_runpod_terminate.py -m network
```

On the pod (repo cloned to `/workspace/massively-parallel-rl`, so `.env` persists on the volume). `.env` lives in the
repo root (gitignored, mode 0600); the CLI loads it at startup without overriding variables already set.

```
scripts/setup.sh                                     # refuses unless /workspace is mounted; uv sync; prompts keys -> .env; preflight
scripts/setup.sh --allow-no-volume --cpu             # local testing: no mount check, no CUDA/RunPod checks
scripts/train.sh                                     # = mprl sweep --name main
scripts/train.sh preflight [--cpu] [--only env volume ...]           # PASS/FAIL per check; exit 1 on any FAIL
scripts/train.sh throughput --counts 64 256 1024 4096 16384          # writes $MPRL_VOLUME_ROOT/measurements.json
scripts/train.sh pilot                                               # long run's first 45 min, then terminates the pod
scripts/train.sh sweep --name main --reuse-pilot [--yes]             # --yes accepts budget cuts unattended
scripts/train.sh train-run --sweep dbg --num-envs 64 --seed 0 --minutes 3 --no-terminate
scripts/train.sh results --name main --primary-metric vel_error_xy   # off the clock; --terminate to delete the pod after
```

`train-run`, `pilot` and `sweep` resolve the pod id and build the terminator before training, and refuse
without `RUNPOD_API_KEY` or a pod id unless `--no-terminate` is given. All take `--device` (default `cuda:0`
when available) and `--wandb-mode online|offline|disabled`.

SIGTERM (`kill`, `docker stop`, a RunPod stop) during `train-run`, `pilot`, `sweep` or `results` is a failure exit:
the handler (`publish.sigterm_raises`, installed only while the command runs) raises `Terminated`, so `state.json`
is saved `failed`, the pod is terminated (unless `--no-terminate`; `results` only with `--terminate`) and the exit
status is 1. A repeated SIGTERM during that cleanup is ignored; a `Terminated` swallowed by library code is raised
again at the next iteration end, after training and at the command's end. Ctrl-C at a terminal still keeps the pod.

W&B metrics: rsl_rl's own `Loss/*`, `Policy/*`, `Perf/*` every iteration; every `diagnostics_every_k` (10) iterations,
off the training clock, `mprl.diagnostics` adds `Diag/*`: gradient, weight, update-ratio and activation statistics,
and the PPO health metrics rsl_rl does not return: `Diag/approx_kl` (k3 estimator), `Diag/clip_fraction` and
`Diag/explained_variance` (omitted when the returns are constant), over the whole update (rollout policy -> updated
policy) on at most 16384 strided rollout rows. A non-finite diagnostic is dropped with a warning, never raised.

## Docker image and CI

**Base:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`
(index digest, `docker buildx imagetools inspect`, 2026-10-28; 10.56 GB compressed). It is the image behind RunPod's
default "PyTorch 2.8.0" pod template, so A100 hosts usually have its layers cached and a pod pulls only ours. It
ships Ubuntu 24.04, CUDA 12.8.1, `/usr/bin/python3.12` with torch 2.8.0 + torchvision 0.23.0 (cu128) and their
nvidia-* wheels in system site-packages, uv at `/bin/uv`, ffmpeg and libgl1; its ENTRYPOINT/CMD (`/start.sh`:
sshd, Jupyter) are inherited, so the image runs as root. A 36.9 GB first attempt (own torch on an older base)
was rejected: never install a second torch.

**Pod `minCudaVersion`: 12.8** (the base's `NVIDIA_REQUIRE_CUDA=cuda>=12.8`; also RunPod's documented default
floor for `create-pod`).

**Only what is missing is installed.** `docker/Dockerfile` creates `/opt/venv` with `--system-site-packages` and
syncs it (dev group included: the pod runs `uv run pytest -m gpu`) through `docker/uv-sync-base.sh`, which runs
`uv sync --frozen` with `--no-install-package` for every name in `docker/base-provided-packages.txt` (torch,
torchvision, triton, nvidia-*). `tests/unit/test_base_provided_packages.py` keeps that list equal to those
entries of `uv.lock`; regenerate it after a re-lock if the test fails. uv ignores system site-packages when it
decides what to install, so a plain `uv sync` into `/opt/venv` would download torch again: `scripts/setup.sh`
therefore syncs through the same helper whenever the environment is `/opt/venv`. The build fails
(`docker/verify_image.py` plus a dry-run sync) unless the venv imports the base's own torch 2.8.0, holds none of
the listed packages, imports mjlab / mjlab.tasks / mujoco_warp / rsl_rl / wandb / huggingface_hub / mprl, and the
setup.sh sync would change nothing. apt adds only those of `libegl1 libgl1 libglib2.0-0t64` the base lacks.
`MPRL_IMAGE_DIGEST` (build arg -> env) carries the git sha.

Do not build the image on the Mac (10.5 GB base pull plus amd64 emulation); CI builds it natively. Local checks:

```
python3 ~/.claude/skills/dockerfile-expert/scripts/lint_dockerfile.py --context . --ignore DF004,DF014 docker/Dockerfile
uvx --from actionlint-py actionlint .github/workflows/*.yml
uvx zizmor .github/workflows
uvx --from shellcheck-py shellcheck scripts/*.sh docker/*.sh
```

(DF004: root is required by RunPod's `/start.sh`. DF014: a false positive, the dependency install runs through
`docker/uv-sync-base.sh` before `COPY . .`.)

**CI** (GitHub Actions, actions pinned by commit SHA): `tests.yml` runs the audits, ruff, pyright, shellcheck and
`uv run pytest -m "not network and not gpu" --cov --cov-branch --cov-fail-under=80` (fast + smoke with the
coverage floor; the local default run is fast-only and has no floor). `image.yml` (pushes to `main` and
`feat/**`) calls tests.yml via `workflow_call`, then builds and pushes `ghcr.io/objones25/massively-parallel-rl`
tagged with the git sha (and `latest` on main), with the GHA layer cache. `workflow_run` was not used: it only
fires for workflows on the default branch.

**Pull** (public package): `docker pull ghcr.io/objones25/massively-parallel-rl:<git sha>`; on RunPod set that as
the pod's container image.

## Where tests run

Tiers are set by directory (`tests/conftest.py`): `tests/unit` is the default fast tier; `tests/smoke`,
`tests/network` and `tests/gpu` are marked by name and deselected by default (under the default `-m` they are not
even collected, so the fast tier never imports mjlab; `tests/unit/test_tiers.py` pins that). pytest-timeout: 60 s
per fast test, 600 s per heavy test; the faulthandler exits after 900 s. Fast-tier tests may not start any process
(an autouse guard fails them); heavy tests start processes only through the `subprocesses` fixture
(`support.Subprocesses`: posix_spawn-compatible launch, no stdin, bounded waits, terminate then kill on timeout). The smoke tier
trains for real only where training is the behavior; inputs come from two session fixtures in
`tests/smoke/conftest.py` (`g1_checkpoint`, `completed_run_template`), copied before any mutation. Shared helpers
live in `tests/support.py` (`tests` is on pytest's pythonpath).

`tests/smoke/test_mjlab_cpu.py` pins mjlab 1.6.0 env stepping and rendering on CPU with 4 envs; rsl_rl 5.4.2
`learn`/`load` are pinned through our runner in `tests/smoke/test_timed_runner.py`. Verified 2026-10-27 on macOS
arm64 and Linux x86_64 CPU (`python:3.12` under `--platform linux/amd64`). Linux needs system libraries before
`import mjlab.tasks` works: `libegl1 libgl1 libglib2.0-0` (the plain `python:3.12` image fails at import with
`AttributeError: 'NoneType' object has no attribute 'eglQueryString'` from PyOpenGL). The Dockerfile installs
them. The fast tier needs no system libraries beyond the venv.

## Architecture

*TBD — fill in once the environment, simulator, and algorithm are chosen.*
