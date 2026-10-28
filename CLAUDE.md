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

Python 3.12, uv only (never pip). The torch wheel is CUDA 12.6 on Linux and the default PyPI wheel on macOS (`[tool.uv.sources]`).

```
uv sync                                              # install (also installs the dev group)
uv run pytest                                        # fast tier only (~10 s): no mjlab import, no training, no coverage
uv run pytest tests/unit/test_clock.py::test_name -v # one test
uv run pytest -m smoke -p no:randomly                # heavy tier (~2 min CPU): real mjlab/rsl_rl, training, W&B offline, ffmpeg
uv run pytest -m network                             # real W&B / HF / RunPod (needs keys in the environment)
uv run pytest -m gpu                                 # CUDA tests (on the pod; skip cleanly without CUDA)
uv run pytest -p randomly -n auto                    # random order, parallel
uv run pytest -m "not network and not gpu" --cov --cov-branch --cov-report=xml --cov-fail-under=80   # CI: fast + smoke, coverage floor
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

## Where tests run

Tiers are set by directory (`tests/conftest.py`): `tests/unit` is the default fast tier; `tests/smoke`,
`tests/network` and `tests/gpu` are marked by name and deselected by default (under the default `-m` they are not
even collected, so the fast tier never imports mjlab; `tests/unit/test_tiers.py` pins that). pytest-timeout: 60 s
per fast test, 600 s per heavy test; the faulthandler exits after 900 s. Tests start processes only through the
`subprocesses` fixture (own session, no stdin, bounded wait, group SIGKILLed on timeout/teardown). The smoke tier
trains for real only where training is the behavior; inputs come from two session fixtures in
`tests/smoke/conftest.py` (`g1_checkpoint`, `completed_run_template`), copied before any mutation. Shared helpers
live in `tests/support.py` (`tests` is on pytest's pythonpath).

`tests/smoke/test_mjlab_cpu.py` pins mjlab 1.6.0 env stepping and rendering on CPU with 4 envs; rsl_rl 5.4.2
`learn`/`load` are pinned through our runner in `tests/smoke/test_timed_runner.py`. Verified 2026-10-27 on macOS
arm64 and Linux x86_64 CPU (`python:3.12` under `--platform linux/amd64`). Linux needs system libraries before
`import mjlab.tasks` works: `libegl1 libgl1 libglib2.0-0` (the plain `python:3.12` image fails at import with
`AttributeError: 'NoneType' object has no attribute 'eglQueryString'` from PyOpenGL). The Dockerfile must install
them. The fast tier needs no system libraries beyond the venv.

## Architecture

*TBD — fill in once the environment, simulator, and algorithm are chosen.*
