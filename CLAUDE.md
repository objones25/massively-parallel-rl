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

*TBD — fill in once the project is scaffolded (setup script, train script, how to run the full
test suite, a single test, lint, type check, and the negative-space / test audits).*

## Architecture

*TBD — fill in once the environment, simulator, and algorithm are chosen.*
