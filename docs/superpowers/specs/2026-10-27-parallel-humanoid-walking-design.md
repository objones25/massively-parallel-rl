# Massively Parallel Humanoid Walking — Design

Date: 2026-10-27. Status: draft for approval.

Decisions are marked **(user)** when the user made them and **(call)** when they are the designer's
default, which the user may override.

## 1. Intent

This is the user's final-project proposal for CS291K (Introduction to Robot Learning). The audience
(classmates and instructor) should leave understanding **how insanely fast a single GPU can train
a robot to walk when thousands of simulated copies learn in parallel**. The showcase is the
parallelism itself, not a polished policy **(user)**.

The project reproduces the *approach* of Rudin et al. 2021, *Learning to Walk in Minutes Using
Massively Parallel Deep RL* (`2109.11978v3.pdf`, the paper's Fig. 4 in particular) on modern,
maintained tooling. Results should be available to cite in the proposal but are not required for
it; the repo is the base for a project that will grow afterwards **(user)**.

## 2. Goals, non-goals, anti-goals

**Goals**

1. Train the **Unitree G1 humanoid** to walk **(user)** with mjlab + rsl_rl **(user)** at several
   numbers of parallel environments, each given the **same 45 minutes of training time on one
   A100** **(user)**.
2. Produce the two headline outputs **(user)**:
   - **Race video**: one video, one panel per environment count, a shared training clock; each
     panel shows that run's policy as it was at the clock time.
   - **Learning-curve chart**: walking quality vs. wall-clock minutes, one line per environment
     count, spread across repeats shaded.
3. One-command setup on the pod and one-command training with automatic resume, checkpoints on
   the persistent volume, weights and results uploaded to Hugging Face, and full W&B observability
   **(user)**.
4. Every result (metrics, charts, videos, checkpoints) is saved to the network volume, W&B and
   Hugging Face so it survives the pod **(user)**.

**Non-goals**

- Rough terrain, stairs, the paper's terrain curriculum, sim-to-real, real hardware. Flat ground
  only for this phase **(call)**: a humanoid on rough terrain within 45 minutes is unlikely, and the
  comparison needs a task the large-N runs can solve inside the budget. Rough terrain is a natural
  later expansion.
- Multi-GPU training. Isaac Gym / legged_gym itself (rejected: Python 3.8, PyTorch 1.10, manual
  NVIDIA download) **(user)**.
- A "swarm" clip of many robots in one frame (mjlab's offscreen renderer is capped at 32
  environments, which undersells the parallelism) **(user)**.

**Anti-goals** (failures even if it "works")

- Anything faked: stubbed training, placeholder videos, mocks that do not mirror the real API,
  numbers quoted from memory instead of measured or read from docs **(user)**.
- An unfair comparison: runs that differ in anything other than environment count without saying
  so, or a clock that charges I/O to some runs more than others.
- Losing results when the pod stops.
- Exceeding $40.

## 3. Constraints

- **Stack (user):** PyTorch, Docker, pytest, Weights & Biases, Hugging Face Hub; **uv** for all
  package management including in the image. Simulator: mjlab (Isaac Lab's manager-based API on
  MuJoCo Warp; trains with rsl_rl).
- **Process (user):** TDD; DRY and SOLID; negative-space programming (copy the skill's
  `negative_space.py` helpers; `require`/`expect`/`check_shape`/`check_finite`; bound every loop
  and retry). Look up every library API in current docs (Context7) or by introspecting the
  installed version — never from memory. **Smoke test each integration against the real library
  or service before writing mocks; mocks must autospec the real API.**
- **Compute (user):** one RunPod A100 80GB, total under $40.
  - Prices read from RunPod's `list-gpu-types` on 2026-10-27: A100 80GB PCIe and SXM, Secure Cloud,
    **$1.79/h**. Availability was "LOW".
  - RunPod network volumes attach **only to Secure Cloud pods** (RunPod concept docs), so Secure
    Cloud pricing applies. Volume storage is billed per GB-hour from creation to deletion; read the
    rate before creating it and size the volume small.
  - Pods mount the network volume at a path that must be given at creation; `/workspace` is the
    convention. Network storage is slow for many small files: the Python environment must be baked
    into the image outside `/workspace` (e.g. `/opt/venv`).
- **Deployment shape (user):**
  - A **GitHub Action builds the Docker image and pushes it to a registry** (GHCR **(call)**), so
    the user's Mac never builds it. No secrets in the image.
  - The user creates a Secure Cloud A100 pod from that image with the network volume at
    `/workspace`, then **runs the setup script on the pod**. A RunPod pod is itself a container, so
    nothing on the pod runs `docker`. "Setup sets up the container" therefore means: everything is
    already installed in the image; the setup script configures and verifies it.
  - Custom images must not clobber the RunPod base image's startup (SSH / web terminal) — inherit
    the base's start command or use its hooks. Verify against the chosen base's docs.
  - **The GitHub repo is public (user)**, so the GHCR image is public. RunPod pulls it and the
    pod clones the repo over HTTPS with no registry login and no GitHub token.
- **Auto-terminate (user: yes):** when the sweep (or a single run) finishes or fails, the scripts
  delete the pod they are running on through the RunPod API, so an idle A100 never bills.
  - This is safe because everything kept lives on the network volume. The volume itself is never
    deleted by the scripts.
  - Setup prompts for a RunPod API key.
  - The script deletes only its own pod: it identifies the pod from the environment RunPod sets
    inside the pod (verify the variable name in RunPod's docs) and refuses to act if that is
    missing.
  - Before deleting, it waits until the uploads have finished or have failed and been recorded.
  - A `--no-terminate` flag exists for interactive debugging sessions.
- **Testing location (user):** almost all tests run on CPU (the user's Mac and CI). GPU time on the
  paid pod is reserved for the few smoke tests that cannot run on CPU, batched into as few pod
  sessions as possible.
- **Deadline (user):** the proposal is due Tuesday 2026-11-03. Results are citable but not
  required for it.

## 4. The experiment

**Fixed across all runs (call):** task `Mjlab-Velocity-Flat-Unitree-G1` (name from mjlab docs;
verify against installed version), mjlab's shipped G1 PPO config (actor/critic MLP 512-256-128
ELU, observation normalization, 24 steps per env per iteration, 5 epochs, 4 minibatches, adaptive
LR with desired KL 0.01, max grad norm 1.0), same hardware. **Only `num_envs` changes.** Because
the number of minibatches is held fixed, the batch and minibatch size grow with `num_envs`; that is
the effect being demonstrated and the write-up must say so explicitly, with the batch arithmetic
(N, T=24, N×T, minibatch size, iterations reached in 45 min, total env steps) per run.

**Step-count schedules in the shipped task (call):** mjlab curricula (`commands_vel`,
`reward_curriculum`) advance on a training step threshold. Whether the G1 flat task ships any must
be checked in the installed version. If it does, and the counter advances per iteration
independent of `num_envs`, then small-N runs, which complete more iterations per minute, reach
later stages sooner in wall-clock time. That favors the small-N runs, so the measured speed-up is
if anything understated. Keep the schedules as shipped and disclose them, and log every
`Curriculum/*` value so the stage each run reached is visible. If the counter turns out to scale
with `num_envs`, stop and raise it with the user before the sweep.

**Environment counts (call):** 64, 256, 1024, 4096, 16384. The throughput test (below) may drop
16384 if it does not fit in memory or saturates; that decision is recorded with its measurement.

**Repeats (call):** 3 seeds per count; charts show mean (or IQM) with the spread shaded.

**The 45-minute clock (user: 45 min; call: what it counts):** the budget counts **training time
only — rollout plus PPO update**. Checkpoint writes, logging, evaluation, rendering and uploads are
excluded from the clock and timed separately, so slow network storage cannot penalize the
small-N runs (which run more iterations and checkpoint more often). The elapsed training time is
part of the checkpoint, so a resumed run continues the same clock rather than restarting it.

**Checkpoints:** saved every **3 minutes of training-clock time (call)** — this is what the race
video and the learning curve are built from — plus the final one. Kept for the whole run on the
volume.

**Walking quality (call):** measured by an offline evaluator, not by training reward, so all runs
are scored identically: each checkpoint is run for a fixed horizon on a fixed set of velocity
commands and seeds, with the same number of evaluation environments regardless of the training N.
Reported: velocity-tracking error, fraction of episodes without a fall, mean episode length, and
the task's tracking reward. The learning curve's y-axis is the primary one of these; which one is
primary is decided once real numbers exist (call, recorded in the results).

**Race video:** for each count, pick the median-scoring seed; render its checkpoint at each
3-minute mark on the same commands and camera; compose the panels side by side with the training
clock and env count overlaid. One panel per count.

**Long run (call):** one longer run at 4096 environments for the best-looking walking video and the
uploaded "headline" model.

## 5. Budget (call; Secure Cloud $1.79/h)

| Item | Hours | Cost |
| --- | --- | --- |
| Throughput test: G1 env steps/s at each count (~15 min) | 0.25 | $0.45 |
| Comparison: 5 counts × 3 seeds × 45 min | 11.25 | $20.14 |
| Long run at 4096 envs | 3 | $5.37 |
| Debugging, smoke tests, evaluation and rendering | 4 | $7.16 |
| **Total GPU** | **18.5** | **$33.12** |

Plus network-volume storage (rate to be read before creation). The off-clock overhead (checkpoint
writes, evaluation, rendering) adds GPU time on top of the 45-minute training clocks; the
throughput test and a short pilot measure it, and the plan is re-costed before the full sweep.
The sweep must refuse to start if the re-costed total exceeds $40.

**Pilot gate:** before the full sweep, a single 45-minute run at 4096 envs confirms the G1 walks
within the budget. If it does not, hours move from repeats into run length (e.g. 2 seeds × longer
runs) and the user is told before money is spent.

**The pilot is not extra spend (call):** the pilot is the first 45 minutes of the long run, which
later resumes to 3 hours. If code and config are frozen at the pilot, its 45-minute state also
counts as seed 0 of the 4096 sweep; otherwise it does not, and the sweep runs all 3 seeds.

**Cut order when the re-costed plan exceeds $40 (call):** the sweep script reports which cuts it
would apply, in this order, and the user confirms:

1. Evaluate and render every 6 minutes of training clock instead of every 3. Checkpoints are
   still saved every 3 minutes.
2. Shorten the long run from 3 h to 1.5 h.
3. Drop 16384 envs.
4. Go from 3 seeds to 2.

The 64-env count is never dropped; it is the contrast the demo exists to show.

## 6. Setup and training flow

1. **CI (GitHub Action):** on push, build the image (uv-managed environment baked into a path
   outside `/workspace`) and push to GHCR. Tests that do not need a GPU run in CI first.
2. **Pod:** user creates a Secure Cloud A100 pod from the image with the network volume at
   `/workspace`, clones the repo (or uses the copy in the image — builder's call, documented).
3. **Setup script (on the pod):** prompts for keys (W&B, Hugging Face, RunPod), writes
   `.env` from the committed `.env.example` (`.env` gitignored, stored where it persists across
   pods), then verifies: GPU visible to PyTorch and MuJoCo Warp, mjlab task constructs and steps,
   W&B auth, HF auth and target repo access, RunPod API key valid and the pod's own id resolvable,
   network volume writable. Each check prints pass/fail; re-running it is safe.
4. **Train script:** launches a single run or the whole sweep. Resumes automatically from the
   latest checkpoint on the volume for each run (including the training clock and W&B run id, so
   the W&B run continues rather than forking). Completed runs are skipped. A run that finishes
   triggers evaluation, upload and (once all runs of a count exist) results generation.
5. **Results:** a command (separate from training) that builds the learning-curve chart, the race
   video and a results CSV from the checkpoints on the volume, and logs/uploads them.

## 7. Observability (W&B)

Policy and critic are MLPs; there is **no attention**, so attention maps are not applicable and
must not be invented. Logged per iteration (or every K iterations where noted):

- **Optimization:** surrogate loss, value loss, entropy, approximate KL, clip fraction, learning
  rate (adaptive schedule makes this informative), global gradient norm before and after clipping,
  per-layer gradient norms, per-layer weight norms and update-to-weight ratio.
- **Activations (every K):** per-layer mean, std and fraction of near-saturated / near-dead ELU
  units for actor and critic; action distribution std; observation-normalizer mean/std drift.
- **Value function:** explained variance, value-prediction error, return and advantage statistics.
- **RL / task:** episode return, episode length, each reward term separately, termination reasons
  (fall vs time-out — truncation must be bootstrapped, not treated as terminal; verify rsl_rl/mjlab
  behavior), velocity-tracking error.
- **Systems:** env steps/s, iteration time split into rollout / update, training-clock minutes,
  GPU utilization and memory, checkpoint write time.
- **Run config:** full resolved config, num_envs, batch arithmetic, git SHA, image digest, library
  versions.
- **Media:** evaluation videos for the long run at checkpoints; final race video and chart.

Group W&B runs by sweep and environment count so the chart can also be reproduced in W&B.

## 8. Storage and uploads

On the network volume, one directory per run (sweep / env count / seed) holding checkpoints,
training-clock state, evaluation results and videos; one results directory per sweep. Hugging Face
(private repo by default **(call)**): final checkpoint of every run, the long-run model, and the
results (chart, race video, CSV) with a model card describing the setup. Uploads retry with a
bound and are idempotent; an upload failure must not lose or corrupt the local checkpoint, and is
reported, not swallowed.

## 9. Testing

- TDD throughout. Pure logic (clock accounting, checkpoint discovery and resume, batch arithmetic,
  budget calculation, results aggregation, video composition layout) gets fast CPU unit tests.
- **Smoke tests first** for every integration, against the real thing, small and marked. Only
  after a smoke test passes may unit tests use doubles, and those must autospec the installed
  library.
  - **On CPU wherever the library supports it (user: prefer CPU):** mjlab environment
    construct-and-step at tiny `num_envs`, a few rsl_rl iterations, a real checkpoint
    save→kill→resume round trip, W&B offline mode and a real run-id resume, a real upload to a
    throwaway private HF repo, an offscreen render of a few frames, and the evaluator and
    video/chart pipeline on CPU-produced checkpoints.
  - **Platform support:** mjlab documents macOS as evaluation-only on CPU and training as Linux +
    NVIDIA. The first task of the plan is to establish, by running them, which of these smoke
    tests work on the Mac, in a Linux CPU container (the CI image without a GPU) and only on a
    GPU. Record the result in CLAUDE.md.
  - **GPU-only smoke tests** are marked, kept minimal, and run in one batched pod session before
    the pilot.
  - **Pod deletion** is smoke-tested once on a real throwaway pod: the cheapest Secure Cloud GPU,
    created by the user, and the test confirms with a read that the pod is gone. Unit tests use an
    autospecced client.
- Tests verify behavior that matters for honesty: the clock excludes I/O, resume continues the
  clock, only `num_envs` differs between sweep runs, the evaluator uses identical commands/seeds.

## 10. Build order against the 2026-11-03 deadline (call)

1. Scaffold, CPU smoke tests, and the CI image build.
2. Training with the clock, time-based checkpoints, resume, W&B, the HF upload and auto-terminate.
   This is everything a paid run depends on.
3. One pod session: the GPU smoke tests, the throughput test, then the pilot. Its result is the
   number the proposal cites.
4. The evaluator, race video and chart, built and tested on CPU against the pilot's real
   checkpoints. They can be built while the sweep runs, because every checkpoint is kept.
5. The sweep and the long run. These may finish after the proposal; the proposal cites the pilot
   and whatever sweep results exist by then.

## 11. Left to the builder

Package and module layout; exact script names; how the training clock and time-based checkpointing
hook into rsl_rl's runner (verify the runner's extension points in the installed version); RunPod
base image tag; whether the repo is cloned onto the volume or used from the image; video
composition library; exact W&B panel layout; K for every-K logging.

## 12. Risks

- The G1 may not walk within 45 minutes at any count — the pilot gate (section 5) catches this.
- 16384 environments may not fit or may saturate the A100 — the throughput test decides.
- A100 Secure Cloud availability was "LOW" on 2026-10-27; the network volume pins a data center,
  so pick one that lists A100 stock when creating it.
- mjlab, rsl_rl and RunPod APIs move; every claim in this document about them must be re-checked
  against the installed versions before code depends on it.
