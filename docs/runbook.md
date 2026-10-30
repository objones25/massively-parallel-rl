# Runbook: the paid RunPod session

Everything here costs money only where it says **Paid**. Prices and stock below were read on **2026-10-28** from
RunPod's catalog (`list-gpu-types`, `list-data-centers`) and pricing docs; re-read them before you create anything.
Every command exists in this repo or in `runpodctl` 2.9.0 (`runpodctl <cmd> --help` checked). Shell blocks are meant
to be pasted; `<...>` are values you fill in.

The plan is three pod sessions on one network volume. Each session ends with the pod deleting itself.

| Session | Commands | Ends with |
| --- | --- | --- |
| A | setup, GPU tests, render and network tests, throughput, pilot | `pilot` terminates the pod |
| (free) | pilot gate on your Mac and in W&B | your go / no-go |
| B | setup, sweep (14 or 15 runs, then the long run to 180 min) | `sweep` terminates the pod |
| C | setup, results | `results --terminate` terminates the pod |

## 1. Budget

Spec §5 (Secure Cloud A100 80GB at **$1.79/h**, read 2026-10-27 and again 2026-10-28):

| Item | Hours | Cost |
| --- | --- | --- |
| Throughput test: G1 env steps/s at each count (~15 min) | 0.25 | $0.45 |
| Comparison: 5 counts × 3 seeds × 45 min | 11.25 | $20.14 |
| Long run at 4096 envs | 3 | $5.37 |
| Debugging, smoke tests, evaluation and rendering | 4 | $7.16 |
| **Total GPU** | **18.5** | **$33.12** |

The cap is **$40**, which is 22.3 GPU-hours at $1.79/h. The off-clock overhead (checkpoint writes, logging,
diagnostics) multiplies every training hour by `1 + io_overhead_fraction`, which the throughput test measures.
`mprl sweep` re-costs the whole project from `measurements.json` (formula in `src/mprl/budget.py`). If the total
is over $40 it applies cuts in this order and asks you to confirm:

1. `eval_every_6min`: evaluate and render every 6 training minutes instead of every 3 (checkpoints stay every 3).
2. `long_run_half`: long run 3 h becomes 1.5 h.
3. `drop_16384`: drop the 16384-env count.
4. `two_seeds`: 3 seeds become 2.

The 64-env count is never dropped. If every cut is not enough, the sweep refuses to start (`BudgetExceeded`).

The price defaults to $1.79/h (`PRICE_PER_HOUR` in `src/mprl/budget.py`). If the A100 price you read differs, pass
it on the command line, `--price-per-hour <price>`, to both `pilot` (its printed cost) and `sweep` (the estimate and
the $40 cap check). Never edit the code instead: a code change after the pilot changes the git sha, and then the
pilot can no longer stand in for a sweep run.

Running total by step: the expected spend at $1.79/h, before the measured overhead. The sweep's printed
estimate is the cap check, and it is deliberately higher than rows B and C (section 6 says by how much).

| Step | Hours | Cost | Running total |
| --- | --- | --- | --- |
| A: pod start, setup, GPU / render / network tests | 0.3 | $0.54 | $0.54 |
| A: throughput test | 0.25 | $0.45 | $0.99 |
| A: pilot (45 min training clock plus env build and overhead) | 0.85 | $1.52 | $2.51 |
| B: pod start, setup | 0.1 | $0.18 | $2.69 |
| B: sweep, 14 runs × 0.75 h (the pilot stands in for the 15th) | 10.5 | $18.80 | $21.49 |
| B: long run, 135 of 180 min left after the pilot | 2.25 | $4.03 | $25.52 |
| C: pod start, setup, results (225 evaluations × measured seconds, race video) | ~2 | ~$3.58 | ~$29.10 |
| Reserve for debugging and re-runs | ~2.6 | ~$4.65 | $33.75 |

Plus the network volume: 10 GB × $0.07/GB/month (standard tier under 1 TB, docs.runpod.io/pods/pricing, read
2026-10-28) = $0.70/month, billed hourly from creation until you delete it.

## 2. One-time prerequisites (free)

**Keys.** `scripts/setup.sh` asks for these and writes them to `.env` on the volume (mode 0600):

- `RUNPOD_API_KEY`: console, Settings, API Keys. Pick permission **All**. The pod reads (`GET /v2/pods/{id}`) and
  deletes (`DELETE`) itself; Read Only cannot delete, and Restricted only scopes Serverless endpoints
  (docs.runpod.io/get-started/credentials).
- `WANDB_API_KEY` (wandb.ai/authorize), optional `WANDB_ENTITY`, `WANDB_PROJECT` (default `mprl-humanoid-walk`).
- `HF_TOKEN` with **write** access (huggingface.co/settings/tokens) and `HF_REPO_ID` as `<user>/<name>`, for
  example `objones25/mprl-g1-walk`. The repo is created private if it does not exist.
- `MPRL_VOLUME_ROOT`: keep the default `/workspace/mprl`.

**SSH key and runpodctl on your Mac.** `runpodctl doctor` stores your API key for the CLI.

```
runpodctl doctor
runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub
runpodctl ssh list-keys
```

**Image.** `ghcr.io/objones25/massively-parallel-rl:<sha>` is public (an anonymous manifest read returned 200 on
2026-10-28). `latest` exists only on `main`; until the branch is merged, use the sha of the newest successful
`image` workflow run on the branch:

```
gh run list --workflow image.yml --branch feat/parallel-humanoid-walking --status success --limit 1 --json headSha --jq '.[0].headSha'
```

**Data center.** A network volume lives in one data center and only Secure Cloud pods can mount it, so pick a data
center that offers network volumes and has A100 80GB Secure Cloud stock *now*:

```
runpodctl gpu list
runpodctl datacenter list
```

On 2026-10-28 the intersection was: A100 SXM 80GB in EU-RO-1, EUR-IS-1, US-KS-2, US-WA-1; A100 PCIe 80GB in
CA-MTL-3, US-KS-2 (all "LOW" stock; US-MD-1 had A100 SXM but offers no network volumes). **US-KS-2** had both
A100 types, so it is the default below. Both types were $1.79/h Secure Cloud.

**Network volume size: 10 GB.** The G1 policy and critic are MLPs 512-256-128 with 99 and 111 inputs and 29
actions: actor 219,165 + 29 (std) = 219,194 parameters, critic 221,697, total 440,891. A checkpoint holds the
weights plus Adam's two moment buffers: 3 × 440,891 × 4 B = 5.3 MB. Checkpoints are kept every 3 training minutes:
15 runs × 15 + long run 60 + throughput 5 = 290 checkpoints × 5.3 MB = 1.5 GB. Add ONNX exports (one per run,
under 1 MB), local W&B files under the repo's `wandb/`, evaluation CSVs, videos and the 3 MB repo: about 3 GB at
the outside, so 10 GB leaves 3× headroom ($0.70/month; 20 GB would be $1.40/month). A volume can grow later, never
shrink. Read the current per-GB price on the console's Storage page before you create it.

```
runpodctl network-volume create --name mprl --size 10 --data-center-id US-KS-2
runpodctl network-volume list
```

## 3. Create a pod (Paid: $1.79/h from the moment it starts)

Settings for every session (console or the command below):

- Cloud: **Secure**. GPU: `NVIDIA A100-SXM4-80GB` or `NVIDIA A100 80GB PCIe`, count 1.
- Data center: the volume's. Network volume: the one above, mounted at **`/workspace`**.
- Image: `ghcr.io/objones25/massively-parallel-rl:<sha>`. Keep its start command (RunPod's `/start.sh` runs sshd).
- **`minCudaVersion` 12.8.** The base `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` is CUDA 12.8.1
  (`NVIDIA_REQUIRE_CUDA=cuda>=12.8`), and RunPod's GPU-selection guide names exactly this image as the reason for
  its default floor of 12.8. Use the floor, not an exact CUDA list: on 2026-10-28 the A100 hosts with stock
  reported CUDA 13.2, which the floor admits and an exact `12.8` would exclude.
- Container disk: 30 GB (the writable layer: `/tmp`, `~/.cache` for Warp kernels; $0.10/GB/month while running,
  about $0.004/h).
- Ports: `22/tcp` (SSH), `8888/http` (Jupyter, optional).
- Environment variables: **none**. RunPod sets `RUNPOD_POD_ID` itself; the keys come from `.env` on the volume.
- `--terminate-after` is a backstop against a forgotten pod: A 3 h, B from the estimate (section 6; about 18 h), C 4 h. If it fires mid-run,
  nothing is lost; resume as in section 8.

```
runpodctl pod create --name mprl-a --cloud-type SECURE --gpu-id "NVIDIA A100-SXM4-80GB" --gpu-count 1 --data-center-ids US-KS-2 --network-volume-id <volume id> --volume-mount-path /workspace --image ghcr.io/objones25/massively-parallel-rl:<sha> --min-cuda-version 12.8 --container-disk-in-gb 30 --ports "22/tcp,8888/http" --terminate-after "$(date -u -v+3H +%Y-%m-%dT%H:%M:%SZ)"
runpodctl ssh info <pod id>
```

`date -u -v+3H` is the macOS (BSD) form. On Linux (GNU date) write the deadline as
`"$(date -u -d '+3 hours' +%Y-%m-%dT%H:%M:%SZ)"`.

Then SSH in with the printed command. Run long commands inside `tmux`, so a dropped SSH connection does not kill
training (SIGHUP would end the process without terminating the pod, and the idle pod would keep billing). The
install line is a no-op if the image already has it; otherwise it installs Ubuntu 24.04's `tmux` package (3.4 in
noble, checked on Launchpad 2026-10-28). The pod's web terminal is an alternative for short sessions, but it also
dies with the browser tab, so prefer tmux for the pilot and the sweep:

```
command -v tmux || (apt-get update && apt-get install -y tmux)
tmux new -s mprl
```

Detach with Ctrl-B then D; reattach with `tmux attach -t mprl`.

## 4. Session A (Paid, about 1.4 h, running total about $2.51)

**Clone and set up** (once; the clone lives on the volume and is reused by sessions B and C):

```
cd /workspace
git clone https://github.com/objones25/massively-parallel-rl.git
cd massively-parallel-rl
git checkout feat/parallel-humanoid-walking
git log -1 --format=%H
scripts/setup.sh
```

`setup.sh` refuses unless `/workspace` is a mount point, re-syncs the image's `/opt/venv` through
`docker/uv-sync-base.sh` (this installs the clone's `mprl` in editable mode and skips torch / triton / nvidia-*, which
the base provides), prompts for every key in `.env.example` (secrets without echo; Enter keeps the current value),
writes `.env`, then runs the preflight. Every line must say PASS:

- `cuda`: PyTorch and Warp see the A100 and Warp allocates on `cuda:0`.
- `env`: the G1 task builds with 2 envs on `cuda:0` and steps 4 times with finite rewards.
- `wandb`: the key is accepted; prints the entity and project.
- `hf`: the token is accepted and `HF_REPO_ID` exists (created private if absent).
- `python-env`: torch is the base's (not under `/workspace` or the repo `.venv`), the venv is off the volume,
  `UV_CACHE_DIR` is off the volume and `UV_NO_SYNC` is set.
- `egl`: `libEGL.so.1` loads (MuJoCo's headless rendering for the videos).
- `runpod`: the pod id resolves (`RUNPOD_POD_ID`, else `/proc/1/environ`) and a GET of this pod returns 200.
- `volume`: the volume root is on a mount other than `/`, and a probe file under `/workspace/mprl` is written, read
  back and deleted.

`hf` also fails a read-only token. An `ImportError`, `AttributeError` or `OSError` inside a check is a FAIL line for
that check, not a traceback.

**Stock image.** The project image is the intended one, but a pod from RunPod's stock PyTorch 2.8.0 image
(`runpod/pytorch:*-cu1281-torch280-ubuntu2404`) also works: `setup.sh` creates `/opt/venv` with
`--system-site-packages` from `/usr/bin/python3.12` (whose site-packages hold torch), moves `UV_CACHE_DIR` off the
volume (the stock image points it at `/workspace/.cache/uv/`), installs `libegl1` if `libEGL.so.1` does not load,
and writes `/etc/profile.d/mprl-uv.sh` so later SSH / tmux login shells use `/opt/venv` and never sync. In the shell
you ran `setup.sh` from, run `. /etc/profile.d/mprl-uv.sh` once. It never installs onto the volume: a
`UV_PROJECT_ENVIRONMENT` under `/workspace` is refused, and without torch in the system python it refuses and names
the project image. (2026-10-30: a stock pod's old fallback re-downloaded torch into a `.venv` on the 30 GB volume
and hit the quota mid-install. `df /workspace` shows the whole storage cluster, not your quota.) `setup.sh` needs a
TTY (it prompts for keys).

Re-run only some checks with `scripts/train.sh preflight --only cuda env`.

**GPU tests.** The image sets `UV_NO_SYNC=1`: a plain `uv run` would otherwise sync `/opt/venv` and download the
base's torch again. `--no-sync` says the same explicitly.

```
uv run --no-sync pytest -m gpu -v
```

Expected: 3 passed. `test_cuda_clock.py` (the clock calls `torch.cuda.synchronize` before each start and stop
time), `test_gpu_smoke.py::test_cuda_training_iteration` (2 PPO iterations at 4096 envs, finite losses),
`test_gpu_smoke.py::test_checkpoint_round_trip_on_volume` (save under `/workspace/mprl`, load into a fresh runner).

**EGL rendering** (the race video and evaluation videos need it; never verified on RunPod):

```
uv run --no-sync pytest -m smoke tests/smoke/test_render.py tests/smoke/test_mjlab_cpu.py::test_play_env_renders_rgb_frame -v --no-cov
```

A failure here is fixed before results (session C), not before training: training never renders.

**Network tests** (never run yet). Tests marked `network` read the keys from the environment (`tests/conftest.py`
strips `WANDB_*`, `HF_*` and `RUNPOD_*` only from the other tiers), so export `.env` into the shell. The first line
parses `.env` with python-dotenv, the same loader the CLI uses, and prints shell-quoted `export` lines, so values
containing `$`, spaces or `#` survive. `RUNPOD_POD_ID` is only in PID 1's environment, so the second line exports
it from there:

```
eval "$(uv run --no-sync python -c 'import shlex, sys; from dotenv import dotenv_values; [print("export " + k + "=" + shlex.quote(v or "")) for k, v in dotenv_values(sys.argv[1]).items()]' .env)"
export RUNPOD_POD_ID=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')
uv run --no-sync pytest -m network -v --no-cov --deselect tests/network/test_runpod_terminate.py
```

`test_wandb_resume.py` tests resuming a finished W&B run by id and the offline fallback against the real service
(project `mprl-tests`); `test_hub_real.py` creates a private `<you>/mprl-smoke-test` repo. **Never skip this tier
before a paid session:** on 2026-10-29 it was skipped (no `WANDB_API_KEY` in the shell), and the long run then died
at `wandb.init` because the server rejected `resume_from` (W&B Rewind is a private preview), which every
autospecced unit test had accepted. If a network test fails, stop and fix it before training (section 9). From the
Mac, `WANDB_API_KEY` can come from `~/.netrc` (machine `api.forge.coreweave.com`, written by `wandb login`).

**Throughput test** (about 15 min):

```
scripts/train.sh throughput --counts 64 256 1024 4096 16384
cat /workspace/mprl/measurements.json
```

It trains 20 iterations per count and records env steps/s, whether the count fits (a CUDA
`torch.OutOfMemoryError` is recorded as `false`, not raised), the off-clock overhead fraction and the seconds one
checkpoint evaluation takes. If 16384 crashes the command with an error that is not `torch.OutOfMemoryError` (for
example a Warp allocation error), measure the other counts and record 16384 as not fitting, so the sweep drops it:

```
scripts/train.sh throughput --counts 64 256 1024 4096
uv run --no-sync python -c "import dataclasses, sys; from pathlib import Path; from mprl.budget import load_measurements, save_measurements; p = Path(sys.argv[1]); m = load_measurements(p); save_measurements(p, dataclasses.replace(m, fits={**m.fits, 16384: False})); print(load_measurements(p))" /workspace/mprl/measurements.json
```

Batch arithmetic per count (shipped G1 PPO config: T = 24 steps per env, 4 minibatches, 5 epochs, so 20 gradient
steps per iteration). Fill the last two columns from `steps_per_second`: frames = steps/s × 2700 s, iterations =
frames / (N × 24).

| N | N × T (batch) | minibatch | frames in 45 min | iterations in 45 min |
| --- | --- | --- | --- | --- |
| 64 | 1,536 | 384 | sps × 2700 | frames / 1,536 |
| 256 | 6,144 | 1,536 | | |
| 1024 | 24,576 | 6,144 | | |
| 4096 | 98,304 | 24,576 | | |
| 16384 | 393,216 | 98,304 | | |

**Pilot** (45 training minutes at 4096 envs, about 0.85 h wall clock; then the pod deletes itself). It first prints
its cost at the given price; add `--price-per-hour <price>` if the price you read is not $1.79:

```
scripts/train.sh pilot
```

The pilot is run `main-long-n04096-s0`, the first 45 minutes of the long run. Do not `git pull` between the pilot
and the sweep: the sweep reuses the pilot only if HEAD equals the pilot's recorded git sha.

## 5. Pilot gate (free: your Mac and W&B)

No threshold for "walks" is defined in the spec or the evaluator, so this is a judgment call. Two references:

- A policy that stands still scores `vel_error_xy` = mean |command xy| over the 6 evaluation commands =
  (0.5 + 1.0 + 0.5 + 0.5 + 0.5 + 0) / 6 = **0.5 m/s**, with `no_fall_fraction` possibly 1.0. Standing is not walking.
- An untrained policy (one iteration, 4 envs, CPU, measured 2026-10-28): `vel_error_xy` 0.68,
  `no_fall_fraction` 0.0, `mean_time_to_fall_s` 1.61 of 10 s.

A suggested bar, to be confirmed with the video: `no_fall_fraction` at least 0.9 and `vel_error_xy` clearly below
0.5 (for example under 0.25). Evaluate the pilot's final checkpoint on your Mac's CPU (about 30 s; the evaluator
is the same one `results` uses: 6 fixed commands × 4 envs, seeds 0 and 1, 10 s horizon). The repo is private, so
put the token in the environment first:

```
export HF_TOKEN=<token>
uv run hf download <HF_REPO_ID> --include "runs/main-long/main-long-n04096-s0/*" --local-dir pilot
uv run python -c "from pathlib import Path; from mprl.checkpoints import CheckpointStore; from mprl.evaluator import EvalConfig, evaluate_checkpoint; ref = CheckpointStore(Path('pilot/runs/main-long/main-long-n04096-s0')).latest(); print(ref); print(evaluate_checkpoint('pilot', ref, EvalConfig(), 'cpu'))"
```

In W&B (project `WANDB_PROJECT`, group `main-long`) look at `Train/mean_episode_length` (climbing toward the
episode limit means it stops falling), `Metrics/twist/error_vel_xy`, `Train/mean_reward`, the `Episode_Reward/*`
terms, `Diag/approx_kl` and `Diag/explained_variance`.

Also check the overhead the throughput test estimated: W&B's Runtime for the pilot divided by its training clock
(`train_seconds` in its `state.json`, 2700) minus 1 should be close to `io_overhead_fraction`. Throughput logged to
TensorBoard, so it does not include W&B's cost. If the pilot's figure is higher, write it into the measurements
(on session B's pod, before the sweep) so the estimate is honest:

```
uv run --no-sync python -c "import dataclasses, sys; from pathlib import Path; from mprl.budget import load_measurements, save_measurements; p = Path(sys.argv[1]); save_measurements(p, dataclasses.replace(load_measurements(p), io_overhead_fraction=float(sys.argv[2]))); print(load_measurements(p))" /workspace/mprl/measurements.json <fraction>
```

**No-go:** if the G1 does not walk at 45 minutes, use the spec's fallback: move hours from seeds into run length
with `sweep` options, no code change. For example 2 seeds × 67 minutes costs about what 3 × 45 does
(5 counts × 2 × 67 = 670 run-minutes against 675):

```
scripts/train.sh sweep --name main --seeds 0 1 --budget-minutes 67
```

A budget over 45 minutes means the 45-minute pilot cannot stand in for `main-n04096-s0`, so that run trains too
(the sweep says why; `--reuse-pilot` is harmless). `--long-run-minutes` changes the long run (at least 45, the
pilot it resumes). **Keep the name `main`.** The long run is `<name>-long-n04096-s0` and the pilot is
`main-long-n04096-s0`, so only `--name main` resumes the pilot. Any other name costs about +1.5 h, +$2.69 at
$1.79/h: its long run trains from zero (180 minutes instead of 135, +0.75 h), and the pilot can no longer stand in
for its `n04096-s0` run, which trains too (+0.75 h). The budget options need no new name: the gate is decided
before any sweep run exists, and if a `main` run had already finished at 45 minutes, a larger `--budget-minutes`
continues it on the same clock and W&B run (the config hash leaves the budget out). The estimate and the cut
confirmation work as below.

## 6. Session B: the sweep (Paid, about 13 h, running total about $25.52)

Size the pod's `--terminate-after` (section 3) from the estimate, not a fixed 16 h: session B trains the sweep
runs and the rest of the long run, so take the estimate's sweep-runs and long-run lines and add about 3 h (pod
start, setup, env builds, slack). From `io_overhead_fraction` in `measurements.json` (session A prints it), the
default sweep is `(15 × 0.75 + 3) × (1 + io) + 3` hours: 17.96 h at io = 0.05, 18.68 h at io = 0.1. This counts
`main-n04096-s0` and the long run's first 45 minutes, which the pilot already trained, so it errs long, which is the
safe side for a backstop. Then SSH in, start `tmux`, and:

```
cd /workspace/massively-parallel-rl
scripts/setup.sh
scripts/train.sh sweep --name main --reuse-pilot
```

Add `--price-per-hour <price>` if the A100 price has changed, and the no-go options (section 5) if you chose them.

Re-run `setup.sh` on **every** new pod: the image's `/opt/venv` points its editable `mprl` at the image's copy
(`/opt/mprl`, no `.git`, no `.env`) until setup re-syncs it to the clone. Press Enter at every prompt to keep `.env`.

The sweep first prints the estimate: one line each for setup (4.25 h), the sweep runs × minutes × (1 + io), the
long run as `180 of 180 min left` × (1 + io), and the evaluations × measured seconds, then the total hours and
dollars. It is conservative on purpose. It costs the whole project against the $40 cap, so setup (session A,
already spent) is included. It costs the long run in full, although the pilot already trained its first 45
minutes. It also costs `main-n04096-s0` even when the pilot stands in for it. At $1.79/h and io = 0 those two
add 2 × 0.75 h = 1.5 h, about $2.69, over the spend in section 1's rows B and C. Two more terms are fixed rather
than measured. Setup is always 4.25 h (the spec's throughput test plus debugging, smoke tests, evaluation and
rendering), against about 1.4 h that session A's rows spend; the rest is the reserve. Evaluations are counted at
285 (15 runs × 15 marks = 225, plus the long run's 60 marks), while `results` evaluates the 225 and spends the
long run's 60 marks on rendering its videos (section 7): a play-env build and a 4 s rollout each, about one
evaluation's time, so that term roughly holds. A plan it accepts therefore fits the cap with at least the 1.5 h
margin.

With no cut needed it starts training at once. With cuts it lists them and asks `Train with these cuts? [y/N]`;
`N` keeps the pod running and trains nothing. `--yes` accepts the cuts without asking (they are still printed);
without a terminal and without `--yes`, it refuses. Over $40 after every cut, it refuses to start.

`--reuse-pilot` lets the pilot stand in for `main-n04096-s0` when HEAD equals the pilot's git sha, the configs
match, and the pilot is complete at 45 minutes; otherwise it prints why not and trains that run too (+0.75 h,
+$1.34). Then every run trains count-major (64 first; GPU memory is released after each run), then
the long run resumes the pilot to 180 minutes (90 with `long_run_half`). Completed runs are skipped; each run
uploads its final checkpoint to Hugging Face when it ends. When everything is done or anything fails, the pod
deletes itself after the uploads have finished or been recorded as failed.

If `long_run_half` was applied and money is left at the end, extend the long run to 180 minutes on a later pod (it
resumes the same clock and W&B run; 1.5 h, $2.69):

```
scripts/train.sh train-run --sweep main-long --num-envs 4096 --seed 0 --minutes 180
```

## 7. Session C: results (Paid, about 2 h, running total about $29.10)

Create a new pod (`--terminate-after` 4 h), SSH in, start `tmux`, then:

```
cd /workspace/massively-parallel-rl
scripts/setup.sh
scripts/train.sh results --name main --primary-metric vel_error_xy --terminate
```

It evaluates every sweep run at every evaluation mark (each mark takes the checkpoint that crossed it), writes
`results.csv`, `learning_curves.png`, `race.mp4` (median seed per count, 1.0 m/s straight ahead, one panel per
count) and `README.md` (the model card), and renders the **long-run videos**: `main-long-n04096-s0` at each of its
evaluation marks (60 at 3-min marks up to 180 min; 30 with `long_run_half`), the checkpoint that crossed the mark,
4 s at 1.0 m/s straight ahead, into `sweeps/main-long/main-long-n04096-s0/videos/t<minutes>min_it<iteration>.mp4`.
It logs everything to the W&B run `main-results` (the videos as `results/long_run_video` against
`results/long_run_train_minutes`), uploads the results to `results/main` and the videos to
`results/main/long_run_videos` on Hugging Face, then deletes the pod. Evaluations and videos are cached per
checkpoint, so a re-run evaluates and renders only what is new. Training is not touched: the videos come from
checkpoints already on the volume. The Hub's `long_run_videos` folder mirrors the local set (stale `*.mp4` are
deleted). A long run whose `state.json` is missing or not `complete` is a WARNING and a card line, not a refusal.

**Exit 1 means a publish failed** (W&B or Hub): `results_publish.json` names it, the pod is still deleted with
`--terminate`, and re-running the same command retries; an unchanged W&B publish is not logged twice.

**Training then results in one go** (e.g. finishing the long run first): chain with `;`, not `&&`, so the pod is
deleted even when training fails (`results` runs with a warning on a failed long run):

```
scripts/train.sh train-run --sweep main-long --num-envs 4096 --seed 0 --minutes 90 --no-terminate; scripts/train.sh results --name main --primary-metric vel_error_xy --terminate
```

`results` still refuses before its pod guard without `RUNPOD_API_KEY` or a pod id; then the pod stays up.

**Primary metric: `vel_error_xy`.** It is the one number that separates walking from standing: a robot that stands
still scores 0.5 m/s on it but can score 1.0 on `no_fall_fraction`. `mean_time_to_fall_s` saturates at the 10 s
horizon, and `mean_tracking_reward` is `exp(-err²/std²)` of the same error, so it adds nothing. `vel_error_xy` is
averaged over standing env-steps only, so read it next to `no_fall_fraction` (both are in the CSV and the card).

## 8. Recovery and checking state

**A pod died, was preempted or hit `--terminate-after`:** create a new pod with the same volume (section 3), then:

```
cd /workspace/massively-parallel-rl
scripts/setup.sh
scripts/train.sh <the same command as before>
```

Resume is automatic: completed runs are skipped, a partial run resumes from its latest complete checkpoint with
the same training clock and W&B run id (a checkpoint is written atomically, so a kill mid-write never counts), and
a complete run whose upload failed retries only the upload. A run whose stored config hash differs refuses to
resume (`ConfigMismatch`) and names the cause.
A resume picks the newest checkpoint that passes a zip CRC check and unpickles; a damaged one is renamed
`<name>.pt.corrupt`, noted in `state.json` `uploads["corrupt_checkpoint/<name>"]`, and the next older one is used.
If none loads, it stops with `CheckpointsCorrupt` (nothing restarts from scratch silently). A resume re-randomizes
episode lengths, as mjlab's train script does, so the logged episode metrics dip for about one episode (~42
iterations at 24 steps), as at a fresh start. Do not `git pull` mid-project; the pilot reuse depends on the sha.

**Where things live on the volume** (`mprl.paths`, root `/workspace/mprl`):

| Path | Content |
| --- | --- |
| `measurements.json` | throughput results (read by `sweep`) |
| `throughput/` | throughput scratch, removed by the next throughput run |
| `sweeps/<sweep>/<run_id>/checkpoints/model_it<7 digits>_s<6 digits>.pt` | checkpoints (iteration, training seconds) and the final `checkpoints.onnx` |
| `sweeps/<sweep>/<run_id>/state.json` | status (`running`, `complete`, `failed`), W&B id, config hash, training seconds, budget, git sha, upload outcomes |
| `sweeps/<sweep>/<run_id>/params/` | resolved env and agent configs |
| `sweeps/<sweep>/<run_id>/eval/eval.csv` | cached evaluations |
| `sweeps/<sweep>-long/<run_id>/videos/` | the long run's videos, one per evaluation mark (written by `results`) |
| `sweeps/<sweep>/results/` | `sweep_manifest.json`, `pilot_reuse.json`, `results.csv`, `learning_curves.png`, `race.mp4`, `README.md` |
| `sweeps/<sweep>/results_publish.json` | outcome of the W&B and Hub publish |

Run ids are `<sweep>-n<5 digits>-s<seed>`: `main-n00064-s0` … `main-n16384-s2`; the pilot and long run is
`main-long-n04096-s0`.

```
grep -H '"status"' /workspace/mprl/sweeps/*/*/state.json
python -m json.tool /workspace/mprl/sweeps/main-long/main-long-n04096-s0/state.json
ls /workspace/mprl/sweeps/main-long/main-long-n04096-s0/checkpoints
python -m json.tool /workspace/mprl/sweeps/main/results/sweep_manifest.json
```

**W&B:** project `WANDB_PROJECT` (default `mprl-humanoid-walk`). Each training run has id `<run_id>-<8 hex>` and
job type `train`, grouped by sweep (`main`, `main-long`) and tagged `n<5-digit num_envs>` and `seed<seed>`, so all
counts compare in one group and a tag (or `config.run.num_envs`) splits them; `config.run_id`, `config.batch` and
`config.git_sha` identify it. `Train/mean_reward` and `Train/mean_episode_length` are charted against
`Perf/train_clock_minutes` (the training clock, logged every iteration); every other chart uses the iteration and
can be switched to that x-axis. rsl_rl's `Train/*/time` wall-time series is not sent to W&B (it would make W&B drop
every later metric). If `wandb.init` fails online (network, auth, timeout), that run logs offline instead, says so
in a warning and in its `state.json` (`uploads.wandb`), and the sweep goes on; upload it later with
`wandb sync /workspace/massively-parallel-rl/wandb/offline-run-*<run id>*`. The results run is `main-results` (job type `results`). Local W&B files are under
`/workspace/massively-parallel-rl/wandb/`.

**Hugging Face** (`HF_REPO_ID`, private): `runs/<sweep>/<run_id>/` holds the final checkpoint, `checkpoints.onnx`,
`params/` and `state.json`; `results/<sweep>/` holds the results directory above.

## 9. Known risks and open verifications

- **Network tests and the real-pod termination test have never run.** Session A runs the network tests. Run the
  termination test once from your Mac against a throwaway pod you create (the cheapest Secure Cloud GPU on
  2026-10-28 was the RTX 2000 Ada at $0.24/h); it deletes only that pod and confirms the deletion with a read:

  ```
  MPRL_THROWAWAY_POD_ID=<pod id> RUNPOD_API_KEY=<key> uv run pytest tests/network/test_runpod_terminate.py -m network
  ```

  Until then, self-termination at the end of `pilot` is the first real test: check `runpodctl pod list` after
  the pilot and delete the pod by hand if it is still there.
- **W&B resume** (fixed 2026-10-29): resumed runs reopen the run by id with `resume="allow"`; `resume_from` (Rewind)
  is a private preview and was rejected with a 400. The offline fallback now calls `wandb.teardown()` first (without
  it the retry raised "run ID ... is in use"). Both are tested live by `test_wandb_resume.py`. After a crash between
  checkpoints, W&B keeps the pre-crash values for the iterations between the last checkpoint and the crash.
- **mujoco_warp out-of-memory class on the A100.** The throughput test records only `torch.OutOfMemoryError` as
  "does not fit"; any other allocation error ends the command. The workaround is in section 4.
- **EGL rendering on RunPod** is unverified; the render tests in session A check it (results need it, training
  does not). `scripts/train.sh preflight` does not render.
- **Overhead measured vs estimated.** `io_overhead_fraction` comes from 20 TensorBoard-logged iterations per count,
  not from W&B-logged training; compare it with the pilot (section 5) before the sweep.
- **A100 stock.** On 2026-10-28 every listed A100 data center was "LOW". If no A100 is free in the volume's data
  center, wait or create a second volume elsewhere; data does not sync between volumes.
