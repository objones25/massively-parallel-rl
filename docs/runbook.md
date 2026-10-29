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

Running total by step, at $1.79/h, before the measured overhead. The sweep's printed estimate replaces rows B and C
once `measurements.json` exists.

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
- `--terminate-after` is a backstop against a forgotten pod: A 3 h, B 16 h, C 4 h. If it fires mid-run,
  nothing is lost; resume as in section 8.

```
runpodctl pod create --name mprl-a --cloud-type SECURE --gpu-id "NVIDIA A100-SXM4-80GB" --gpu-count 1 --data-center-ids US-KS-2 --network-volume-id <volume id> --volume-mount-path /workspace --image ghcr.io/objones25/massively-parallel-rl:<sha> --min-cuda-version 12.8 --container-disk-in-gb 30 --ports "22/tcp,8888/http" --terminate-after "$(date -u -v+3H +%Y-%m-%dT%H:%M:%SZ)"
runpodctl ssh info <pod id>
```

Then SSH in with the printed command. Run long commands inside `tmux`, so a dropped SSH connection does not kill
training (SIGHUP would end the process without terminating the pod, and the idle pod would keep billing):

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
- `runpod`: the pod id resolves (`RUNPOD_POD_ID`, else `/proc/1/environ`) and a GET of this pod returns 200.
- `volume`: a probe file under `/workspace/mprl` is written, read back and deleted.

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
strips `WANDB_*`, `HF_*` and `RUNPOD_*` only from the other tiers), so export `.env` into the shell. `RUNPOD_POD_ID`
is only in PID 1's environment, so export it from there:

```
set -a; . ./.env; set +a
export RUNPOD_POD_ID=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')
uv run --no-sync pytest -m network -v --no-cov --deselect tests/network/test_runpod_terminate.py
```

`test_wandb_resume.py` is the only test of W&B `resume_from` against the real service (project `mprl-tests`);
`test_hub_real.py` creates a private `<you>/mprl-smoke-test` repo. If the W&B resume test fails, report it before
the sweep (section 9).

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
pilot it resumes). Run ids do not change with the budget, so use a new `--name` (for example `main67`) if a
45-minute sweep was already started under `main`, and pass the same name to `results`. The estimate and the
cut confirmation work as below.

## 6. Session B: the sweep (Paid, about 13 h, running total about $25.52)

Create a new pod (section 3, `--terminate-after` 16 h), SSH in, start `tmux`, then:

```
cd /workspace/massively-parallel-rl
scripts/setup.sh
scripts/train.sh sweep --name main --reuse-pilot
```

Add `--price-per-hour <price>` if the A100 price has changed, and the no-go options (section 5) if you chose them.

Re-run `setup.sh` on **every** new pod: the image's `/opt/venv` points its editable `mprl` at the image's copy
(`/opt/mprl`, no `.git`, no `.env`) until setup re-syncs it to the clone. Press Enter at every prompt to keep `.env`.

The sweep first prints the estimate: one line each for setup (4.25 h), the sweep runs × minutes × (1 + io), the
long run's remaining minutes × (1 + io), and the evaluations × measured seconds, then the total hours and dollars.
With no cut needed it starts training at once. With cuts it lists them and asks `Train with these cuts? [y/N]`;
`N` keeps the pod running and trains nothing. `--yes` accepts the cuts without asking (they are still printed);
without a terminal and without `--yes`, it refuses. Over $40 after every cut, it refuses to start.

`--reuse-pilot` lets the pilot stand in for `main-n04096-s0` when HEAD equals the pilot's git sha, the configs
match, and the pilot is complete at 45 minutes; otherwise it prints why not and trains that run too (+0.75 h,
+$1.34). The estimate still counts that run, so it errs high. Then every run trains count-major (64 first), then
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
count) and `README.md` (the model card), logs them to the W&B run `main-results`, uploads them to
`results/main` on Hugging Face, then deletes the pod. Evaluations are cached per checkpoint, so a re-run evaluates
only what is new.

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
resume (`ConfigMismatch`) and names the cause. Do not `git pull` mid-project; the pilot reuse depends on the sha.

**Where things live on the volume** (`mprl.paths`, root `/workspace/mprl`):

| Path | Content |
| --- | --- |
| `measurements.json` | throughput results (read by `sweep`) |
| `throughput/` | throughput scratch, removed by the next throughput run |
| `sweeps/<sweep>/<run_id>/checkpoints/model_it<7 digits>_s<6 digits>.pt` | checkpoints (iteration, training seconds) and the final `checkpoints.onnx` |
| `sweeps/<sweep>/<run_id>/state.json` | status (`running`, `complete`, `failed`), W&B id, config hash, training seconds, budget, git sha, upload outcomes |
| `sweeps/<sweep>/<run_id>/params/` | resolved env and agent configs |
| `sweeps/<sweep>/<run_id>/eval/eval.csv` | cached evaluations |
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
job type `train`, grouped by sweep (`main`, `main-long`); `config.run_id`, `config.batch` and `config.git_sha`
identify it. The results run is `main-results` (job type `results`). Local W&B files are under
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
- **W&B `resume_from` is untested online** (it is a beta W&B feature). `test_wandb_resume.py` in session A tests it.
  If it fails, resumed runs fail at `wandb.init`; the fallback is `--wandb-mode offline` on the resumed command.
- **mujoco_warp out-of-memory class on the A100.** The throughput test records only `torch.OutOfMemoryError` as
  "does not fit"; any other allocation error ends the command. The workaround is in section 4.
- **EGL rendering on RunPod** is unverified; the render tests in session A check it (results need it, training
  does not). `scripts/train.sh preflight` does not render.
- **Overhead measured vs estimated.** `io_overhead_fraction` comes from 20 TensorBoard-logged iterations per count,
  not from W&B-logged training; compare it with the pilot (section 5) before the sweep.
- **A100 stock.** On 2026-10-28 every listed A100 data center was "LOW". If no A100 is free in the volume's data
  center, wait or create a second volume elsewhere; data does not sync between volumes.
