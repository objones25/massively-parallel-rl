# Massively Parallel Humanoid Walking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the Unitree G1 humanoid to walk with mjlab + rsl_rl at 64–16384 parallel environments, each for 45 minutes of training-clock time on one A100. Produce a race video and a learning-curve chart, with resume, W&B observability, Hugging Face uploads and pod auto-termination.

**Architecture:** A Python package `mprl` (in `src/mprl/`) wraps mjlab's registered task `Mjlab-Velocity-Flat-Unitree-G1`. Its central piece is a `TimedOnPolicyRunner`, a subclass of mjlab's `VelocityOnPolicyRunner`. It runs rsl_rl's training loop under a `TrainingClock` that counts only rollout and PPO update time, and saves checkpoints on training-clock minutes. Around it sit small single-purpose modules: checkpoint store, diagnostics, resumable W&B writer, HF publisher, RunPod terminator, budget, evaluator, renderer, race-video composer, learning curves. Thin `tyro` CLIs and two bash scripts sit on top.

**Tech Stack:** Python 3.12, uv, mjlab 1.6.0 (pulls rsl-rl-lib 5.4.2, torch 2.14.1, mujoco-warp 3.11.0, wandb 0.31.0), huggingface_hub, httpx, imageio + ffmpeg, Pillow, matplotlib, tyro, pytest + hypothesis, ruff, pyright. Docker image built by GitHub Actions and pushed to GHCR; RunPod Secure Cloud A100 80GB.

**Spec:** `docs/superpowers/specs/2026-10-27-parallel-humanoid-walking-design.md`. Read it before any task. The plan argues from it and does not repeat its rationale.

## Global Constraints

- Task id `Mjlab-Velocity-Flat-Unitree-G1`. Use mjlab's shipped G1 PPO config unchanged. The **only** per-run difference is `num_envs` (and seed).
- Environment counts `(64, 256, 1024, 4096, 16384)`; seeds `(0, 1, 2)`; budget `45.0` training-clock minutes per sweep run; checkpoint every `3.0` training-clock minutes; long run `4096` envs, `180.0` minutes.
- The training clock counts rollout + `compute_returns` + PPO `update` only. On CUDA it synchronizes before reading time. Checkpoint I/O, logging, diagnostics, evaluation, rendering and uploads are off the clock.
- Price `$1.79/h` (Secure Cloud A100, read 2026-10-27). Hard cap `$40.00`. Cut order: `eval_every_6min`, `long_run_half`, `drop_16384`, `two_seeds`. Never drop 64.
- Volume root from env `MPRL_VOLUME_ROOT` (default `/workspace/mprl`). The Python environment lives at `/opt/venv` in the image, never under `/workspace`.
- uv only (`uv sync`, `uv run`, `uv.lock` committed). No pip, no requirements.txt.
- Negative space: copy the skill's `negative_space.py` to `src/mprl/negative_space.py`. Use `require`/`expect`/`check_shape`/`check_finite`, never bare `assert` in `src/`. Bound every loop and retry. Programmer errors crash; operating errors (network, OOM, missing keys, bad user input) raise typed exceptions from `src/mprl/errors.py` and are handled.
- TDD: failing test first, watch it fail, then implement. **Smoke-test every integration against the real library before writing any double; doubles use `create_autospec` of the installed class.**
- Never quote an API, default or number from memory. Facts below were verified on 2026-10-27 against the installed packages. Re-verify anything else with Context7 or `inspect` before depending on it.
- Pytest markers: `smoke` (real libraries, CPU, runs by default), `network` (real W&B/HF/RunPod; needs keys; deselected by default), `gpu` (CUDA; deselected by default).
- Attention maps are not applicable (MLP policy). Do not add them.

### Verified facts (macOS arm64, Python 3.12, `uv pip install mjlab`, 2026-10-27)

- `mjlab.tasks.registry`: `load_env_cfg(task, play=False)`, `load_rl_cfg(task)`, `load_runner_cls(task)` → `VelocityOnPolicyRunner` (`mjlab/tasks/velocity/rl/runner.py`). Its `save()` also exports ONNX and may `wandb.save` when `cfg["upload_model"]`.
- `MjlabOnPolicyRunner.save` stores `infos["env_state"]["common_step_counter"]`; `load` restores it.
- **Loading into a runner that has already trained in the same process fails** (`Inplace update to inference tensor outside InferenceMode`). Loading in a fresh runner works. Resume is therefore always a fresh process or a fresh runner.
- `rsl_rl.runners.OnPolicyRunner.learn` (rsl-rl-lib 5.4.2, `rsl_rl/runners/on_policy_runner.py`) is one loop. Per iteration: `alg.act` → `env.step` → `check_nan` → `alg.process_env_step` → `logger.process_env_step` (×`num_steps_per_env`, under `torch.inference_mode()`), then `alg.compute_returns(obs)`, `alg.update()`, `logger.log(...)`, save every `save_interval`. There is no time-based hook.
- Timeouts are bootstrapped: `RslRlVecEnvWrapper` sets `extras["time_outs"]` and `PPO.process_env_step` adds `gamma * values * time_outs`.
- PPO clips actor and critic gradients separately with `max_grad_norm` inside `update()` (`rsl_rl/algorithms/ppo.py:302-303`).
- `rsl_rl.utils.Logger.init_logging_writer` builds the writer from `cfg["logger"]`, a dict with a `"class_name"` key resolved by `resolve_callable`. The remaining keys go to `writer_class(log_dir=..., **kwargs)`. Writers subclass `rsl_rl.utils.log_writer.LogWriter` (only `add_scalar` is abstract). The built-in `WandbLogWriter` calls `wandb.init` **without `id`/`resume`**, so it cannot resume a run.
- The Logger already writes `Loss/*`, `Loss/learning_rate`, `Perf/total_fps`, `Perf/collection_time`, `Perf/learning_time`, `Train/mean_reward`, `Train/mean_episode_length`, and the env's `extras["log"]` (`Episode_Reward/*`, `Metrics/twist/*`, curriculum values).
- The G1 flat task ships curriculum `command_vel`, which advances on `env.common_step_counter`. That counter is incremented once per `env.step` (`mjlab/envs/manager_based_rl_env.py:447`), independent of `num_envs`. Stages are at steps `0` (lin_vel_x ±1.0, ang_vel_z ±0.5), `5000*24` (lin_vel_x −1.5..2.0, ang_vel_z ±0.7) and `10000*24` (lin_vel_x −2.0..3.0). The `terrain_levels` curriculum is removed for flat.
- Play config (`play=True`): `episode_length_s = 1e9`, no observation corruption, no `push_robot`, `curriculum = {}`. Its twist ranges are `lin_vel_x (-1.5, 2.0)`, `lin_vel_y (-1.0, 1.0)`, `ang_vel_z (-0.7, 0.7)`.
- Terminations: `time_out` (time_out=True) and `fell_over` (`bad_orientation`, 70°).
- CPU on the Mac: G1 flat env with `num_envs=4` builds, trains 2 PPO iterations in ~2 s, and saves `model_*.pt` plus an ONNX file. A fresh runner `load` restores `iter` and `common_step_counter`. `render_mode="rgb_array"` renders `(240, 320, 3)` frames (`ViewerConfig.height=240, width=320`).
- `wandb.init(id=..., resume="allow"|"must", resume_from="{id}?_step={n}", group=..., job_type=..., mode="online"|"offline"|"disabled")`. `resume_from` takes precedence over `id` and is beta.
- `HfApi.create_repo(repo_id, *, private=..., repo_type=..., exist_ok=...)`.
- RunPod: the pod's own id is in `RUNPOD_POD_ID` **for PID 1 only**. SSH login shells do not get pod env vars, so fall back to `/proc/1/environ`. REST v2 uses `Authorization: Bearer <key>`. `DELETE /v2/pods/{id}` terminates the pod; the network volume is detached and kept. **Verify the REST base URL against docs.runpod.io/api-reference-v2 in Task 8.**

## Review Focus

1. **Killed mid-checkpoint write** (pod preempted, OOM-killer). Expected: resume uses the last *complete* checkpoint, and a partial file never counts. Owned by Task 3 (`test_partial_temp_file_is_ignored`) and Task 9 (`test_resume_after_kill_continues_clock`).
2. **Resuming with a changed config or code** (num_envs edited, code changed between pilot and sweep). Expected: refuse to resume a run whose stored config hash differs, with a message naming the field. Owned by Task 9 (`test_resume_refuses_config_mismatch`).
3. **Re-running `train` after everything finished, or extending a budget** (pilot 45 → long run 180). Expected: completed runs are skipped with no GPU time spent, and a raised budget continues the same clock and W&B run. Owned by Task 9 (`test_completed_run_is_skipped`, `test_raised_budget_continues`).
4. **Network failure at the end of a run** (HF 503, W&B down). Expected: checkpoints intact on the volume, failure recorded in `state.json`, and the pod terminated only after uploads finished or were recorded as failed. Owned by Task 7 (`test_retries_then_raises_upload_failed`) and Task 9 (`test_terminate_runs_after_failed_upload`).
5. **16384 envs do not fit on the A100.** Expected: the throughput test records "does not fit" from the CUDA OOM, and the sweep drops that count with the measurement cited instead of crashing. Owned by Task 10 (`test_oom_count_recorded_not_raised`, `test_sweep_excludes_counts_that_do_not_fit`).

---

## File Structure

```
pyproject.toml, uv.lock, .python-version            uv project, pytest/ruff/pyright config
.env.example                                        documented keys, no values
src/mprl/negative_space.py                          copied helpers (require/expect/check_shape/...)
src/mprl/errors.py                                  typed operating errors
src/mprl/config.py                                  RunSpec, SweepSpec, batch arithmetic, config hash
src/mprl/paths.py                                   on-volume layout per run / sweep
src/mprl/clock.py                                   TrainingClock
src/mprl/checkpoints.py                             CheckpointStore (atomic, minute-indexed)
src/mprl/timed_runner.py                            TimedOnPolicyRunner
src/mprl/diagnostics.py                             grad / activation / weight metrics
src/mprl/wandb_writer.py                            ResumableWandbWriter (rsl_rl LogWriter)
src/mprl/hub.py                                     HubPublisher (bounded retry)
src/mprl/runpod.py                                  own pod id + PodTerminator
src/mprl/run_state.py                               state.json (status, wandb id, config hash, uploads)
src/mprl/train.py                                   train_run(): one run end-to-end
src/mprl/budget.py                                  cost estimate + cut order
src/mprl/throughput.py                              steps/s and fit per env count
src/mprl/sweep.py                                   sweep orchestration, pilot reuse
src/mprl/evaluator.py                               fixed-command offline evaluation
src/mprl/render.py                                  rollout frames for a checkpoint
src/mprl/race_video.py                              panel composition + writer
src/mprl/curves.py                                  aggregation + learning-curve figure
src/mprl/results.py                                 build all results for a sweep, publish
src/mprl/env_file.py                                .env.example parsing / .env writing
src/mprl/preflight.py                               setup checks
src/mprl/cli.py                                     tyro subcommands: setup, preflight, throughput, train-run, sweep, results
scripts/setup.sh, scripts/train.sh                  thin bash entry points (on the pod)
docker/Dockerfile, .dockerignore                    image with /opt/venv
.github/workflows/tests.yml, image.yml              CI tests; build + push to GHCR
tests/unit/…, tests/smoke/…, tests/network/…, tests/gpu/…
```

---

### Task 1: Project scaffold, test gates, and the CPU platform matrix

**Files:**
- Create: `pyproject.toml`, `.python-version` (`3.12`), `uv.lock`, `src/mprl/__init__.py`, `src/mprl/negative_space.py` (copy of `~/.claude/skills/negative-space-programming/assets/negative_space.py`), `src/mprl/errors.py`, `tests/conftest.py` (from pytest-expert `assets/conftest.py` + `assets/conftest_ml.py`), `scripts/audit_tests.py` (copy from pytest-expert), `scripts/audit_negative_space.py` (copy from the negative-space skill), `tests/smoke/test_mjlab_cpu.py`
- Modify: `CLAUDE.md` (Commands section)

**Interfaces:**
- Produces: `mprl.errors`: `class MprlOperatingError(Exception)` and its subclasses `ConfigMismatch`, `UploadFailed`, `OwnPodUnknown`, `TerminateFailed`, `BudgetExceeded`, `EnvDoesNotFit`, `MissingSecret`. Pytest markers `smoke`, `network`, `gpu`. `addopts` deselects `network` and `gpu`.

- [ ] **Step 1: Write `pyproject.toml`.** Pin `mjlab==1.6.0`. Add `huggingface_hub`, `httpx`, `imageio[ffmpeg]`, `pillow`, `matplotlib`, `tyro`, `python-dotenv`; dev group `pytest`, `hypothesis`, `pytest-randomly`, `pytest-xdist`, `ruff`, `pyright`. Use src layout; console script `mprl = "mprl.cli:main"`. Merge the pytest-expert `assets/pytest-config.toml` strict settings and `assets/ruff-pytest.toml`. Run `uv lock` and `uv sync`, and check that the lock resolves for both `darwin-arm64` and `linux-x86_64` (with a CUDA torch wheel on Linux; verify torch's index setup in Context7 for uv).
- [ ] **Step 2: Write the failing smoke tests** in `tests/smoke/test_mjlab_cpu.py`, marked `smoke`:
  - `test_g1_flat_env_steps_on_cpu`: build with `num_envs=4` on `"cpu"`, step 24 times with zero actions; obs finite; `env.common_step_counter == 24`.
  - `test_g1_runner_trains_two_iterations_on_cpu`: `model_0.pt` and `model_1.pt` exist under the log dir.
  - `test_fresh_runner_restores_iteration_and_step_counter`: after 3 iterations, a fresh runner's `load` gives `current_learning_iteration == 2` and `common_step_counter == 72`.
  - `test_play_env_renders_rgb_frame`: frame shape `(240, 320, 3)`, dtype `uint8`.
- [ ] **Step 3: Run** `uv run pytest tests/smoke/test_mjlab_cpu.py -v`. These exercise library behavior, so they should pass immediately. They pin the facts above, and they will fail on any mjlab upgrade that breaks them. If any fails, stop and report: the plan's assumptions are wrong.
- [ ] **Step 4: Platform matrix.** Run the same file inside `docker run --platform linux/amd64` with a CPU-only `python:3.12` image and `uv sync`. Record which tests pass on macOS arm64 and which on Linux x86_64 CPU in `CLAUDE.md` under "Where tests run". Anything that fails on both CPUs becomes a `gpu` test later.
- [ ] **Step 5: Gates.** `uv run python scripts/audit_tests.py tests/` prints `clean: no findings`; `uv run ruff check .`; `uv run pyright src` (strict on `src/mprl`); `uv run python scripts/audit_negative_space.py src/ --select NSP002,NSP003,NSP005,NSP006,NSP007` exits 0.
- [ ] **Step 6: Fill in `CLAUDE.md` Commands:** `uv sync`, `uv run pytest`, single test (`uv run pytest tests/unit/test_clock.py::test_name -v`), markers (`-m network`, `-m gpu`), the audits, ruff and pyright.
- [ ] **Step 7: Commit** `chore: scaffold uv project, test gates, mjlab CPU smoke tests`.

### Task 2: Run configuration, batch arithmetic, paths, config hash

**Files:**
- Create: `src/mprl/config.py`, `src/mprl/paths.py`, `tests/unit/test_config.py`, `tests/unit/test_paths.py`

**Interfaces:**
- Produces:
  - `TASK_ID: Final = "Mjlab-Velocity-Flat-Unitree-G1"`
  - `@dataclass(frozen=True) class RunSpec: sweep: str; num_envs: int; seed: int; budget_minutes: float; checkpoint_every_minutes: float = 3.0; task_id: str = TASK_ID`, with `run_id -> str` (`f"{sweep}-n{num_envs:05d}-s{seed}"`) and `config_hash() -> str` (sha256 of the canonical JSON of every field **except `budget_minutes`**, so raising a budget is not a config change)
  - `@dataclass(frozen=True) class SweepSpec: name: str; env_counts: tuple[int, ...] = (64, 256, 1024, 4096, 16384); seeds: tuple[int, ...] = (0, 1, 2); budget_minutes: float = 45.0; checkpoint_every_minutes: float = 3.0; eval_every_minutes: float = 3.0; long_run_envs: int = 4096; long_run_minutes: float = 180.0`, with `runs() -> list[RunSpec]` (count-major, seed-minor) and `long_run() -> RunSpec` (sweep `f"{name}-long"`, seed 0)
  - `@dataclass(frozen=True) class BatchArithmetic: num_envs: int; steps_per_env: int; num_mini_batches: int`, with properties `batch_size` and `minibatch_size`
  - `def batch_arithmetic(num_envs: int, steps_per_env: int = 24, num_mini_batches: int = 4) -> BatchArithmetic`
  - `@dataclass(frozen=True) class RunPaths: root: Path; spec: RunSpec`, with `run_dir`, `checkpoints_dir`, `state_file`, `eval_dir`, `videos_dir`, `tensorboard_dir`; `def volume_root(environ: Mapping[str, str]) -> Path`; `def sweep_results_dir(root: Path, sweep: str) -> Path`

- [ ] **Step 1: Write the failing tests:**
  - `test_runs_cover_every_count_and_seed`: 15 runs; the first is `n=64,s=0` and the last is `n=16384,s=2`.
  - `test_run_id_is_zero_padded`: `RunSpec("main", 64, 1, 45.0).run_id == "main-n00064-s1"`.
  - `test_config_hash_ignores_budget`: equal hashes for budgets 45 and 180.
  - `test_config_hash_changes_with_num_envs`.
  - `test_batch_arithmetic_64_envs`: batch 1536, minibatch 384.
  - `test_batch_arithmetic_16384_envs`: batch 393216, minibatch 98304.
  - `test_batch_arithmetic_rejects_indivisible`: `pytest.raises(ValueError, match="divisible")` for `batch_arithmetic(10, 3, 4)`. This is a config value, so it is validation, not an assertion.
  - `test_run_paths_layout`: `run_dir == root/"sweeps"/"main"/"main-n00064-s1"`.
  - `test_volume_root_defaults_to_workspace` and `test_volume_root_from_env` (via a plain dict, no real env mutation).
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_config.py tests/unit/test_paths.py -v`. Expected: FAIL (import error).
- [ ] **Step 3: Implement** in `config.py` and `paths.py`. `RunSpec.__post_init__` uses `require` for `num_envs > 0`, `budget_minutes > 0`, and `0 < checkpoint_every_minutes <= budget_minutes`.
- [ ] **Step 4: Run** the same command. Expected: PASS. Then the gates from Task 1 Step 5.
- [ ] **Step 5: Commit** `feat: run/sweep specs, batch arithmetic, volume layout`.

### Task 3: Training clock and atomic checkpoint store

**Files:**
- Create: `src/mprl/clock.py`, `src/mprl/checkpoints.py`, `tests/unit/test_clock.py`, `tests/unit/test_checkpoints.py`

**Interfaces:**
- Produces:
  - `class TrainingClock: __init__(self, elapsed_s: float = 0.0, *, now: Callable[[], float] = time.perf_counter, sync: Callable[[], None] = lambda: None)`. Members: `measure() -> ContextManager[None]`; `elapsed_s: float` property; `elapsed_minutes: float` property; `state_dict() -> dict[str, float]`; `@classmethod from_state_dict(cls, d, *, now=..., sync=...)`.
  - `def cuda_sync_for(device: str) -> Callable[[], None]`: returns `torch.cuda.synchronize` for `cuda:*`, a no-op for `cpu`.
  - `@dataclass(frozen=True, order=True) class CheckpointRef: iteration: int; train_seconds: float; path: Path`
  - `class CheckpointStore: __init__(self, directory: Path)`. Methods: `save_atomic(self, write: Callable[[Path], None], iteration: int, train_seconds: float) -> CheckpointRef` (writes to `.<name>.tmp` in the same directory, then `os.replace`); `list(self) -> list[CheckpointRef]`; `latest(self) -> CheckpointRef | None`; `at_or_before(self, train_minutes: float) -> CheckpointRef | None`. Filename `model_it{iteration:07d}_s{round(train_seconds):06d}.pt`.

- [ ] **Step 1: Write the failing tests:**
  - `test_measure_adds_only_inside_interval`: a fake `now` returns 0, 10, 15, 18. Measuring the 10→15 interval gives `elapsed_s == pytest.approx(5.0)`.
  - `test_sync_called_at_both_ends`: a counter equals 2 after one `measure`.
  - `test_nested_measure_crashes`: a programmer error, so expect `AssertionError` raised by `require`. Check what the copied helper actually raises and match on it.
  - `test_clock_round_trips_state`.
  - `test_resumed_clock_continues`: `from_state_dict({"elapsed_s": 120.0})` and a 5 s measure give 125.
  - `test_save_atomic_returns_ref_with_minutes`.
  - `test_list_sorted_by_iteration`.
  - `test_partial_temp_file_is_ignored` (Review Focus 1): a stray `.model_it0000003_s000180.pt.tmp` does not appear in `list()`.
  - `test_failed_write_leaves_no_file`: a `write` that raises `OSError` leaves no checkpoint and no tmp file, and re-raises.
  - `test_at_or_before_picks_latest_not_after`: refs at 180 s, 360 s, 540 s; `at_or_before(7.0)` is the 360 s ref.
  - A Hypothesis property, `test_at_or_before_never_exceeds_query`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_clock.py tests/unit/test_checkpoints.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** The clock uses `require(delta >= 0)` on every interval. `list()` only parses names that match the exact regex.
- [ ] **Step 4: Run.** Expected: PASS. Then the gates.
- [ ] **Step 5: Commit** `feat: training clock and atomic minute-indexed checkpoint store`.

### Task 4: TimedOnPolicyRunner

**Files:**
- Create: `src/mprl/timed_runner.py`, `tests/smoke/test_timed_runner.py`, `tests/unit/test_checkpoint_schedule.py`

**Interfaces:**
- Consumes: `TrainingClock`, `cuda_sync_for`, `CheckpointStore`, `CheckpointRef` (Task 3).
- Produces:
  - `def next_checkpoint_due(elapsed_s: float, every_s: float, last_saved_s: float) -> bool`. Pure: true when `elapsed_s` has crossed the first multiple of `every_s` above `last_saved_s`.
  - `@dataclass(frozen=True) class LearnResult: iterations_run: int; final: CheckpointRef; budget_exhausted: bool; overshoot_s: float`
  - `class TimedOnPolicyRunner(VelocityOnPolicyRunner)`, with method `learn_for(self, clock: TrainingClock, budget_s: float, checkpoint_every_s: float, store: CheckpointStore, *, fresh_start: bool, max_iterations: int = 10_000_000, on_iteration: Callable[[int], None] | None = None, on_checkpoint: Callable[[CheckpointRef], None] | None = None) -> LearnResult`. It overrides `save(path, infos=None)` to add `infos["clock"] = clock.state_dict()`.
  - Payload key `infos["clock"]["elapsed_s"]` is what resume reads.

- [ ] **Step 1: Write the failing unit tests** for `next_checkpoint_due`: 179 s with every 180 s and last 0 → False; 181 → True; 361 with last 180 → True; 361 with last 360 → False.
- [ ] **Step 2: Write the failing smoke tests** (real mjlab, CPU, `num_envs=4`):
  - `test_learn_for_stops_when_budget_reached`: budget 3 s; the result has `budget_exhausted` and `clock.elapsed_s >= 3.0`.
  - `test_checkpoints_follow_training_clock`: budget 4 s, every 1 s. Checkpoint `train_seconds` are increasing, about 1 s apart, and there is a final one.
  - `test_io_is_off_the_clock`: wrap `store.save_atomic` with a 1.0 s busy-wait; the clock's elapsed time does not include it. Inject the delay via a fake `write` around the real one, not a sleep in the test body.
  - `test_matches_rsl_rl_learn`: same seed and config. `OnPolicyRunner.learn(2)` and `learn_for(..., max_iterations=2)` give `torch.equal` actor parameters. If MuJoCo Warp CPU stepping turns out non-deterministic, record that with evidence and replace the check with an equal-iteration-count plus finite-parameters assertion. Do not loosen silently.
  - `test_checkpoint_payload_has_clock_and_step_counter`.
- [ ] **Step 3: Run** `uv run pytest tests/unit/test_checkpoint_schedule.py tests/smoke/test_timed_runner.py -v`. Expected: FAIL.
- [ ] **Step 4: Implement `learn_for`.** Mirror `OnPolicyRunner.learn` from rsl-rl-lib 5.4.2 call for call, with two changes:
  - Wrap the rollout + `compute_returns` and the `update()` in **two separate** `clock.measure()` blocks. This leaves a gap between them where Task 5 can snapshot weights off the clock.
  - Replace the iteration-count save with `next_checkpoint_due` + `store.save_atomic(self.save, ...)`. The final iteration's checkpoint is always saved.
  
  Pass `init_at_random_ep_len` only when `fresh_start`. Call `self.logger.init_logging_writer()` once, and `stop_logging_writer()` at the end. Bound the loop with `max_iterations` via `require`. `overshoot_s = clock.elapsed_s - budget_s`. Add a header comment naming the rsl_rl version and file it mirrors, plus a smoke test that fails if `rsl_rl.__version__` / package metadata differs from `5.4.2`, so an upgrade forces a re-check.
- [ ] **Step 5: Run.** Expected: PASS. Then the gates.
- [ ] **Step 6: Commit** `feat: time-budgeted runner with training-clock checkpoints`.

### Task 5: Training diagnostics

**Files:**
- Create: `src/mprl/diagnostics.py`, `tests/unit/test_diagnostics.py`, `tests/smoke/test_diagnostics_g1.py`
- Modify: `src/mprl/timed_runner.py` (call diagnostics through the `on_iteration` gap; no new loop)

**Interfaces:**
- Consumes: `TimedOnPolicyRunner` (Task 4).
- Produces: `class TrainingDiagnostics: __init__(self, actor: nn.Module, critic: nn.Module, every_k: int)`. Methods:
  - `attach() -> None` registers `register_post_accumulate_grad_hook` on every parameter to accumulate squared grad norms per backward (pre-clip).
  - `detach() -> None`
  - `before_update() -> None` snapshots weights when the iteration is a K-iteration.
  - `after_update(it: int) -> dict[str, float]`
  - `activation_stats(obs: TensorDict) -> dict[str, float]`, run under `torch.no_grad()` off the clock.
  
  Metric keys:
  - `Diag/grad_norm_preclip/{actor,critic}` (mean over minibatches)
  - `Diag/clip_coef/{actor,critic}` = `min(1, max_grad_norm / preclip)`
  - `Diag/grad_norm/{actor,critic}/{param_name}`
  - `Diag/weight_norm/{...}`
  - `Diag/update_ratio/{...}` = `‖Δw‖ / ‖w‖`
  - `Diag/act_mean|act_std|act_frac_saturated/{actor,critic}/layer{i}`, where ELU saturated means output `< -0.99`
  - `Diag/action_std`
  - `Diag/obs_norm_mean_abs` and `Diag/obs_norm_std_mean`

- [ ] **Step 1: Write the failing unit tests** on a tiny `nn.Sequential(Linear, ELU, Linear)`:
  - `test_grad_norm_matches_manual`: one backward; the reported pre-clip norm equals `torch.linalg.vector_norm` over all grads, `pytest.approx` rel 1e-6.
  - `test_update_ratio_zero_when_weights_unchanged`
  - `test_saturated_fraction_counts_elu_floor`: inputs forcing outputs to −1 give a fraction of 1.0.
  - `test_non_k_iteration_returns_empty`
  - `test_detach_removes_hooks`
  - `test_metrics_are_finite`: uses `check_finite`.
- [ ] **Step 2: Write the failing smoke test** `test_diagnostics_on_g1_runner`: 2 iterations with `every_k=1`. All `Diag/grad_norm_preclip/*` keys are present and positive. **Inspect the real `MLPModel` module tree first** (print `named_modules()` in the test's arrange step's helper) and name layers from it.
- [ ] **Step 3: Run** both. Expected: FAIL.
- [ ] **Step 4: Implement.** Get the actor and critic from `runner.alg`: inspect `PPO` attributes in the installed version (the source shows `_raw_actor`; prefer a public accessor if one exists). Write metrics through `runner.logger.writer.add_scalar(key, value, it)` so the steps align with rsl_rl's own metrics.
- [ ] **Step 5: Run.** Expected: PASS. Then gates.
- [ ] **Step 6: Commit** `feat: per-layer gradient, weight, update-ratio and activation diagnostics`.

### Task 6: Resumable W&B writer

**Files:**
- Create: `src/mprl/wandb_writer.py`, `tests/smoke/test_wandb_writer_offline.py`, `tests/network/test_wandb_resume.py`

**Interfaces:**
- Produces:
  - `class ResumableWandbWriter(LogWriter): __init__(self, log_dir: str, project: str, run_id: str, group: str, job_type: str = "train", resume_from_iteration: int | None = None, mode: str = "online", entity: str | None = None, tags: tuple[str, ...] = (), extra_config: dict | None = None)`. `add_scalar(tag, value, global_step)` → `wandb.log({tag: value}, step=global_step)`. `store_config` puts env and train config into `wandb.config`. `save_model` is a no-op (uploads are owned by Task 7/9). `save_video` logs `wandb.Video`. `stop` → `wandb.finish()`.
  - `def logger_cfg(project: str, run_id: str, group: str, resume_from_iteration: int | None, mode: str) -> dict`. Returns the dict for `agent_cfg["logger"]` (with `"class_name"`); its format comes from what `resolve_callable` accepts in the installed rsl_rl.
  - Fresh run: `wandb.init(id=run_id, resume="allow", ...)`. Resume: `wandb.init(resume_from=f"{run_id}?_step={iteration}", ...)`, so history after the checkpoint is truncated.
- [ ] **Step 1: Write the failing smoke test** (`mode="offline"`, `WANDB_DIR` set to `tmp_path` via monkeypatch). Build a real rsl_rl `Logger` with `logger_cfg(...)`, call `init_logging_writer()`, `add_scalar("Loss/x", 1.0, 0)`. An offline run directory exists, containing the run id.
- [ ] **Step 2: Write the network test** `test_resume_from_truncates_history` (marked `network`, real W&B, throwaway project `mprl-tests`):
  1. Log steps 0–5 and finish.
  2. Re-init with `resume_from_iteration=3` and log step 4 with a new value.
  3. The API history for that run id has a step-4 value equal to the new one.
  
  If `resume_from` is unavailable for the account, fall back to `resume="must"` and log only steps above the last logged one. Record which path was taken in `CLAUDE.md`.
- [ ] **Step 3: Run** `uv run pytest tests/smoke/test_wandb_writer_offline.py -v` (FAIL), then `-m network tests/network/test_wandb_resume.py` (FAIL).
- [ ] **Step 4: Implement.** Run both. Expected: PASS. Gates.
- [ ] **Step 5: Commit** `feat: resumable W&B log writer for rsl_rl`.

### Task 7: Hugging Face publisher

**Files:**
- Create: `src/mprl/hub.py`, `tests/network/test_hub_real.py`, `tests/unit/test_hub_retry.py`

**Interfaces:**
- Produces: `class HubPublisher: __init__(self, api: HfApi, repo_id: str, *, private: bool = True, max_attempts: int = 5, base_delay_s: float = 2.0, sleep: Callable[[float], None] = time.sleep)`. Methods: `ensure_repo() -> None` (`create_repo(..., private=..., repo_type="model", exist_ok=True)`), `upload_folder(local: Path, path_in_repo: str, message: str) -> str` (returns the commit URL). Retries only on server 5xx/429 and connection errors, with exponential backoff, at most `max_attempts`. 4xx raises immediately. Exhaustion raises `UploadFailed` with the last error chained.
- [ ] **Step 1: Write the network smoke test first** (`network`): against a throwaway private repo `{user}/mprl-smoke-test`, call `ensure_repo` twice (idempotent), upload a tmp folder with one file, and `list_repo_files` contains it. This establishes the real exception types for 5xx and connection errors. Record them in the test module docstring.
- [ ] **Step 2: Run** `uv run pytest -m network tests/network/test_hub_real.py -v`. Expected: FAIL (no module), then PASS after a minimal `HubPublisher`.
- [ ] **Step 3: Write the failing unit tests** with `create_autospec(HfApi, instance=True)` raising the **real exception classes observed in Step 1**:
  - `test_retries_then_succeeds`: two 503s then OK; `sleep` called with `[2.0, 4.0]`.
  - `test_retries_then_raises_upload_failed` (Review Focus 4): `max_attempts` 503s → `UploadFailed`, and exactly 5 calls.
  - `test_client_error_not_retried`: one 401 → raises after 1 call.
- [ ] **Step 4: Implement the retry, run, gates. Commit** `feat: HF publisher with bounded retry`.

### Task 8: RunPod self-termination

**Files:**
- Create: `src/mprl/runpod.py`, `tests/unit/test_runpod.py`, `tests/network/test_runpod_terminate.py`

**Interfaces:**
- Produces:
  - `def own_pod_id(environ: Mapping[str, str], proc1_environ: Path = Path("/proc/1/environ")) -> str`. Reads `RUNPOD_POD_ID`, falls back to the NUL-separated `/proc/1/environ`, and raises `OwnPodUnknown` otherwise.
  - `class PodTerminator: __init__(self, client: httpx.Client, api_key: str, pod_id: str, *, base_url: str, max_attempts: int = 5, sleep=time.sleep)`. `terminate() -> None` sends `DELETE {base_url}/pods/{id}`, then `GET` must return 404 or `status == "TERMINATED"`. Retries 5xx and connection errors (bounded); raises `TerminateFailed`.
  - `RUNPOD_REST_BASE: Final[str]`, the value verified in Step 1.
- [ ] **Step 1: Verify the REST base URL and the DELETE/GET shapes** in RunPod's REST v2 docs (Context7 or the RunPod MCP `search-docs`). Put the URL and doc link in a comment next to `RUNPOD_REST_BASE`.
- [ ] **Step 2: Write the failing unit tests:**
  - `test_pod_id_from_env`
  - `test_pod_id_from_proc1_when_env_missing`: write a fake `environ` file in `tmp_path`.
  - `test_pod_id_missing_raises`: `pytest.raises(OwnPodUnknown, match="RUNPOD_POD_ID")`.
  - `test_terminate_confirms_with_read`, using `httpx.MockTransport`. Real `httpx` client and transport, not a hand mock. It asserts the DELETE path, the `Bearer` header, and the follow-up GET.
  - `test_terminate_retries_5xx_then_raises`.
- [ ] **Step 3: Run.** Expected: FAIL. **Implement**, run (PASS), gates.
- [ ] **Step 4: Network smoke test** (`network`, run once by the user's instruction on a throwaway pod they create, the cheapest Secure Cloud GPU): `test_terminate_real_pod` takes the pod id from env `MPRL_THROWAWAY_POD_ID`. It terminates the pod, then confirms with a read that it is gone. Document how to run it in `CLAUDE.md`; it never runs in CI.
- [ ] **Step 5: Commit** `feat: RunPod self-termination with confirmed read`.

### Task 9: One training run end-to-end (`train_run`)

**Files:**
- Create: `src/mprl/run_state.py`, `src/mprl/train.py`, `tests/unit/test_run_state.py`, `tests/smoke/test_train_run.py`

**Interfaces:**
- Consumes: `RunSpec`, `RunPaths`, `batch_arithmetic` (2); `TrainingClock`, `CheckpointStore` (3); `TimedOnPolicyRunner` (4); `TrainingDiagnostics` (5); `logger_cfg` (6); `HubPublisher` (7); `PodTerminator`, `own_pod_id` (8).
- Produces:
  - `@dataclass class RunState: run_id: str; config_hash: str; wandb_run_id: str; status: Literal["running", "complete", "failed"]; train_seconds: float; budget_minutes: float; git_sha: str; uploads: dict[str, str]` (artifact → `"ok"` or the error text). `load_state(path) -> RunState | None` and `save_state(path, state)` (atomic write).
  - `@dataclass(frozen=True) class TrainDeps: publisher: HubPublisher | None; wandb_mode: str; device: str; diagnostics_every_k: int = 10` (builder's call on K; document it)
  - `def train_run(spec: RunSpec, root: Path, deps: TrainDeps) -> RunState`
  - `def finish_and_terminate(states: list[RunState], terminator: PodTerminator | None) -> None`
- Behavior:
  1. If the state is `complete` and the stored budget ≥ `spec.budget_minutes`, return immediately (skip).
  2. If the stored `config_hash` ≠ `spec.config_hash()`, raise `ConfigMismatch` naming the run id.
  3. Otherwise build the env with `num_envs`, seed both agent and env with `spec.seed`, and set `agent_cfg["logger"] = logger_cfg(...)`.
  4. Load `store.latest()` into a **fresh** runner, restore the clock from `infos["clock"]`, and call `learn_for` with `budget_s = spec.budget_minutes * 60`.
  5. Write `state.json` (`complete`).
  6. Upload the final checkpoint folder via the publisher, and record the upload outcome in `state.uploads` without raising.
  7. Log the batch arithmetic, git SHA and image digest (env `MPRL_IMAGE_DIGEST`, if set) to the W&B config.
- [ ] **Step 1: Write the failing unit tests** for `run_state`: round trip; atomic write ignores tmp; `status` validated.
- [ ] **Step 2: Write the failing smoke tests** (real mjlab CPU, `num_envs=4`, W&B `offline`, publisher `None`, budgets of a few seconds):
  - `test_fresh_run_completes_and_writes_state`
  - `test_completed_run_is_skipped` (Review Focus 3): the second call returns at once and the checkpoint count is unchanged.
  - `test_raised_budget_continues` (Review Focus 3): budget 2 s then 4 s. Final `train_seconds >= 4`, the same `wandb_run_id`, and the first checkpoint from run 1 is still present.
  - `test_resume_after_kill_continues_clock` (Review Focus 1): run in a subprocess, kill it with SIGKILL after the first checkpoint, and drop a fake partial tmp file. Re-running resumes from the last complete checkpoint, with `train_seconds` ≥ that checkpoint's. Bound the subprocess wait with a timeout.
  - `test_resume_refuses_config_mismatch` (Review Focus 2): `pytest.raises(ConfigMismatch, match="main-n00004-s0")`.
  - `test_terminate_runs_after_failed_upload` (Review Focus 4): use a publisher built on an autospecced `HfApi` raising the real 503 type, and a terminator built on `httpx.MockTransport`. The upload failure is recorded in state, **then** the DELETE is sent. The ordering is asserted by a shared event list.
  - `test_only_num_envs_and_seed_differ_between_sweep_runs`: build the resolved `agent_cfg` and `env_cfg` dicts for two `RunSpec`s (64/s0 and 4096/s1) and diff them. The only differences are `scene.num_envs`, the seeds and the logger run id.
- [ ] **Step 3: Run.** Expected: FAIL. **Implement**, run (PASS), gates.
- [ ] **Step 4: Commit** `feat: train_run with resume, skip, state, uploads and termination ordering`.

### Task 10: Budget, throughput test, and sweep orchestration

**Files:**
- Create: `src/mprl/budget.py`, `src/mprl/throughput.py`, `src/mprl/sweep.py`, `tests/unit/test_budget.py`, `tests/unit/test_sweep.py`, `tests/smoke/test_throughput_cpu.py`

**Interfaces:**
- Consumes: `SweepSpec`, `RunSpec` (2); `train_run`, `finish_and_terminate`, `RunState` (9).
- Produces:
  - `@dataclass(frozen=True) class Measurements: eval_seconds_per_checkpoint: float; io_overhead_fraction: float; fits: dict[int, bool]; steps_per_second: dict[int, float]`, loaded from the throughput/pilot JSON.
  - `@dataclass(frozen=True) class CostEstimate: lines: tuple[tuple[str, float], ...]; total_hours: float; total_usd: float`
  - `def estimate(sweep: SweepSpec, m: Measurements, price_per_hour: float = 1.79, setup_hours: float = 4.25, long_run_done_minutes: float = 0.0) -> CostEstimate`. Here `setup_hours` = debugging 4 + throughput 0.25 from the spec's table. Sweep hours = runs × budget × (1 + io_overhead_fraction) + evaluations × eval_seconds_per_checkpoint, with evaluations every `eval_every_minutes`.
  - `CUT_ORDER: Final = ("eval_every_6min", "long_run_half", "drop_16384", "two_seeds")`
  - `def plan_within_budget(sweep: SweepSpec, m: Measurements, cap_usd: float = 40.0) -> tuple[SweepSpec, tuple[str, ...], CostEstimate]`. Applies cuts in order until within the cap; raises `BudgetExceeded` with the estimate if still over. 64 is never removed.
  - `def measure_throughput(env_counts: Sequence[int], iterations: int, device: str) -> Measurements`. A CUDA `torch.OutOfMemoryError` for a count records `fits[n] = False` and continues. **Verify the OOM exception class in the installed torch.**
  - `def run_sweep(sweep: SweepSpec, root: Path, deps: TrainDeps, m: Measurements, *, confirm: Callable[[CostEstimate, tuple[str, ...]], bool], terminator: PodTerminator | None, reuse_pilot: bool) -> list[RunState]`
  - Pilot: `run_sweep` with `sweep.long_run()` at `budget_minutes=45.0` is the pilot. With `reuse_pilot`, if the pilot's `config_hash` and `git_sha` equal those of the sweep's `n=4096, s=0` spec (with the sweep name substituted) and the current HEAD, a manifest `sweep_results_dir/.../pilot_reuse.json` maps that run id to the pilot's run dir, capped at 45 minutes, and the run is skipped.
- [ ] **Step 1: Write the failing unit tests:**
  - `test_estimate_matches_spec_table`: with `io_overhead_fraction=0` and `eval_seconds_per_checkpoint=0`, the total hours are 18.5 and `total_usd == pytest.approx(33.115)` (spec table: 0.25 + 11.25 + 3 + 4).
  - `test_cuts_apply_in_order`: a measurement that is too expensive returns the cuts `("eval_every_6min", "long_run_half")` exactly when two suffice.
  - `test_64_never_dropped`
  - `test_budget_exceeded_after_all_cuts`
  - `test_sweep_excludes_counts_that_do_not_fit` (Review Focus 5): `fits={16384: False}` gives runs without 16384, and the reason is recorded.
  - `test_sweep_requires_confirmation_when_cuts_applied`: `confirm` returning False → no `train_run` call. Use an autospecced `train_run` *only after* Task 9's smoke tests have pinned its real behavior.
  - `test_pilot_reused_only_when_hash_and_sha_match`
  - `test_sweep_terminates_pod_on_crash`: a run raising `RuntimeError` still triggers `finish_and_terminate`, then re-raises.
- [ ] **Step 2: Write the failing smoke test** `test_oom_count_recorded_not_raised` (Review Focus 5). Run `measure_throughput` on CPU with counts `(4, 8)`, injecting OOM for 8 via a wrapper that raises the real `torch.OutOfMemoryError` class (verified). `fits == {4: True, 8: False}` and `steps_per_second[4] > 0`.
- [ ] **Step 3: Run.** Expected: FAIL. **Implement**, run (PASS), gates.
- [ ] **Step 4: Commit** `feat: budget with cut order, throughput measurement, sweep orchestration`.

### Task 11: Fixed-command evaluator

**Files:**
- Create: `src/mprl/evaluator.py`, `tests/unit/test_eval_commands.py`, `tests/smoke/test_evaluator.py`

**Interfaces:**
- Consumes: `CheckpointRef` (3).
- Produces:
  - `EVAL_COMMANDS: Final[tuple[tuple[float, float, float], ...]] = ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.0, -0.5))`. These are `(lin_x, lin_y, ang_z)`, all inside curriculum stage 0 (lin_vel_x ±1.0, ang_vel_z ±0.5), so every checkpoint is asked for things it was trained on from the start.
  - `@dataclass(frozen=True) class EvalConfig: commands = EVAL_COMMANDS; seeds: tuple[int, ...] = (0, 1); envs_per_command: int = 4; horizon_s: float = 10.0`
  - `@dataclass(frozen=True) class EvalResult: run_id: str; iteration: int; train_minutes: float; vel_error_xy: float; vel_error_yaw: float; no_fall_fraction: float; mean_time_to_fall_s: float; mean_tracking_reward: float`
  - `def evaluate_checkpoint(run_id: str, ref: CheckpointRef, cfg: EvalConfig, device: str) -> EvalResult`
  - `def write_eval_csv(results: Sequence[EvalResult], path: Path) -> None` and `def read_eval_csv(path: Path) -> list[EvalResult]`
- [ ] **Step 1: Write the failing unit tests:**
  - `test_eval_commands_inside_stage0_ranges`: parametrized over commands; each is within ±1.0 / ±0.5.
  - `test_eval_csv_round_trip`
- [ ] **Step 2: Write the failing smoke tests** (CPU; checkpoint produced by a 2-iteration run in a session-scoped fixture):
  - `test_commands_are_held_fixed`: the env's command tensor after the rollout equals the requested commands, per env.
  - `test_same_checkpoint_same_result`: two evaluations give equal results (or document non-determinism with evidence, as in Task 4).
  - `test_result_metrics_finite_and_bounded`: `0 <= no_fall_fraction <= 1`.
  - `test_eval_does_not_depend_on_training_num_envs`: checkpoints from 4-env and 8-env runs evaluate with the same env count and command tensor.
- [ ] **Step 3: Implement.** Start from `load_env_cfg(TASK_ID, play=True)` and set `num_envs = len(commands) * envs_per_command`. **Read `mjlab/tasks/velocity/mdp/velocity_command.py`** to find how to pin per-env commands and disable resampling (e.g. a resampling time longer than the horizon plus writing the command buffer after reset). Detect falls from the `fell_over` termination term. Load the policy with a fresh runner's `load` + `get_inference_policy(device)`.
- [ ] **Step 4: Run** (PASS), gates. **Commit** `feat: fixed-command checkpoint evaluator`.

### Task 12: Rollout rendering and the race video

**Files:**
- Create: `src/mprl/render.py`, `src/mprl/race_video.py`, `tests/unit/test_race_video.py`, `tests/smoke/test_render.py`

**Interfaces:**
- Consumes: `CheckpointRef` (3), `EVAL_COMMANDS` (11).
- Produces:
  - `def render_rollout(ref: CheckpointRef, command: tuple[float, float, float], seconds: float, *, height: int = 360, width: int = 480, seed: int = 0, device: str = "cpu") -> np.ndarray`. Returns shape `(T, height, width, 3)` uint8, with `T = round(seconds / step_dt)` and `step_dt` read from the env.
  - `@dataclass(frozen=True) class Panel: label: str; frames: np.ndarray`
  - `def compose_frame(panels: Sequence[Panel], clock_label: str, *, columns: int) -> np.ndarray` lays panels in a grid. Each panel's label (e.g. `"4,096 robots"`) sits in its corner, with one shared clock banner on top (e.g. `"Training time 12:00 / 45:00"`). Uses Pillow for text.
  - `def race_timeline(checkpoints: Mapping[int, CheckpointStore], marks_minutes: Sequence[float]) -> list[tuple[float, dict[int, CheckpointRef | None]]]`. For each mark, the checkpoint at or before that mark for each env count; `None` before the first checkpoint shows as "starting".
  - `def write_video(frames: Iterable[np.ndarray], path: Path, fps: int) -> None` via `imageio` (ffmpeg).
- [ ] **Step 1: Write the failing unit tests** (synthetic arrays):
  - `test_compose_frame_shape`: 5 panels of 360×480 with 3 columns give the expected banner + 2×3 grid size.
  - `test_panel_order_preserved`: panel i is filled with color i; sample its centre pixel.
  - `test_timeline_picks_checkpoint_at_or_before_mark`
  - `test_timeline_none_before_first_checkpoint`
  - `test_write_video_round_trip`: write, then read the frame count back with imageio.
- [ ] **Step 2: Write the failing smoke test** `test_render_rollout_cpu`: a real 2-iteration checkpoint renders `(T, 360, 480, 3)` uint8, not all-black (`frames.max() > 0`). Set the viewer height and width through the env cfg's `viewer` (verified fields `height`, `width`).
- [ ] **Step 3: Implement**, run (PASS), gates. **Commit** `feat: rollout renderer and race-video composer`.

### Task 13: Learning curves and results publishing

**Files:**
- Create: `src/mprl/curves.py`, `src/mprl/results.py`, `tests/unit/test_curves.py`, `tests/smoke/test_results_cpu.py`

**Interfaces:**
- Consumes: `read_eval_csv`, `EvalResult` (11); `race_timeline`, `compose_frame`, `render_rollout`, `write_video` (12); `HubPublisher` (7); `sweep_results_dir` (2).
- Produces:
  - `@dataclass(frozen=True) class CurvePoint: num_envs: int; minutes: float; mean: float; low: float; high: float; n_seeds: int`
  - `def aggregate(results: Sequence[EvalResult], metric: str, run_envs: Mapping[str, int], marks_minutes: Sequence[float]) -> list[CurvePoint]`. Mean across seeds; the band is min–max. **(call: with 3 seeds an IQM is just the median, so mean + range is more informative; state this in the chart caption.)**
  - `def median_seed_run(results, metric: str, run_envs, at_minutes: float) -> dict[int, str]`. Per env count, the run id whose final metric is the median.
  - `def plot_learning_curves(points: Sequence[CurvePoint], metric_label: str, out: Path) -> None` (matplotlib PNG + SVG). One line per count with the shaded band; x = training-clock minutes; caption with batch arithmetic per count.
  - `def build_results(sweep: SweepSpec, root: Path, *, primary_metric: str, publisher: HubPublisher | None, wandb_mode: str, device: str) -> Path`. Steps:
    1. Evaluate any un-evaluated checkpoint at `eval_every_minutes` marks.
    2. Write `results.csv`.
    3. Write the chart.
    4. Render the race video from the median seeds.
    5. Write a `README.md` model card: the setup, the batch arithmetic table, which cuts were applied, disclosure of the `command_vel` curriculum, and the evaluation commands.
    6. Log chart and video to a W&B `results` run in the sweep group.
    7. Upload the results dir to HF. Re-running is idempotent.
- [ ] **Step 1: Write the failing unit tests:**
  - `test_aggregate_mean_and_range`: hand-computed values for 3 seeds.
  - `test_aggregate_missing_seed_reduces_n`
  - `test_median_seed_selection`
  - `test_plot_writes_png_and_svg`: files exist and are non-empty.
  - `test_metric_name_validated`: an unknown metric raises `ValueError` matching the valid names.
- [ ] **Step 2: Write the failing smoke test** `test_build_results_end_to_end_cpu`. A tiny `SweepSpec` (counts `(4, 8)`, seeds `(0, 1)`, budget 0.1 min, checkpoint every 0.05 min, short render) trained via `train_run` on CPU with W&B offline and no publisher. The results dir contains `results.csv`, `learning_curves.png`, `race.mp4` and `README.md`; the video frame count is > 0.
- [ ] **Step 3: Implement**, run (PASS), gates. **Commit** `feat: learning curves, race video and results publishing`.

### Task 14: `.env`, setup, preflight, CLI and bash entry points

**Files:**
- Create: `.env.example`, `src/mprl/env_file.py`, `src/mprl/preflight.py`, `src/mprl/cli.py`, `scripts/setup.sh`, `scripts/train.sh`, `tests/unit/test_env_file.py`, `tests/unit/test_preflight.py`, `tests/smoke/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `.env.example` keys, each with a comment line saying where to get it: `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT=mprl-humanoid-walk`, `HF_TOKEN`, `HF_REPO_ID`, `RUNPOD_API_KEY`, `MPRL_VOLUME_ROOT=/workspace/mprl`.
  - `@dataclass(frozen=True) class EnvVar: name: str; default: str | None; comment: str; secret: bool`, where `secret` is true when the name ends in `_KEY` or `_TOKEN`. `def parse_env_example(text: str) -> list[EnvVar]`. `def render_env(vars: Sequence[EnvVar], values: Mapping[str, str]) -> str`. `def write_env(path: Path, content: str, *, overwrite: bool) -> None` writes mode `0o600` and raises `FileExistsError` unless `overwrite`.
  - `@dataclass(frozen=True) class CheckResult: name: str; ok: bool; detail: str`. `def run_checks(checks: Sequence[tuple[str, Callable[[], str]]]) -> list[CheckResult]`: each check returns its detail or raises; operating errors become `ok=False`, and nothing is swallowed silently.
  - Real checks, as `default_checks(env: Mapping[str, str], require_gpu: bool)`:
    - CUDA visible; MuJoCo Warp runs on the device
    - G1 env with `num_envs=2` constructs and steps
    - W&B auth via `wandb.Api().viewer`. Verify the attribute in the installed wandb.
    - HF `whoami` plus `ensure_repo`
    - RunPod own pod id resolvable and `GET` of the own pod returns 200
    - Volume writable: write, read back and delete a temp file under `MPRL_VOLUME_ROOT`
  - CLI `mprl` (tyro subcommands):
    - `setup`: prompts with `getpass` for secrets and `input` for the rest; shows defaults; writes `.env` next to the repo; then runs preflight.
    - `preflight [--cpu]`
    - `throughput --counts ... --iterations 20`: writes `measurements.json` under the volume.
    - `train-run --sweep --num-envs --seed --minutes [--no-terminate]`
    - `pilot [--no-terminate]`
    - `sweep --name [--reuse-pilot] [--no-terminate] [--yes]`
    - `results --name`
  - `train-run`, `pilot` and `sweep` call `own_pod_id` **before** training and refuse to start unless `--no-terminate` is given. Fail fast before spending.
  - `scripts/setup.sh` (bash, `set -euo pipefail`):
    1. Check it is running on the pod (`/workspace` mounted).
    2. `UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen` (installs only the project; dependencies are already baked in).
    3. `exec uv run --no-sync mprl setup "$@"`.
  - `scripts/train.sh` (bash, strict): `exec uv run --no-sync mprl "${@:-sweep --name main}"`, with `.env` loaded by the CLI (python-dotenv).
- [ ] **Step 1: Write the failing unit tests:**
  - `test_parse_env_example_marks_secrets`
  - `test_render_env_preserves_order_and_comments`
  - `test_write_env_refuses_overwrite` (`pytest.raises(FileExistsError)`)
  - `test_write_env_mode_0600`
  - `test_run_checks_reports_failure_detail`: a check raising `MissingSecret("HF_TOKEN")` gives `ok=False` with the detail mentioning `HF_TOKEN`.
  - `test_run_checks_does_not_swallow_programmer_errors`: an `AssertionError` from `require` propagates.
- [ ] **Step 2: Write the failing smoke tests:**
  - `test_cli_help_lists_subcommands` (subprocess `uv run mprl --help`)
  - `test_train_run_refuses_without_pod_id`: env without `RUNPOD_POD_ID`, no `--no-terminate` → exit code ≠ 0 and the message contains `--no-terminate`.
  - `test_preflight_cpu_passes_local_checks`: `--cpu` with only the volume and env checks enabled, root in `tmp_path`.
  - `shellcheck scripts/*.sh` exits 0 (use the bash-scripting-specialist skill's audit).
- [ ] **Step 3: Implement**, run (PASS), gates. **Commit** `feat: env setup, preflight checks, CLI and entry scripts`.

### Task 15: Docker image and CI

**Files:**
- Create: `docker/Dockerfile`, `.dockerignore`, `.github/workflows/tests.yml`, `.github/workflows/image.yml`

**Interfaces:**
- Produces: image `ghcr.io/<owner>/massively-parallel-rl:{sha,latest}`, with dependencies in `/opt/venv`, `MPRL_IMAGE_DIGEST` available at runtime (build arg → env), and the RunPod base startup preserved.
- [ ] **Step 1: Choose the base.** Use the dockerfile-expert skill and the RunPod `runpod-usage/building-images` guide. Pick a pinned `runpod/pytorch:<tag>` whose CUDA works with the torch 2.14.1 wheel in `uv.lock` (check the torch CUDA variant in the lock), or a pinned CUDA runtime base plus RunPod's SSH start pattern. Do not override `CMD`/`ENTRYPOINT`; inherit `/start.sh`. Record the tag, its digest and the matching pod `minCudaVersion` in `CLAUDE.md`.
- [ ] **Step 2: Write the Dockerfile.**
  - Install uv (pinned).
  - `COPY pyproject.toml uv.lock` → `UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-install-project --no-dev`, using a BuildKit cache mount.
  - Then copy the source and `uv sync --frozen --no-dev`.
  - `ENV PYTHONUNBUFFERED=1 PATH=/opt/venv/bin:$PATH`.
  - No secrets.
  
  Build for `linux/amd64`.
- [ ] **Step 3: Verify locally:**
  - `docker build --platform linux/amd64 -f docker/Dockerfile .` succeeds.
  - `docker run --rm --platform linux/amd64 <img> /opt/venv/bin/python -c "import mjlab, rsl_rl, wandb, huggingface_hub"` exits 0.
  - `docker run ... uv run --no-sync pytest -m smoke -q` passes on CPU (if Task 1's platform matrix said Linux CPU works).
  - Run the dockerfile-expert linter. No findings.
- [ ] **Step 4: `tests.yml`:** on push/PR, `ubuntu-latest`. `astral-sh/setup-uv` pinned by SHA, `uv sync --frozen`, then the audits, ruff, pyright and `uv run pytest -m "not network and not gpu"`. Harden per pytest-expert's `assets/github/workflows/tests.yml` (pinned SHAs, `permissions: contents: read`).
- [ ] **Step 5: `image.yml`:** on push to `main`, after tests, `docker/build-push-action` (pinned SHA) to GHCR with `GITHUB_TOKEN` (`packages: write` only in this job), tags `sha` and `latest`, and build arg `MPRL_IMAGE_DIGEST`. Make the package public (one-time manual step in GHCR settings, documented in the README).
- [ ] **Step 6: Verify** with `actionlint` and `zizmor` (no findings). Push to GitHub. Both workflows go green; `docker pull ghcr.io/<owner>/massively-parallel-rl:latest` works anonymously.
- [ ] **Step 7: Commit** `ci: test workflow and GHCR image build`.

### Task 16: Pod session — GPU smoke tests, throughput test, pilot (paid; needs the user)

**Files:**
- Create: `tests/gpu/test_gpu_smoke.py`, `docs/runbook.md`
- Modify: `CLAUDE.md` (Commands, Architecture, measured numbers)

**Interfaces:**
- Consumes: the CLI (Task 14), the image (Task 15).

- [ ] **Step 1: Write the GPU smoke tests** (`gpu`):
  - `test_cuda_training_iteration`: 2 iterations at `num_envs=4096` on `cuda:0`, finite losses.
  - `test_clock_syncs_on_cuda`: `cuda_sync_for("cuda:0")` is `torch.cuda.synchronize`.
  - `test_checkpoint_write_to_volume`: a save + load round trip under `MPRL_VOLUME_ROOT`.
- [ ] **Step 2: Write `docs/runbook.md`.** Cover:
  1. Read the network-volume price (RunPod MCP or console) and create a 20 GB **(call)** volume in a data center that lists A100 80GB Secure Cloud stock.
  2. Create the A100 pod from `ghcr.io/<owner>/massively-parallel-rl:latest` with the volume at `/workspace` and the `minCudaVersion` from Task 15.
  3. `git clone` into `/workspace/massively-parallel-rl`, then `scripts/setup.sh`.
  4. `uv run --no-sync pytest -m gpu`.
  5. `scripts/train.sh throughput --counts 64 256 1024 4096 16384`.
  6. `scripts/train.sh pilot`.
  7. Check the pilot gate (does the G1 walk at 45 min? Look at eval `no_fall_fraction` and `vel_error_xy`).
  8. `scripts/train.sh sweep --name main --reuse-pilot`.
  9. `scripts/train.sh results --name main`.
  10. Resume the long run to 180 min.
  
  Every paid step lists the price read and the expected hours.
- [ ] **Step 3: With the user** (they create the pod and approve spend), run steps 4–7. Quote the real outputs: steps/s per count, which counts fit, the measured overheads, and the pilot's eval metrics. Feed them into `estimate` and show the user the cost and any cuts **before** the sweep.
- [ ] **Step 4: Update `CLAUDE.md`** Architecture (module map from this plan, data flow: CLI → sweep → train_run → TimedOnPolicyRunner → store/W&B/HF → results) and the measured numbers with their dates. **Commit** `docs: runbook, GPU smoke tests, measured throughput and pilot`.
- [ ] **Step 5: Sweep, results and long run** follow the runbook after the user confirms the estimate. These may finish after 2026-11-03; the proposal cites the pilot and whatever results exist.

---

## Self-review notes

- Spec coverage, by section:
  - §2 goals: Tasks 9–13.
  - §3 constraints: auto-terminate (8, 9, 10, 14); public GHCR (15); CPU-first testing (1, matrix); deadline order (task order + 16).
  - §4 clock, checkpoints, curriculum disclosure, evaluator, race video: Tasks 3, 4, 11, 12 and 13 (README disclosure).
  - §5 budget, pilot gate, cut order, pilot reuse: Tasks 10 and 16.
  - §6 setup and training flow: Tasks 14–16.
  - §7 observability: rsl_rl Logger built-ins + Task 5 + Task 6. Termination reasons: verify that `Episode_Termination/*` (or equivalent) keys appear in `extras["log"]` during Task 5's smoke test; add them from the termination manager if absent.
  - §8 storage and uploads: Tasks 2, 7, 9 and 13.
  - §9 testing: every task.
- Type names are consistent across tasks: `RunSpec`, `SweepSpec`, `CheckpointRef`, `CheckpointStore`, `TrainingClock`, `TrainDeps`, `RunState`, `Measurements`, `EvalResult`, `HubPublisher`, `PodTerminator`.
