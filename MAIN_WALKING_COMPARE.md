# Main walking comparison preparation

This worktree prepares **`main-walking-compare`**, based on local `main` at
**`f81679b72cca2cb227e190beeea18eed7894c10c`**, for a fresh comparison with the
completed published-era walking run. Source, environments, and results stay in
ToddlerBot. The repository is at `/Users/ygli/projects/toddlerbot`; the paper
analysis is in `PAPER_WALKING_RESULTS.md` on `paper-walking-baseline`.

## Experiment

Run main's current `toddlerbot_2xc` walking implementation with **native Brax PPO**,
fresh **seed 0**, and **300M requested environment transitions**. Load
`walk.gin`, then `main_compare.gin`; the native entry point loads `walk` first.

| Setting | Prepared main comparison | Completed paper run |
| --- | --- | --- |
| Robot | `toddlerbot_2xc` | Published-era `toddlerbot` |
| Budget / seed | 300M / 0 | 300M / 0 |
| Collection / optimization | 1,024 envs; 20-step rollout; 256 × 4 minibatches; 4 passes | Same |
| Actor / critic | 512–256–128; ELU; 1,260 / 2,265 inputs | Same widths; Swish; 1,245 / 1,890 inputs |
| Action distribution | Normal; independent learned log std, initial std 0.5 | State-dependent Normal followed by tanh |
| Learning rate | 3e-5, constant native Brax schedule | 1e-4 |
| XY reward | Weight 2; `exp(−1000 e²)` | Weight 5; `exp(−100 e²)` |
| Leg imitation | `motor_pos` weight 0; home-pose penalty 0.5 | Motor reference MSE weight 5 |
| Survival training | Alive weight 1; native height [0.20, 1.0] m | Fall penalty weight 10; [0.20, 0.40] m |
| Commands | X ±0.10, Y ±0.05 m/s; yaw ±1.0 rad/s | X [−0.10, +0.20], Y ±0.10; yaw ±0.50 |
| Control / physics | 5 ms × 4; MuJoCo/MJX 3.3.4 | 4 ms × 5; MuJoCo/MJX 3.2.4 |
| Runtime | Python 3.11; Brax 0.14.2; JAX 0.6.2 | Python 3.10; pinned Brax fork 0.10.5; JAX 0.6.2 Blackwell override |

Main's active walking rewards, model, observation history, command sampling and
training randomization come from its existing source and `walk.gin`. Brax's native
random-command evaluations are disabled (`run_evals=False`); the quantitative
suite below evaluates each exported policy. Timeout bootstrapping is explicitly
disabled, matching main's existing native call. Main's default RSL-RL backend is
a separate experiment.

The batch settings satisfy main's `num_envs == batch_size * num_minibatches`
constraint. Native Brax rounds the requested budget to whole batches: this run
exports step **0** and **15 trained policies**, every **20,008,960** transitions,
through **300,134,400** transitions. Checkpoint counters encode actual collected
transitions; optimization passes do not multiply that counter.

## GPU machine setup

The preparation branch is local. A Git bundle can transfer its commits without
publishing a branch. Copy `.worktrees/main-walking-compare.bundle` from the Mac to
the GPU machine, then run these commands in the GPU ToddlerBot clone:

```bash
git bundle verify /tmp/main-walking-compare.bundle
git fetch /tmp/main-walking-compare.bundle main-walking-compare:main-walking-compare
git worktree add .worktrees/main-walking-compare main-walking-compare
cd .worktrees/main-walking-compare
```

The bundle requires base commit `f81679b`; the GPU clone's previously reported
main already has it. Git verifies that prerequisite before importing. This
local comparison branch has no remote tracking branch until explicitly published.

Prepare its separate environment once:

```bash
uv venv --python 3.11 .venv-main-compare
uv pip sync --python .venv-main-compare/bin/python requirements-main-compare.lock
export UV_PROJECT_ENVIRONMENT=.venv-main-compare UV_PYTHON=3.11 WANDB_MODE=offline
```

The Linux lock targets CUDA 12 and includes NVCC **12.9.86** and cuDNN **9.8.0.87**
for the RTX 5070. Brax 0.14.2 requires Python 3.11 and supports main's
`run_evals`, `value_obs_key`, and Normal distribution options. The existing
paper Python 3.10 environment uses a different Brax API.

Use `--no-sync` when running this focused environment: the project dependency
file still specifies JAX 0.4.28. The training locks are installed explicitly with
`uv pip sync` and checked by the preflight. Avoid `uv sync` in this environment.

### Verify on the actual GPU

```bash
export UV_PROJECT_ENVIRONMENT=.venv-main-compare UV_PYTHON=3.11 WANDB_MODE=offline
uv run --no-sync -m toddlerbot.locomotion.main_environment
uv run --no-sync -m toddlerbot.locomotion.train_mjx --brax --gin-file "main_compare main_compare_smoke" --skip-video
```

The preflight verifies the checkout lineage, source import location, installed
lock pins, dependency metadata, native Brax APIs, GPU backend and a compiled JAX
matrix operation. The smoke run performs native PPO updates, exports finite
policies and executes all 15 evaluator conditions in both profiles.

The smoke configuration uses **40 transitions, 2 environments and a 20-step
evaluation horizon**. It verifies execution; it does not reach the 3-second
transition windows, measure learned walking quality, or prove memory capacity
at the full 1,024-environment batch size. First compilation can take minutes.

### Full training command

Set the environment in the same terminal used to launch training:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-main-compare UV_PYTHON=3.11 WANDB_MODE=offline
uv run --no-sync -m toddlerbot.locomotion.train_mjx --brax --gin-file main_compare --skip-video
```

No restore argument is supplied: this initializes a fresh seed-0 policy. The
comparison path checks its runtime again at launch and rejects a CPU backend
for a long run. The Mac has no verified NVIDIA GPU; full GPU training is pending.

### JAX / CUDA plugin mismatch

If a traceback names `.venv/lib/python3.11/site-packages`, `uv run` selected the
general project environment. Environment exports apply only to the current
terminal. Set the export above in each new terminal before running the comparison.

An error such as `Mismatched PJRT plugin PJRT API version` or a CUDA 13 plugin's
missing `register_custom_type_handler` indicates incompatible JAX packages in
the selected environment. Prepare `.venv-main-compare` with the Linux lock above;
its JAX, JAXlib and CUDA 12 plugins are all pinned to **0.6.2**. `uv pip sync`
removes extra packages from that separate environment. The comparison checks
dependencies and rejects unpinned CUDA plugins before creating robot JAX arrays.
Confirm that preflight prints a Python path inside `.venv-main-compare`, then
complete the GPU smoke check before launching 300M steps.

## Measurements and artifacts

Each policy runs **15 conditions × 128 episodes × 1,000 control steps** with
deterministic inference in two profiles:

- **Nominal:** flat terrain, home initialization at rest, nominal model/actuators,
  observation noise, calibration variation and pushes disabled.
- **Native randomization:** main's existing model, reset, actuator and observation
  distributions. Main's current `add_push=False` remains the profile's push setting.

Both profiles stop at a shared **torso height outside [0.20, 0.40] m** on flat
ground. Automatic reset and main's internal motion teleport are disabled in
the measurement rollout. `native_terminal_count_before_benchmark_stop` also
records whether main's original height bound was crossed before this stop;
it is not the result of a second, longer rollout with native termination.

The signed commands match the paper evaluator: forward/backward 0.10 and 0.05
m/s, left/right 0.05, four diagonals, standing and four start/stop schedules.
Transitions occur at **3, 9, 12 and 17 seconds**, with separate 0.5-second start
and stop measurement windows. All, standing, walking, start and stop groups retain:

- Signed mean local X/Y velocity, instantaneous vector tracking error and speed.
- Leg motor RMSE against this implementation's own ZMP reference.
- Per-episode length, falls, nonfinite events, eligible episode/sample counts,
  individual measurements, means and population standard deviations.

Finite terminal samples are included, then the episode is frozen. Nonfinite
samples are excluded and recorded separately. Measurements are normalized within
each eligible episode before averaging episodes. Nominal fixes model/reset
settings, but the GPU archive records small variation between batch instances.
Their standard deviation does not measure uncertainty across independent
training seeds.

Evaluation seed **12102** provides independent evaluation model/reset draws;
the same draws are reused across conditions and checkpoints. Host mass-grid
permutations are explicitly seeded. Main's different randomization distributions
do not create equivalent perturbations to the paper run merely by sharing a seed.

Results are written under `results/toddlerbot_2xc_walk_brax_<timestamp>/`:

| Artifact | Contents |
| --- | --- |
| `ckpt/model_<actual_steps>` | Every native policy export, including step 0 |
| `checkpoints.jsonl` | Actual counter, file, SHA-256 and parameter finiteness |
| `metrics.jsonl` | Unmodified numeric Brax progress metrics and elapsed time |
| `xy_evaluation.jsonl` | Both profiles at each checkpoint, including per-episode arrays |
| `train_config.json`, `env_config.json`, `args.json` | Effective experiment settings |
| `provenance.json`, `source/` | Runtime, device, lock/model/reference hashes and source snapshots |
| `model_best`, `model_last` | Native training-reward-selected and final exports |
| `output.log`, `git/` | Console and Git diff inside the result directory |
| Worktree root `wandb/` | Offline SDK records |

`model_best` follows main's training episode return, with the nearest retained
checkpoint selected; it is not selected by the signed-XY benchmark. Rank retained
policies by tracking, survival and standing measurements, and confirm candidates
on a new held-out evaluation seed. Nonfinite numeric training metrics, exported
weights or measurement rollouts stop the comparison before checkpoint promotion.

Re-evaluate a retained main policy without Torch conversion or videos:

```bash
uv run --no-sync -m toddlerbot.locomotion.train_mjx --brax --gin-file main_compare --eval results/<run>/ckpt/model_200089600 --gin-config "PPOConfig.evaluation_seed=22102" --skip-video
```

This writes `xy_<checkpoint-name>_seed22102.json` beside the checkpoint with both profiles and the policy
hash. Use the saved run's network/environment settings when overriding this
command for a different experiment.

## Reading the eventual comparison

Evaluate policies near **180M, 200M, 240M and 300M**. The completed paper run's
180M policy had the fewest native-variation falls; 200M had the lowest fixed-command
tracking error; native reward selected 240M; final 300M regressed in survival.
The paper's earlier result document retains those measurements and limitations.

Compare physical metrics and native reward curves within their own recipes.
The existing paper measurements include its native pushes/noise. Re-evaluate
paper policies under the nominal profile before presenting a controlled nominal
comparison. A robustness comparison also needs common perturbation distributions
and a shared reference to isolate imitation differences. Own-reference leg RMSE
measures each policy's imitation against its own target.

The assembled main model's nominal COM height is recorded in provenance; normalize
velocity and timing as `v / sqrt(g h)` and `T / sqrt(h / g)` when comparing geometry.
This first pair changes robot, observations, rewards and runtime together. A
recipe-only comparison requires training both recipes on the same model/runtime;
repeatability requires additional independent training seeds.

## Local verification

The separate Mac CPU environment uses `requirements-main-compare-cpu.lock`.
Its setup is the same as above, using that lock, and `WANDB_MODE=disabled` for
local execution checks. GPU offline W&B is configured by the GPU command above;
local JSON artifacts are also saved when SDK tracking is disabled.

```bash
uv run --no-sync -m toddlerbot.locomotion.main_environment --cpu
uv run --no-sync -m pytest -q tests/test_walk_reference_geometry.py tests/test_main_comparison.py
uv run --no-sync -m toddlerbot.locomotion.train_mjx --brax --gin-file "main_compare main_compare_smoke" --skip-video
```

Verified on **2026-10-16**, with code commit **`e70e949`**:

- All **102 CPU lock pins**, dependency metadata, source imports, native Brax APIs
  and CPU matrix JIT passed.
- **4 tests passed**, including the existing 2xc/2xm reference geometry gate and
  measurement timing/history/aggregation/termination gates.
- Native PPO completed **40 steps**; step-0 and step-40 exports were finite. All
  **60 condition rows** across its four checkpoint/profile suites were finite.
- The retained step-40 policy was re-evaluated on the committed source after the
  nominal-at-rest refinement. Both profiles passed all 15 conditions; native
  per-episode measurements exactly replayed the training-time results.
- Native disabled-mode SDK logging passed with TensorBoard syncing disabled.
  The Mac's offline `wandb-core` process exited with `-9` in the first setup
  attempt; CPU execution checks therefore used `WANDB_MODE=disabled`.

The [verification record](docs/main_walking_compare_20261016/verification.json)
contains source/runtime hashes, commands, checkpoint checks and raw smoke losses.
The smoke source is frozen under
`results/toddlerbot_2xc_walk_brax_20261016_113551/source/`; its exact preparation
state is retained in that run's provenance. The standalone evaluation records the
committed evaluator revision separately.

**Actual GPU verification and full training are pending.** CPU execution checks
establish code/runtime correctness; walking quality is measured after training.
