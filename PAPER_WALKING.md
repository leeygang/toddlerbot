# Paper walking baseline

Run the experiment directly in this ToddlerBot repository on branch
`paper-walking-baseline`. The branch starts at the local `v1.0.0` release,
`59832cde2be3ab029486958f2e5ba92604f87307`, and retains its Brax gitlink,
`14147806f6c25724ba830ba36764df3243889b0d`.

The reference is the [ToddlerBot paper, Tables VI and VII](https://arxiv.org/html/2502.00893v2).
`paper.gin` sets all paper reward weights and a fresh 300M-step run, seed 0.
Native actor initialization, home + 0.25 rad bounded actions, robot model,
0.72 s gait, ZMP lookup, noise, randomization and command sampling are retained.
The original ZMP cache SHA256 is
`43b1363638bf974c52598ff92b1981fad7b43fe63d8b5c9787abfd617b370b39`.

## Install the training environment

Install on each machine that will run training or tests. The environment
prepared on the Mac does not install dependencies on the GPU host. Run from
the repository root; on the GPU host this is `~/projects/toddlerbot`.

Fetch the published branch and verify its history before installing. The
prepared code commit is `586d4dd65bed4d5e8066a98b6210b2b3a85f7144`.
`origin` must point to the fork containing `paper-walking-baseline`. A local
branch created from newer `main` may have the same name but different code;
if the fast-forward fails, preserve that branch and inspect its history
before proceeding.

For a clean checkout where the current `paper-walking-baseline` branch is
still at the newer `main` commit `f81679b`, preserve it under a backup name
and check out the published branch with tracking:

```bash
git fetch origin paper-walking-baseline
git branch -m paper-walking-baseline-before-import-f81679b
git switch --track -c paper-walking-baseline origin/paper-walking-baseline
git rev-parse --short HEAD
```

The published prepared code should show `586d4dd`. Then install below.

```bash
(
  set -eu
  git fetch origin paper-walking-baseline
  git switch paper-walking-baseline
  git merge --ff-only origin/paper-walking-baseline
  git branch --set-upstream-to=origin/paper-walking-baseline paper-walking-baseline
  git merge-base --is-ancestor 586d4dd65bed4d5e8066a98b6210b2b3a85f7144 HEAD
  git submodule update --init --recursive
  if [ ! -x .venv-paper/bin/python ]; then
    uv venv --python 3.10 .venv-paper
  fi
  .venv-paper/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Recreate .venv-paper with Python 3.10"'
  uv pip sync --python .venv-paper/bin/python requirements-paper.lock
  uv pip check --python .venv-paper/bin/python
)
```

The separate `.venv-paper` environment uses Python 3.10, JAX/JAXlib 0.4.28,
MuJoCo/MJX 3.2.4, the pinned Brax fork, Flax 0.8.4, Optax 0.2.2 and
Orbax 0.5.23. Linux installs the matching CUDA 12 plugin. The existing `.venv`
is preserved. Training requires no WR2 checkout or launcher script.

If an earlier `uv run --no-sync` created an empty Python 3.11 `.venv-paper`,
recreate that environment and install the complete lock:

```bash
uv venv --clear --python 3.10 .venv-paper
uv pip sync --python .venv-paper/bin/python requirements-paper.lock
```

`--clear` replaces only `.venv-paper`. `uv run --no-sync` runs an already
installed environment; it does not install the lock, and can create an empty
environment when none exists.

## RTX 5070 / Blackwell compatibility environment

The RTX 5070 has compute capability 12.0 (Blackwell). JAX 0.4.28 predates
this architecture: device enumeration can succeed while the first kernel
fails with `ptxas fatal: Program with .target 'sm_90a' cannot be compiled to
future architecture`. The reported driver 580.178.04 supports the CUDA 12.9
runtime used below.

Use the separate `requirements-paper-blackwell.lock` on Linux x86_64 with
Python 3.10. It pins JAX/JAXlib/CUDA plugins 0.6.2, cuDNN 9.8.0.87, Flax
0.10.6, Optax 0.2.5, Chex 0.1.90 and Orbax 0.11.13. This is a hardware
compatibility variant of the published-era baseline. The Brax gitlink,
MuJoCo/MJX 3.2.4, robot/reference files, native PPO and Table VI/VII settings
are retained. The original `requirements-paper.lock` and environments remain
available for comparison.

First bring the compatibility changes from this repository to the GPU
checkout. The original `586d4dd` commit does not contain this lock, the
preflight module or the JAX matmul-precision compatibility fix.

Create and install the separate environment once:

```bash
git submodule update --init --recursive
uv venv --python 3.10 .venv-paper-blackwell
uv pip sync --python .venv-paper-blackwell/bin/python requirements-paper-blackwell.lock
```

Select it once per terminal, then check the source, all active locked
packages, other dependency requirements, GPU detection and JIT compilation:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-paper-blackwell UV_PYTHON=3.10 WANDB_MODE=offline
uv run --no-sync -m toddlerbot.locomotion.paper_environment --blackwell
```

The pinned Brax fork's package metadata still requires JAX/JAXlib 0.4.28 and
Orbax 0.5.23. The compatibility lock explicitly overrides those three pins.
`uv pip check` reports those three expected metadata conflicts in this
environment; the preflight verifies the compatibility lock and permits only
those declared Brax overrides. The Brax source remains clean at its pin.

Run the short native PPO check before the long run:

```bash
uv run --no-sync -m toddlerbot.locomotion.train_mjx --gin-files "paper paper_smoke" --skip-video
```

`paper_smoke.gin` requests 40 transitions, one checkpoint and the XY suite
with two short episodes per condition. It checks compilation, PPO updates,
finite checkpoints and evaluation execution. It does not measure walking
quality or establish memory fit for the full 1024 environments. The trainer
records the actual installed packages and copies the matching dependency
lock into each run's results.

After both checks succeed, start the fresh 300M run:

```bash
uv run --no-sync -m toddlerbot.locomotion.train_mjx --gin-files paper --skip-video
```

## Launch the original baseline

For the original baseline environment, select it once per terminal:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-paper UV_PYTHON=3.10 WANDB_MODE=offline
```

Check dependencies, the pinned source, GPU detection and actual JIT
compilation before the long run. This uses the trainer's XLA settings.
Continue only if this succeeds:

```bash
uv run --no-sync -m toddlerbot.locomotion.paper_environment
```

Then launch the native trainer directly:

```bash
uv run --no-sync -m toddlerbot.locomotion.train_mjx --gin-files paper --skip-video
```

The robot and environment default to `toddlerbot` and `walk`. `--no-sync`
retains the packages installed from the selected dependency lock.

Start fresh. Native `--restore` is a policy/value/normalizer warm start; it does
not restore Adam, RNG, counters or environment state. `--skip-video` skips the
final video while retaining quantitative checkpoint evaluations.

## Results and readiness

Native rounding gives 300,134,400 transitions and 15 trained checkpoints,
about 20M transitions apart. Each checkpoint has native reward evaluation and
15 signed XY, diagonal, standing and start/stop conditions, 128 episodes each.
Policy inference is deterministic during this quantitative suite; released
noise, robot variation and pushes remain enabled. Nonfinite parameters or
metrics stop training after retaining the checkpoint.

Keep the complete `results/toddlerbot_walk_ppo_<timestamp>/` directory,
including `metrics.jsonl`, `xy_evaluation.jsonl`, `provenance.json`, effective
configs, dependency lock and all checkpoint directories. Compare intermediate
checkpoints using signed velocity, instantaneous vector tracking error,
standing speed, leg motor RMSE, survival and recovery.

The reference fixes, native recipe and checkpoint/evaluation execution have
passed CPU checks. The Blackwell compatibility stack also passed all nine
focused tests and the 40-transition native PPO smoke run, including a finite
checkpoint and all 15 XY conditions, using its CPU dependency projection on
the Mac. GPU compatibility, memory fit and learned walking still require
execution on the GPU host. The existing focused checks run with:

```bash
WANDB_MODE=disabled .venv-paper/bin/python -m unittest discover \
  -s tests -p test_paper_walking.py -v
```
