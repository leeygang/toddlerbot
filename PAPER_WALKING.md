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

Once the prepared branch is available in your existing TB checkout, run from
its repository root. On the GPU host this is `~/projects/ToddlerBot`.

```bash
(
  set -eu
  git switch paper-walking-baseline
  git submodule update --init --recursive
  uv venv --python 3.10 .venv-paper
  uv pip sync --python .venv-paper/bin/python requirements-paper.lock
)
```

The separate `.venv-paper` environment uses Python 3.10, JAX/JAXlib 0.4.28,
MuJoCo/MJX 3.2.4, the pinned Brax fork, Flax 0.8.4, Optax 0.2.2 and
Orbax 0.5.23. Linux installs the matching CUDA 12 plugin. The existing `.venv`
is preserved. Training requires no WR2 checkout or launcher script.

## Launch the native trainer

From the TB repository root on the GPU host:

```bash
(
  set -eu
  .venv-paper/bin/python -c 'import jax; print(jax.devices()); assert jax.default_backend() == "gpu"'
  WANDB_MODE=offline .venv-paper/bin/python -m toddlerbot.locomotion.train_mjx \
    --robot toddlerbot --env walk --gin-files paper --skip-video
)
```

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
passed CPU checks. GPU compatibility, memory fit and learned walking still
require execution on the GPU host. The existing focused checks run with:

```bash
WANDB_MODE=disabled .venv-paper/bin/python -m unittest discover \
  -s tests -p test_paper_walking.py -v
```
