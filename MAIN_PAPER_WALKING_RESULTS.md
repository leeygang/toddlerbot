# Main versus paper walking results — 2026-10-17

## Analysis status

**The paper baseline is verified. The completed main run has not yet been found
in this local checkout, so its training progress and walking quality remain
unmeasured here.** The user reported that the new GPU training completed; that
completion has not yet been checked against transferred artifacts.

The prepared comparison training code is `main-walking-compare` commit
`990b8b0`, based on `main` at `f81679b`. The experiment is documented in
[MAIN_WALKING_COMPARE.md](MAIN_WALKING_COMPARE.md).

Local data inspected:

| Dataset | Location | Use in this report |
| --- | --- | --- |
| Completed paper GPU run | `results/toddlerbot_walk_ppo_20261015_194347/` | Verified 300,134,400-transition baseline |
| Main CPU smoke run | `.worktrees/main-walking-compare/results/toddlerbot_2xc_walk_brax_20261016_113551/` | 40 transitions; execution check only |
| Historical main archive | `results/toddlerbot_2xc_walk_brax_20260515_114507/` | Previous 1B-transition experiment; final console metrics are NaN; excluded from the fresh comparison |
| New main GPU run | Not located under `results/` or the linked worktree's `results/` | Comparison pending |
| Local W&B records | `wandb/offline-run-20261015_194347-h1vcwang/` | Paper run; no new main W&B run located |

The main smoke run and historical NaN archive cannot establish the outcome of
the newly completed main GPU run. No main-versus-paper performance difference
or winner is inferred from them.

## Paper reference candidates

**Retain paper 180M for survival and 200M for tracking.** Native return selects
240M, while the final 300M policy has substantially more falls. These rankings
come from the existing native-variation evaluation seed and need confirmation
on fresh evaluation draws before selecting a deployment policy.

| Paper checkpoint | Native evaluation return | Fixed walking XY error (m/s) | Fixed falls / 1,280 | Standing falls / 128 | Transition falls / 512 | All falls / 1,920 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 180,080,640 — survival candidate | 372.745 | 0.038379 | 2 | 0 | 1 | **3 (0.156%)** |
| 200,089,600 — tracking candidate | 371.733 | **0.037025** | 5 | 0 | 3 | **8 (0.417%)** |
| 240,107,520 — native return-best | **380.249** | 0.038228 | 39 | 2 | 16 | 57 (2.969%) |
| 300,134,400 — final | 369.533 | 0.037999 | 84 | 6 | 44 | **134 (6.979%)** |

At the final paper checkpoint, reset-standing speed is **0.030955 m/s** and
fixed-command leg motor RMSE is **0.223045 rad**. The latter is error against
the paper implementation's own ZMP motor reference.

## Paper training progress

![Paper baseline progress; main artifacts pending](docs/main_paper_walking_20261017/paper_progress.png)

The green interval marks the 180–200M candidates; the dashed line marks the
240M native return-best. The [PDF](docs/main_paper_walking_20261017/paper_progress.pdf)
and [full progress CSV](docs/main_paper_walking_20261017/paper_progress.csv) retain
the plotted measurements, elapsed time and optimizer losses.

Fixed-command error and leg RMSE are equal means across ten moving conditions.
Standing is separate; the four transition conditions mix walking and standing.
Native return comes from the paper's separate stochastic random-command evaluator.

| Collected steps (M) | Native return | Fixed XY error (m/s) | Fixed falls | Standing speed (m/s) | Standing falls | Transition falls | Fixed leg RMSE (rad) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20.009 | 296.612 | 0.062016 | 6 | 0.044195 | 0 | 3 | 0.235851 |
| 40.018 | 342.340 | 0.047711 | 13 | 0.036662 | 2 | 5 | 0.236410 |
| 60.027 | 345.986 | 0.044307 | 7 | 0.034053 | 0 | 4 | 0.229813 |
| 80.036 | 351.904 | 0.039653 | 11 | 0.030820 | 0 | 6 | 0.235668 |
| 100.045 | 344.376 | 0.043000 | 38 | 0.033406 | 4 | 18 | 0.233893 |
| 120.054 | 358.583 | 0.039524 | 30 | 0.032792 | 0 | 9 | 0.232381 |
| 140.063 | 362.073 | 0.038058 | 15 | 0.031532 | 0 | 5 | 0.232898 |
| 160.072 | 360.000 | 0.038245 | 25 | 0.031062 | 1 | 8 | 0.231607 |
| 180.081 | 372.745 | 0.038379 | 2 | 0.030846 | 0 | 1 | 0.232178 |
| 200.090 | 371.733 | 0.037025 | 5 | 0.031964 | 0 | 3 | 0.231847 |
| 220.099 | 371.688 | 0.038976 | 31 | 0.031256 | 2 | 9 | 0.228396 |
| 240.108 | 380.249 | 0.038228 | 39 | 0.030143 | 2 | 16 | 0.225606 |
| 260.116 | 374.612 | 0.037131 | 66 | 0.032279 | 9 | 21 | 0.224469 |
| 280.125 | 378.557 | 0.039430 | 82 | 0.036108 | 15 | 37 | 0.223500 |
| 300.134 | 369.533 | 0.037999 | 84 | 0.030955 | 6 | 44 | 0.223045 |

Tracking improves mainly by 80M, then stays near 0.037–0.043 m/s. From the
180M checkpoint to the final policy, falls rise from 3 to 134 across the suite,
while leg imitation error improves. Native return therefore gives a different
checkpoint ranking from physical tracking and survival measurements.

The final recorded progress callback is at **11,243.460 seconds**, before the
last quantitative suite and cleanup. The native cumulative training timer is
**8,892.766 seconds**; the median training throughput after the first stage is
approximately **34,400 transitions/s**. These timers include different work and
should retain their native definitions when comparing main's runtime.

## Verification performed for this report

The [audit](docs/main_paper_walking_20261017/paper_audit.json) records input SHA-256
hashes and the checks run directly against the copied paper artifacts:

- Training metrics and evaluation step lists match: 15 checkpoints through
  300,134,400 actual transitions, fresh training seed 0.
- All native scalar metrics are finite. The 225 condition summaries cover
  28,800 episode-condition evaluations, with zero nonfinite episodes.
- Fall counts, mean episode lengths, group sample counts, eligible episode
  counts, episode-first means and population standard deviations reconstruct
  from the per-episode arrays to within 1e-6.
- The archived source revision is `1982ef3bf540fd5280bfe89937b57866f94de8de`.
  Provenance retains its recorded dirty-source flag.

The earlier paper report and detailed source/checkpoint audit are retained on
`paper-walking-baseline` as `PAPER_WALKING_RESULTS.md` and
`docs/paper_walking_20261016/audit.json`. This report's newly computed baseline
uses the raw JSON files present in this checkout.

## How the main comparison will be interpreted

These are the **prepared** main settings, pending confirmation from the new
run's saved effective config and provenance:

| Setting | Paper GPU run | Prepared main comparison |
| --- | --- | --- |
| Budget and seed | 300M requested / seed 0 | Same |
| Collection and updates | 1,024 envs; rollout 20; batch 256; 4 minibatches; 4 passes | Same |
| Robot | Published-era `toddlerbot` | `toddlerbot_2xc` |
| Runtime | Python 3.10; pinned Brax fork 0.10.5; JAX 0.6.2 | Python 3.11; Brax 0.14.2; JAX 0.6.2 |
| Physics | MuJoCo/MJX 3.2.4; 4 ms × 5 | MuJoCo/MJX 3.3.4; 5 ms × 4 |
| Policy | Swish; state-dependent Normal followed by tanh | ELU; Normal with independent learned log std |
| Learning rate | Approximately 1e-4 over this budget | Constant 3e-5 |
| XY reward | Weight 5; tracking coefficient 100 | Weight 2; tracking coefficient 1,000 |
| Leg motor reference reward | Weight 5 | Weight 0; separate home-pose penalty 0.5 |
| Native survival training | Fall penalty 10; height [0.20, 0.40] m | Alive reward 1; height [0.20, 1.0] m |
| Logged native return | Independent random-command evaluation episodes | Collected training episodes; native Brax evaluations disabled |
| Quantitative profiles | Native variation including pushes | Nominal and main-native variation; pushes disabled |

Main's quantitative evaluator uses the shared [0.20, 0.40] m benchmark height
range and the same 15 signed conditions, 128 episodes per condition, 1,000-step
horizon, 20 ms control period and evaluation seed 12102. Automatic resets and
internal motion teleports are disabled during measurement. Verify these values
in the transferred suites before combining progress tables.

Compare physical tracking, standing, falls, episode length and leg imitation
at matched actual transition counts, especially approximately 180M, 200M, 240M
and 300M. Keep main's nominal and native profiles separate. The existing paper
data include its native pushes and other variation; a controlled nominal
comparison requires a nominal re-evaluation of the paper checkpoints.

XY error is the episode-first mean instantaneous vector error in the full torso
quaternion frame. Standing speed is mean instantaneous XY speed. Leg RMSE is
episode-first error against each implementation's own reference. Include falls,
nonfinite counts, eligible sample counts and episode lengths beside error, since
failed episodes have shorter measurement windows. Episode standard deviations
describe one training seed's evaluation draws, not training-seed uncertainty.

Raw native returns have different reward definitions and sampling sources; use
their curves to understand each run's progress. Different robots, references,
observations, action distributions, randomization and runtimes prevent attributing
any eventual performance difference to one reward or trainer change. Compare
geometry-normalized velocity and time once both nominal COM heights are available.

The paper's Table II value of 0.016 m/s comes from its square-trajectory experiment.
That experiment's filtering and aggregation differ from the instantaneous vector
error measured here, so it is not an identical acceptance threshold for this suite.

## Data needed to finish

Provide the actual local path of the newly completed main run, or finish copying
its full `results/toddlerbot_2xc_walk_brax_<timestamp>/` folder into this repository.
The essential files are `metrics.jsonl`, `xy_evaluation.jsonl`, `checkpoints.jsonl`,
`train_config.json`, `env_config.json`, `provenance.json`, `output.log` and `ckpt/`.
Its complete W&B offline directory can confirm exit status and GPU timing/load.

The remaining report work is to audit that run, add both main profile progress
curves, compare matched checkpoints and final signed-condition measurements,
identify main tracking/survival candidates, and state the observed differences
with the protocol limitations above.

Reference: [ToddlerBot paper v2](https://arxiv.org/html/2502.00893v2).
