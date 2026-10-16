"""Paper signed-XY measurement protocol adapted to the current main interfaces."""

import copy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.wrappers.training import DomainRandomizationVmapWrapper

from toddlerbot.utils.math_utils import get_local_vec


CONDITIONS = (
    ("forward_010", (0.10, 0.0), False),
    ("backward_010", (-0.10, 0.0), False),
    ("backward_005", (-0.05, 0.0), False),
    ("forward_005", (0.05, 0.0), False),
    ("right_005", (0.0, -0.05), False),
    ("left_005", (0.0, 0.05), False),
    ("backward_right", (-0.07071067811865475, -0.035355339059327376), False),
    ("backward_left", (-0.07071067811865475, 0.035355339059327376), False),
    ("forward_right", (0.07071067811865475, -0.035355339059327376), False),
    ("forward_left", (0.07071067811865475, 0.035355339059327376), False),
    ("standing", (0.0, 0.0), False),
    ("forward_transitions", (0.10, 0.0), True),
    ("backward_transitions", (-0.10, 0.0), True),
    ("right_transitions", (0.0, -0.05), True),
    ("left_transitions", (0.0, 0.05), True),
)
GROUPS = ("all", "standing", "walking", "start", "stop")
MEASUREMENTS = (
    "velocity_x_m_s",
    "velocity_y_m_s",
    "xy_error_m_s",
    "xy_speed_m_s",
    "leg_motor_rmse_rad",
)


def command_and_groups(step, target, transitions, dt=0.02):
    """3 s stand, 6 s walk, 3 s stand, 5 s walk, 3 s stand; 0.5 s recovery."""
    start1, stop1, start2, stop2 = (int(round(t / dt)) for t in (3, 9, 12, 17))
    window = int(round(0.5 / dt))
    moving = jnp.linalg.norm(target) > 1e-6
    scheduled_walk = ((step >= start1) & (step < stop1)) | (
        (step >= start2) & (step < stop2)
    )
    walking = moving & jnp.where(transitions, scheduled_walk, True)
    start = (
        transitions
        & walking
        & (
            ((step >= start1) & (step < start1 + window))
            | ((step >= start2) & (step < start2 + window))
        )
    )
    stop = (
        transitions
        & moving
        & ~walking
        & (
            ((step >= stop1) & (step < stop1 + window))
            | ((step >= stop2) & (step < stop2 + window))
        )
    )
    command = jnp.zeros(8).at[5:7].set(jnp.where(walking, target, 0.0))
    return command, jnp.asarray([True, ~walking, walking, start, stop])


def set_command(state, command):
    """Update only the newest frame's command, preserving native noisy history."""
    info = {**state.info, "command": command, "command_obs": command[..., 5:8]}
    return state.replace(
        info=info,
        obs={key: value.at[..., 2:5].set(command[..., 5:8])
             for key, value in state.obs.items()},
    )


class CommandVmapWrapper(DomainRandomizationVmapWrapper):
    nominal_reset = False

    def reset(self, rng, command):
        def reset(sys, key, cmd):
            with self.v_env_fn(sys) as env:
                state = env.reset(key, cmd)
                if self.nominal_reset:
                    # Main normally initializes root velocity from its lookup.
                    # Nominal comparisons start from rest on both implementations.
                    p = env.pipeline_init(state.pipeline_state.q,
                                          jnp.zeros_like(state.pipeline_state.qd))
                    obs, info = env._get_obs(
                        p, state.info, jnp.zeros_like(state.obs["state"]),
                        jnp.zeros_like(state.obs["privileged_state"]),
                    )
                    state = state.replace(pipeline_state=p, obs=obs, info=info)
                return state

        return jax.vmap(reset, in_axes=[self._in_axes, 0, 0])(
            self._sys_v, rng, command
        )


def summarize(sums, counts):
    """Normalize within each episode before comparing episodes of unequal length."""
    groups = {}
    per_episode = {}
    for group_index, name in enumerate(GROUPS):
        count = counts[:, group_index]
        eligible = count > 0
        mean = np.divide(
            sums[:, group_index],
            count[:, None],
            out=np.zeros_like(sums[:, group_index]),
            where=count[:, None] > 0,
        )
        mean[:, -1] = np.sqrt(np.maximum(mean[:, -1], 0.0))
        groups[name] = {
            "sample_count": int(count.sum()),
            "episode_count": int(eligible.sum()),
        }
        per_episode[name] = {"sample_count": count.tolist()}
        for index, measurement in enumerate(MEASUREMENTS):
            values = mean[eligible, index]
            groups[name][measurement] = {
                "mean": float(values.mean()) if values.size else None,
                "std": float(values.std()) if values.size else None,
            }
            per_episode[name][measurement] = [
                float(value) if valid else None
                for value, valid in zip(mean[:, index], eligible)
            ]
    return groups, per_episode


class XYEvaluator:
    def __init__(
        self,
        env,
        make_policy,
        num_envs,
        episode_length,
        randomization_fn=None,
        seed=12102,
        profile="native",
    ):
        # Brax's wrapper assigns sys while tracing. Do not share that mutable
        # environment object with PPO's training or random-command evaluator.
        env = copy.copy(env)
        env.cfg = copy.deepcopy(env.cfg)
        if profile not in ("native", "nominal"):
            raise ValueError(f"Unknown evaluation profile: {profile}")
        if env.cfg.terrain.scene is not None or env.cfg.terrain.manual_map != [["flat"]]:
            raise ValueError("The comparison benchmark requires the single flat terrain.")
        if not env.use_phase_signal or env.obs_size != 2 + 3 + 2 * env.nu + env.num_action + 3 + 4:
            raise ValueError("Unexpected walking observation layout for leg-error extraction.")
        self.profile = profile
        self.native_height_range = tuple(env.healthy_z_range)
        # Shared physical stop criterion, independent of each recipe's native bound.
        env.healthy_z_range = (0.20, 0.40)
        if profile == "nominal":
            env.add_domain_rand = False
            env.add_push = False
            env.rand_init_state_indices = [0]
            randomization_fn = None
        env.resample_steps = episode_length + 1
        # Evaluate survival without the main environment's internal motion teleport.
        env.episode_length = episode_length + int(np.max(env.rand_init_state_indices)) + 1
        self.num_envs = num_envs
        self.episode_length = episode_length
        self.dt = float(env.dt)
        self.make_policy = make_policy
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)
        model_key, reset_key = jax.random.split(self.key)
        self.reset_keys = jax.random.split(reset_key, num_envs)
        if randomization_fn is None:

            def randomization_fn(sys, rng):
                return sys, jax.tree.map(lambda _: None, sys)

        self.env = CommandVmapWrapper(
            env,
            lambda sys: randomization_fn(
                sys, rng=jax.random.split(model_key, num_envs)
            ),
        )
        self.env.nominal_reset = profile == "nominal"
        self.leg_actuator_indices = env.leg_actuator_indices
        # Main computes this unscaled encoder/reference error before it resets
        # state_ref on a terminal step. Extract it from the newest critic frame.
        self.motor_error_offset = env.obs_size
        self._run = jax.jit(self._rollout)

    def _rollout(self, params, target, transitions):
        policy = self.make_policy(params, deterministic=True)
        command, _ = command_and_groups(0, target, transitions, self.dt)
        command = jnp.broadcast_to(command, (self.num_envs, 8))
        state = self.env.reset(self.reset_keys, command)
        alive = jnp.ones(self.num_envs, dtype=bool)
        lengths = jnp.zeros(self.num_envs, dtype=jnp.int32)
        falls = jnp.zeros(self.num_envs, dtype=bool)
        nonfinite = jnp.zeros(self.num_envs, dtype=bool)
        native_terminals = jnp.zeros(self.num_envs, dtype=bool)
        counts = jnp.zeros((self.num_envs, len(GROUPS)), dtype=jnp.int32)
        sums = jnp.zeros((self.num_envs, len(GROUPS), len(MEASUREMENTS)))

        def step(carry, index):
            state, alive, lengths, falls, nonfinite, native_terminals, counts, sums = carry
            command, groups = command_and_groups(index, target, transitions, self.dt)
            command = jnp.broadcast_to(command, (self.num_envs, 8))
            state = set_command(state, command)
            action, _ = policy(state.obs, self.key)
            # Native step mutates its info dictionary while tracing. Rebuild it
            # so the old carry remains available when freezing inactive episodes.
            next_state = self.env.step(jax.tree.map(lambda x: x, state), action)
            p = next_state.pipeline_state
            velocity = jax.vmap(get_local_vec)(p.xd.vel[:, 0], p.x.rot[:, 0])
            error_start = self.motor_error_offset
            motor_error = next_state.obs["privileged_state"][
                :, error_start + self.leg_actuator_indices
            ]
            values = jnp.stack(
                [
                    velocity[:, 0],
                    velocity[:, 1],
                    jnp.linalg.norm(velocity[:, :2] - command[:, 5:7], axis=1),
                    jnp.linalg.norm(velocity[:, :2], axis=1),
                    jnp.mean(jnp.square(motor_error), axis=1),
                ],
                axis=1,
            )
            finite = (
                jnp.isfinite(p.q).all(axis=1)
                & jnp.isfinite(p.qd).all(axis=1)
                & jnp.isfinite(next_state.obs["state"]).all(axis=1)
                & jnp.isfinite(next_state.obs["privileged_state"]).all(axis=1)
                & jnp.isfinite(next_state.reward)
                & jnp.isfinite(action).all(axis=1)
                & jnp.isfinite(values).all(axis=1)
            )
            active = alive & finite
            mask = active[:, None] & groups[None, :]
            counts += mask.astype(jnp.int32)
            sums += jnp.where(mask[:, :, None], values[:, None, :], 0.0)
            lengths += alive.astype(jnp.int32)
            falls |= alive & finite & (next_state.done > 0)
            nonfinite |= alive & ~finite
            height = p.x.pos[:, 0, 2]  # This benchmark uses a flat ground at z=0.
            native_terminals |= active & (
                (height < self.native_height_range[0])
                | (height > self.native_height_range[1])
            )
            # Include the terminal finite sample, then stop that episode.
            next_state = jax.tree.map(
                lambda new, old: jnp.where(
                    active.reshape((self.num_envs,) + (1,) * (new.ndim - 1)), new, old
                ),
                next_state,
                state,
            )
            alive &= finite & (next_state.done == 0)
            return (next_state, alive, lengths, falls, nonfinite, native_terminals, counts, sums), None

        result, _ = jax.lax.scan(
            step,
            (state, alive, lengths, falls, nonfinite, native_terminals, counts, sums),
            jnp.arange(self.episode_length),
        )
        return result[2:]

    def evaluate(self, params: Any):
        conditions = []
        for name, target, transitions in CONDITIONS:
            lengths, falls, nonfinite, native_terminals, counts, sums = jax.device_get(
                self._run(params, jnp.asarray(target), jnp.asarray(transitions))
            )
            groups, per_episode = summarize(sums, counts)
            conditions.append(
                {
                    "name": name,
                    "command_xy_m_s": list(target),
                    "transitions": transitions,
                    "episodes": self.num_envs,
                    "mean_episode_length": float(lengths.mean()),
                    "fall_count": int(falls.sum()),
                    "nonfinite_count": int(nonfinite.sum()),
                    "native_terminal_count_before_benchmark_stop": int(native_terminals.sum()),
                    "groups": groups,
                    "per_episode": {
                        "length": lengths.tolist(),
                        "fell": falls.tolist(),
                        "nonfinite": nonfinite.tolist(),
                        "native_terminal_before_benchmark_stop": native_terminals.tolist(),
                        **per_episode,
                    },
                }
            )
        return {
            "protocol": "tb_signed_xy_v1",
            "profile": self.profile,
            "seed": self.seed,
            "benchmark_height_range_m": [0.20, 0.40],
            "native_height_range_m": list(self.native_height_range),
            "automatic_resets": False,
            "internal_motion_teleport": False,
            "initial_generalized_velocity": "zero" if self.profile == "nominal" else "native_reference",
            "episode_length": self.episode_length,
            "control_period_s": self.dt,
            "deterministic_policy": True,
            "conditions": conditions,
        }
