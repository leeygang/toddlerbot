"""Measurement gates for signed tracking, history, and terminal samples."""

import os

os.environ["USE_JAX"] = "true"

import jax
import jax.numpy as jnp
import numpy as np
from brax import base
from brax.envs.base import State

from toddlerbot.locomotion.evaluate_xy import (
    XYEvaluator, command_and_groups, set_command, summarize,
)


def test_transition_schedule_and_history():
    commands, groups = jax.vmap(lambda step: command_and_groups(
        step, jnp.array([-0.1, 0.0]), True
    ))(jnp.arange(1000))
    np.testing.assert_array_equal(np.asarray(groups).sum(axis=0), [1000, 450, 550, 50, 50])
    np.testing.assert_allclose(commands[150:450, 5], -0.1)
    np.testing.assert_allclose(commands[450:600, 5], 0.0)
    np.testing.assert_allclose(commands[600:850, 5], -0.1)
    history = jnp.arange(1260.0).reshape(1, -1)
    state = State(pipeline_state=None, obs={"state": history, "privileged_state": history},
                  reward=jnp.zeros(1), done=jnp.zeros(1), metrics={}, info={})
    updated = set_command(state, jnp.zeros((1, 8)).at[:, 5].set(-0.1))
    for obs in updated.obs.values():
        np.testing.assert_array_equal(obs[:, 5:], history[:, 5:])
        np.testing.assert_array_equal(obs[:, :2], history[:, :2])
        np.testing.assert_allclose(obs[:, 2:5], [[-0.1, 0.0, 0.0]])


def test_episode_first_aggregation_and_leg_rmse():
    counts = np.zeros((2, 5), dtype=int)
    counts[:, 0] = [1, 9]
    sums = np.zeros((2, 5, 5))
    sums[:, 0, 0] = [-0.1, -0.9]
    sums[:, 0, 2] = [1, 27]  # Per-episode errors 1 and 3, mean 2, not 2.8.
    sums[:, 0, 4] = [4, 144]  # Per-episode leg MSE 4 and 16 -> RMSE 2 and 4.
    groups, per_episode = summarize(sums, counts)
    assert groups["all"]["xy_error_m_s"] == {"mean": 2.0, "std": 1.0}
    assert groups["all"]["leg_motor_rmse_rad"] == {"mean": 3.0, "std": 1.0}
    assert groups["standing"]["xy_error_m_s"]["mean"] is None
    assert per_episode["all"]["sample_count"] == [1, 9]


class TerminalEnv:
    """One finite fall at step 3; one nonfinite state at step 2."""

    def reset(self, keys, command):
        pipeline = base.State(
            q=jnp.zeros((2, 2)), qd=jnp.zeros((2, 2)),
            x=base.Transform(pos=jnp.zeros((2, 1, 3)),
                             rot=jnp.tile(jnp.array([1.0, 0, 0, 0]), (2, 1, 1))),
            xd=base.Motion(ang=jnp.zeros((2, 1, 3)), vel=jnp.zeros((2, 1, 3))),
            contact=None,
        )
        return State(pipeline_state=pipeline,
                     obs={"state": jnp.zeros((2, 84)), "privileged_state": jnp.zeros((2, 151))},
                     reward=jnp.zeros(2), done=jnp.zeros(2), metrics={},
                     info={"step": jnp.zeros(2, dtype=int),
                           "command": command, "command_obs": command[:, 5:8]})

    def step(self, state, action):
        step = state.info["step"] + 1
        p = state.pipeline_state
        q = p.q.at[1, 0].set(jnp.where(step[1] == 2, jnp.nan, 0.0))
        pos = p.x.pos.at[:, 0, 2].set(jnp.where(step == 3, 0.15, 0.30))
        velocity = p.xd.vel.at[:, 0, 0].set(-0.08)
        # The critic's terminal encoder error is retained even if native info
        # resets the reference on failure. Its RMSE must remain 1 rad here.
        privileged = state.obs["privileged_state"].at[:, 84:96].set(1.0)
        return state.replace(
            pipeline_state=p.replace(q=q, x=p.x.replace(pos=pos), xd=p.xd.replace(vel=velocity)),
            obs={**state.obs, "privileged_state": privileged},
            done=(step == 3).astype(float), info={**state.info, "step": step},
        )


def test_freeze_after_first_terminal_or_nonfinite_sample():
    evaluator = XYEvaluator.__new__(XYEvaluator)
    evaluator.env = TerminalEnv()
    evaluator.make_policy = lambda params, deterministic: lambda obs, key: (jnp.zeros((2, 12)), {})
    evaluator.num_envs = 2
    evaluator.episode_length = 10
    evaluator.dt = 0.02
    evaluator.key = jax.random.PRNGKey(1)
    evaluator.reset_keys = jax.random.split(evaluator.key, 2)
    evaluator.motor_error_offset = 84
    evaluator.leg_actuator_indices = jnp.arange(12)
    evaluator.native_height_range = (0.2, 1.0)
    lengths, falls, nonfinite, native, counts, sums = jax.device_get(
        evaluator._rollout(None, jnp.array([-0.1, 0.0]), False)
    )
    np.testing.assert_array_equal(lengths, [3, 2])
    np.testing.assert_array_equal(falls, [True, False])
    np.testing.assert_array_equal(nonfinite, [False, True])
    np.testing.assert_array_equal(native, [True, False])
    np.testing.assert_array_equal(counts[:, 0], [3, 1])
    groups, _ = summarize(sums, counts)
    np.testing.assert_allclose(groups["all"]["velocity_x_m_s"]["mean"], -0.08, atol=1e-7)
    np.testing.assert_allclose(groups["all"]["xy_error_m_s"]["mean"], 0.02, atol=1e-7)
    np.testing.assert_allclose(groups["all"]["leg_motor_rmse_rad"]["mean"], 1.0)
