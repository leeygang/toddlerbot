"""Inspect archived policy finiteness and raw-action history feedback on CPU.

The feedback probe holds motor/IMU inputs at nominal home values. It is a
synthetic network experiment, not a physics rollout or walking evaluation.
"""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks
from flax import linen


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "results/toddlerbot_2xc_walk_brax_20260515_114507"
STEPS = (0, 202342400, 222576640, 232693760, 242810880, 1001594880)


def numeric_summary(tree):
    arrays = [np.asarray(x) for x in jax.tree.leaves(tree)]
    arrays = [x for x in arrays if x.dtype.kind in "biufc"]
    finite_values = [np.abs(x[np.isfinite(x)]) for x in arrays]
    return {
        "numeric_leaves": len(arrays),
        "numeric_values": sum(x.size for x in arrays),
        "nonfinite_values": sum(int((~np.isfinite(x)).sum()) for x in arrays),
        "finite": all(np.isfinite(x).all() for x in arrays),
        "max_abs_finite_value": max(
            (float(x.max()) for x in finite_values if x.size), default=None
        ),
    }


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def feedback_probe(policy, params, n_steps=500):
    scenarios = [
        ("forward", [0.1, 0.0, 0.0], 0.0),
        ("backward", [-0.1, 0.0, 0.0], 0.0),
        ("left", [0.0, 0.05, 0.0], 0.0),
        ("right", [0.0, -0.05, 0.0], 0.0),
        ("standing", [0.0, 0.0, 0.0], 0.0),
        ("forward_history_positive", [0.1, 0.0, 0.0], 1.0),
        ("forward_history_negative", [0.1, 0.0, 0.0], -1.0),
    ]
    count = len(scenarios)
    # Both variants emit unbounded means. Only the history input differs.
    bounded = jnp.array([False] * count + [True] * count)
    frame = jnp.zeros((2 * count, 84), dtype=jnp.float32)
    frame = frame.at[:, 1].set(1.0)  # phase cosine
    frame = frame.at[:, 2:5].set(jnp.array([s[1] for s in scenarios] * 2))
    frame = frame.at[:, 80].set(1.0)  # torso quaternion w
    initial_actions = jnp.repeat(
        jnp.array([s[2] for s in scenarios] * 2)[:, None], 12, axis=1
    )
    frame = frame.at[:, 65:77].set(initial_actions)
    history = jnp.repeat(frame[:, None, :], 15, axis=1)
    delay = jnp.repeat(initial_actions[:, None, :], 2, axis=1)

    @jax.jit
    def run(initial_history, initial_delay):
        def step(carry, i):
            history, delay = carry
            means, std = policy.apply(params[0], params[1], {
                "state": history.reshape(2 * count, 1260)
            })
            phase = 2 * jnp.pi * ((i + 1) * 0.02) / 0.72
            next_frame = frame.at[:, 0].set(jnp.sin(phase))
            next_frame = next_frame.at[:, 1].set(jnp.cos(phase))
            previous = jnp.where(
                bounded[:, None], jnp.clip(delay[:, 0], -1.0, 1.0), delay[:, 0]
            )
            next_frame = next_frame.at[:, 65:77].set(previous)
            # Matches the history order and extra sensor lag in main's step:
            # _get_obs reads last_act before that step updates last_act.
            history = jnp.concatenate([next_frame[:, None], history[:, :-1]], axis=1)
            delay = jnp.concatenate([delay[:, 1:], means[:, None]], axis=1)
            return (history, delay), (
                jnp.max(jnp.abs(means), axis=1), jnp.max(std, axis=1)
            )
        return jax.lax.scan(step, (initial_history, initial_delay), jnp.arange(n_steps))[1]

    maxima, std = jax.device_get(run(history, delay))
    summary = []
    for i, (name, command, initial) in enumerate(scenarios * 2):
        values = maxima[:, i]
        good = np.isfinite(values)
        summary.append({
            "scenario": name,
            "command_xy_yaw": command,
            "initial_history_action": initial,
            "history_input": "clipped_to_1" if i >= count else "raw",
            "steps": n_steps,
            "max_abs_mean": float(values[good].max()) if good.any() else None,
            "first_nonfinite_step": int(np.flatnonzero(~good)[0]) if (~good).any() else None,
            "last_abs_mean": float(values[-1]),
            "max_std": float(np.max(std[:, i])),
        })
    return summary, maxima, std


def main():
    jax.config.update("jax_default_matmul_precision", "highest")
    cfg = json.loads((RUN / "train_config.json").read_text())
    net = networks.make_ppo_networks(
        {"state": 1260, "privileged_state": 2265}, 12,
        policy_hidden_layer_sizes=cfg["policy_hidden_layer_sizes"],
        value_hidden_layer_sizes=cfg["value_hidden_layer_sizes"],
        activation=getattr(linen, cfg["activation"]),
        value_obs_key="privileged_state", distribution_type=cfg["distribution_type"],
        noise_std_type=cfg["noise_std_type"], init_noise_std=cfg["init_noise_std"],
        state_dependent_std=False,
    )
    result = {
        "run": str(RUN.relative_to(ROOT)),
        "runtime": {"jax": jax.__version__, "devices": [str(x) for x in jax.devices()]},
        "experiment": "synthetic fixed-body policy history feedback; no physics rollout",
        "probe_settings": {"steps": 500, "control_period_s": 0.02, "gait_s": 0.72,
                           "history_frames": 15, "physical_inputs": "home motor deltas, zero velocities, upright IMU",
                           "variants": "raw history versus history input clipped to [-1, 1]; both emit raw means"},
        "checkpoints": [], "probes": [],
    }
    traces = {}
    for step in STEPS:
        path = RUN / "ckpt" / f"model_{step}"
        params = model.load_params(str(path))
        assert len(params) == 3
        stats = {name: numeric_summary(part)
                 for name, part in zip(("normalizer", "actor", "value"), params)}
        result["checkpoints"].append({"env_steps": step,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "parts": stats})
        print("Checkpoint", step, {k: {"finite": v["finite"], "max_abs": v["max_abs_finite_value"]}
                                  for k, v in stats.items()}, flush=True)
        if step in (222576640, 232693760) and stats["actor"]["finite"]:
            summary, maxima, std = feedback_probe(net.policy_network, params)
            result["probes"].append({"env_steps": step, "scenarios": summary})
            traces[f"means_{step}"] = maxima
            traces[f"std_{step}"] = std
            print("Probe", step, [(s["scenario"], s["history_input"], s["max_abs_mean"])
                                  for s in summary], flush=True)
    np.savez_compressed(OUT / "historical_policy_probe_traces.npz", **traces)
    (OUT / "historical_policy_audit.json").write_text(
        json.dumps(clean(result), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
