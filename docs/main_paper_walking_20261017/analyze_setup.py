"""Reproduce source-based reward comparisons and historical main diagnostics.

Run from the repository with .venv-main-compare/bin/python. This reads local
artifacts and writes report assets; it does not train or evaluate robot physics.
"""

import ast
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PAPER = ROOT / "results/toddlerbot_walk_ppo_20261015_194347"
OLD = ROOT / "results/toddlerbot_2xc_walk_brax_20260515_114507"
SMOKE = ROOT / ".worktrees/main-walking-compare/results/toddlerbot_2xc_walk_brax_20261016_113551"


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def parse_console(path):
    """Preserve native console definitions, including rounded floats and NaNs."""
    rows = []
    row = None
    for line in path.read_text().splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        match = re.search(r"Learning steps (\d+)/(\d+)", line)
        if match:
            row = {"env_steps": int(match[1]), "requested_steps": int(match[2])}
            rows.append(row)
            continue
        if row is None:
            continue
        match = re.match(r"\s*Mean episode ([a-zA-Z_]+):\s*(\S+)", line)
        if match:
            key = "mean_episode_length" if match[1] == "length" else match[1]
            row[key] = float(match[2])
            continue
        match = re.match(r"\s*Mean (reward|episode length):\s*(\S+)", line)
        if match:
            row["mean_" + match[1].replace(" ", "_")] = float(match[2])
    assert all(a["env_steps"] < b["env_steps"] for a, b in zip(rows, rows[1:]))
    return rows


def save_figure(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=180, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def main():
    paper_env = read_json(PAPER / "env_config.json")
    paper_train = read_json(PAPER / "train_config.json")
    main_env = read_json(SMOKE / "env_config.json")
    main_train = read_json(SMOKE / "train_config.json")
    # Replace the smoke-only training settings with the full prepared overlay.
    for line in (ROOT / "toddlerbot/locomotion/main_compare.gin").read_text().splitlines():
        match = re.match(r"PPOConfig\.(\w+)\s*=\s*(.*)", line)
        if match:
            main_train[match[1]] = ast.literal_eval(match[2])
    old_env = read_json(OLD / "env_config.json")
    old_train = read_json(OLD / "train_config.json")
    env_matches = {k: old_env[k] == v for k, v in main_env.items()}
    assert all(env_matches.values())

    smoke_provenance = read_json(SMOKE / "provenance.json")
    checked_sources = (
        "toddlerbot/locomotion/walk.gin",
        "toddlerbot/locomotion/main_compare.gin",
        "toddlerbot/locomotion/walk_env.py",
        "toddlerbot/locomotion/mjx_env.py",
        "toddlerbot/reference/walk_zmp_ref.py",
    )
    source_hashes = {p: sha256(ROOT / p) for p in checked_sources}
    expected_hashes = smoke_provenance["source_sha256"]
    assert all(source_hashes[p] == expected_hashes[p] for p in checked_sources)

    n_actions = len(main_env["rewards"]["pose_weights"])
    assert n_actions == 12
    paper_weight = paper_env["reward_scales"]["lin_vel_xy"]
    paper_k = paper_env["rewards"]["tracking_sigma"]
    main_weight = main_env["reward_scales"]["lin_vel_xy"]
    main_k = main_env["rewards"]["lin_vel_tracking_sigma"]
    paper_rate = paper_env["reward_scales"]["leg_action_rate"]
    main_rate = main_env["reward_scales"]["penalty_action_rate"]
    ratio = main_rate * n_actions / paper_rate
    assert math.isclose(ratio, 480.0)
    examples = [{
        "xy_vector_error_m_s": e,
        "paper_weighted_reward_before_dt": paper_weight * math.exp(-paper_k * e**2),
        "main_weighted_reward_before_dt": main_weight * math.exp(-main_k * e**2),
        "paper_over_main": paper_weight / main_weight * math.exp((main_k - paper_k) * e**2),
    } for e in (0.0, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)]

    rows = parse_console(OLD / "output.log")
    first_nonfinite, first_nan = {}, {}
    for row in rows:
        for key, value in row.items():
            if not math.isfinite(value):
                event = {"env_steps": row["env_steps"], "value": str(value)}
                first_nonfinite.setdefault(key, event)
                if math.isnan(value):
                    first_nan.setdefault(key, event)
    assert first_nonfinite["total_loss"]["env_steps"] == 231219200
    assert first_nan["policy_dist_max_loc"]["env_steps"] == 232816640
    assert first_nan["mean_reward"]["env_steps"] == 232857600
    checkpoints = sorted(
        int(p.name.removeprefix("model_")) for p in (OLD / "ckpt").glob("model_*")
        if p.name.removeprefix("model_").isdigit()
    )
    by_step = {r["env_steps"]: r for r in rows}
    sample_steps = set(checkpoints) | {
        229990400, 231178240, 231219200, 232693760, 232816640, 232857600,
    }
    sampled = [by_step[s] for s in sorted(sample_steps) if s in by_step]
    columns = ["env_steps", "mean_reward", "mean_episode_length", "lin_vel_xy",
               "feet_phase", "penalty_action_rate", "policy_dist_min_loc",
               "policy_dist_max_loc", "policy_dist_mean_std", "policy_dist_max_std",
               "v_loss", "total_loss", "sps"]
    with (OUT / "historical_main_progress.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(sampled)

    dominance = []
    for step in (229990400, 231178240, 231219200, 232693760):
        row = by_step[step]
        scaled_penalty = 0.02 * row["penalty_action_rate"]
        relative_gap = abs(scaled_penalty - row["mean_reward"]) / abs(row["mean_reward"])
        # Episode aggregation and printed float precision leave a small gap.
        assert relative_gap < 1e-6
        dominance.append({"env_steps": step, "native_return": row["mean_reward"],
                          "dt_times_action_penalty": scaled_penalty,
                          "relative_gap": relative_gap})

    inputs = [
        PAPER / "env_config.json", PAPER / "train_config.json", PAPER / "provenance.json",
        PAPER / "locomotion/mjx_env.py", PAPER / "locomotion/train_mjx.py",
        SMOKE / "env_config.json", SMOKE / "train_config.json", SMOKE / "provenance.json",
        OLD / "output.log", OLD / "train_config.json", OLD / "env_config.json",
        OLD / "args.json", OLD / "git/toddlerbot.diff",
        ROOT / "toddlerbot/locomotion/train_mjx.py",
        ROOT / "toddlerbot/brax/brax/training/agents/ppo/train.py",
        ROOT / "toddlerbot/brax/brax/training/distribution.py",
        ROOT / ".venv-main-compare/lib/python3.11/site-packages/brax/training/networks.py",
        ROOT / ".venv-main-compare/lib/python3.11/site-packages/brax/training/agents/ppo/train.py",
        OUT / "historical_policy_audit.json", OUT / "historical_policy_probe_traces.npz",
    ]
    analysis = {
        "status": "prepared setup verified; historical main failure diagnosed; new main GPU results absent",
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        "current_sources_match_main_smoke_provenance": source_hashes,
        "paper_train_config": paper_train,
        "prepared_main_train_config": main_train,
        "prepared_main_config_source": "main_compare.gin overlay plus smoke defaults; not the missing GPU run config",
        "historical_main_train_config": old_train,
        "historical_main_env_sections_match_prepared_main": env_matches,
        "reward_comparison": {
            "control_period_s": 0.02, "controlled_leg_actions": n_actions,
            "xy_formulas_before_dt": {"paper": "5 * exp(-100 * e^2)", "main": "2 * exp(-1000 * e^2)"},
            "xy_e_fold_width_m_s": {"paper": 1 / math.sqrt(paper_k), "main": 1 / math.sqrt(main_k)},
            "xy_examples": examples,
            "action_penalty_main_over_paper_for_equal_numeric_deltas": ratio,
            "main_sum_weight_matching_paper_mean_weight": paper_rate / n_actions,
            "action_rms_delta_0_1_penalty_before_dt": {"paper": paper_rate * 0.1**2, "main": main_rate * n_actions * 0.1**2},
        },
        "historical_console": {
            "progress_blocks": len(rows), "actual_final_steps": rows[-1]["env_steps"],
            "checkpoint_count": len(checkpoints), "checkpoint_steps": checkpoints,
            "first_nonfinite_by_field": first_nonfinite, "first_nan_by_field": first_nan,
            "catastrophic_return_matches_action_penalty": dominance,
            "precision_note": "native scalar console output is rounded, mostly to four decimal places",
        },
    }
    (OUT / "setup_analysis.json").write_text(json.dumps(clean(analysis), indent=2, allow_nan=False) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    paper_color, main_color = "#197b58", "#b74128"
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    e = np.linspace(0, 0.2, 501)
    axs[0].plot(e, paper_weight * np.exp(-paper_k * e**2), color=paper_color, label="Paper: 5 exp(−100 e²)")
    axs[0].plot(e, main_weight * np.exp(-main_k * e**2), color=main_color, label="Main: 2 exp(−1000 e²)")
    axs[0].set(xlabel="XY vector tracking error (m/s)", ylabel="Weighted reward before 20 ms factor",
               title="Main has a narrower velocity reward")
    axs[0].legend()
    delta = np.linspace(0.001, 0.2, 501)
    axs[1].semilogy(delta, paper_rate * delta**2, color=paper_color, label="Paper: 0.05 mean(Δa²)")
    axs[1].semilogy(delta, main_rate * n_actions * delta**2, color=main_color, label="Main: 2 sum(Δa²)")
    axs[1].set(xlabel="RMS numeric action change across 12 joints", ylabel="Penalty magnitude before 20 ms factor",
               title="Equal numeric deltas: main penalty is 480×")
    axs[1].legend()
    for ax in axs:
        ax.grid(alpha=0.2)
    fig.suptitle("Source-derived reward curves; these are not training results", fontsize=13)
    save_figure(fig, "reward_setup_comparison")

    steps = np.array([r["env_steps"] for r in rows]) / 1e6
    def series(key):
        return np.array([r.get(key, np.nan) for r in rows])
    means = np.maximum(np.abs(series("policy_dist_min_loc")), np.abs(series("policy_dist_max_loc")))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axs[0, 0].semilogy(steps, means, color=main_color)
    axs[0, 0].set(xlim=(195, 234), xlabel="Collected transitions (M)", ylabel="Reported policy mean magnitude",
                  title="Policy means explode while weights remain finite")
    axs[0, 1].plot(steps, series("policy_dist_mean_std"), color="#47678d", label="Mean std")
    axs[0, 1].plot(steps, series("policy_dist_max_std"), color="#9f7d3b", label="Maximum std")
    axs[0, 1].set(xlim=(195, 234), ylim=(0.04, 0.09), xlabel="Collected transitions (M)",
                  ylabel="Reported exploration standard deviation", title="Exploration std stays nearly constant at failure")
    axs[0, 1].legend()
    for ax in axs[0]:
        ax.axvline(231.2192, color="black", linestyle=":", alpha=0.6)
        ax.axvline(232.81664, color="black", linestyle="--", alpha=0.6)
    negative_return = -series("mean_reward")
    negative_penalty = -0.02 * series("penalty_action_rate")
    mask = (steps > 225) & (negative_return > 1e6) & np.isfinite(negative_return)
    axs[1, 0].semilogy(steps[mask], negative_return[mask], color=main_color, linewidth=3, label="Negative native episode return")
    axs[1, 0].semilogy(steps[mask], negative_penalty[mask], color="#47678d", linestyle="--", label="Negative action penalty × 0.02 s")
    axs[1, 0].set(xlim=(228, 234), xlabel="Collected transitions (M)", ylabel="Magnitude in native return units",
                  title="Raw action penalty accounts for catastrophic return")
    axs[1, 0].legend(fontsize=9)
    traces = np.load(OUT / "historical_policy_probe_traces.npz")
    for step, color in ((222576640, "#7c55a0"), (232693760, "#b74128")):
        data = traces[f"means_{step}"]
        seconds = np.arange(len(data)) * 0.02
        axs[1, 1].semilogy(seconds, data[:, 5], color=color, label=f"{step/1e6:.2f}M: raw history")
        axs[1, 1].semilogy(seconds, data[:, 12], color=color, linestyle="--", label=f"{step/1e6:.2f}M: bounded history")
    axs[1, 1].set(xlabel="Synthetic network time (s)", ylabel="Maximum absolute emitted policy mean",
                  title="Same fixed weights; only action history input differs")
    axs[1, 1].legend(fontsize=9)
    for ax in axs.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Historical May main run: loss first Inf at 231.22M; policy statistics NaN at 232.82M\n"
                 "Bottom right: synthetic fixed-body probe with initial action history +1; no physics or training", fontsize=12)
    save_figure(fig, "historical_main_diagnosis")
    print("Reward ratio:", ratio, "Console blocks:", len(rows), "Sampled rows:", len(sampled))
    print("First nonfinite loss:", first_nonfinite["total_loss"])
    print("XY examples:", examples[:6])


if __name__ == "__main__":
    main()
