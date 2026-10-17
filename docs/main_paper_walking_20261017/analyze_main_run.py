"""Audit the transferred 300M main run and compare saved walking measurements.

Run with .venv-main-compare/bin/python. Reads archived results; no training or
physics rollouts are performed. Native variation profiles remain distinct.
"""

import ast
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import jax
import matplotlib
import numpy as np
from brax.io import model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MAIN = ROOT / "results/toddlerbot_2xc_walk_brax_20261016_191310"
PAPER = ROOT / "results/toddlerbot_walk_ppo_20261015_194347"
SMOKE = ROOT / "results/toddlerbot_2xc_walk_brax_20261016_182237"
GROUPS = ("all", "standing", "walking", "start", "stop")
MEASUREMENTS = ("velocity_x_m_s", "velocity_y_m_s", "xy_error_m_s",
                "xy_speed_m_s", "leg_motor_rmse_rad")
EXPECTED_STEPS = [i * 20008960 for i in range(16)]
COLORS = {"paper_native": "#197b58", "main_nominal": "#47678d", "main_native": "#b74128"}
LABELS = {"paper_native": "Paper native", "main_nominal": "Main nominal", "main_native": "Main native"}


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    assert path.read_bytes().endswith(b"\n"), path
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def expected_counts(length, moving, transitions):
    i = np.arange(length)
    walk = moving & (((i >= 150) & (i < 450)) | ((i >= 600) & (i < 850))) if transitions else np.full(length, moving)
    return {"all": length, "standing": int((~walk).sum()), "walking": int(walk.sum()),
            "start": int((walk & (((i >= 150) & (i < 175)) | ((i >= 600) & (i < 625)))).sum()) if transitions else 0,
            "stop": int((~walk & (((i >= 450) & (i < 475)) | ((i >= 850) & (i < 875)))).sum()) if transitions and moving else 0}


def audit_evaluations(suites):
    max_diff = 0.0
    episodes = 0
    condition_count = 0
    nonfinite_count = 0
    for suite in suites:
        assert suite["protocol"] == "tb_signed_xy_v1"
        assert suite["seed"] == 12102 and suite["episode_length"] == 1000
        assert suite["control_period_s"] == 0.02 and suite["deterministic_policy"]
        assert suite["benchmark_height_range_m"] == [0.2, 0.4]
        assert suite["native_height_range_m"] == [0.2, 1.0]
        assert suite["automatic_resets"] is False and suite["internal_motion_teleport"] is False
        assert suite["initial_generalized_velocity"] == ("zero" if suite["profile"] == "nominal" else "native_reference")
        assert len(suite["conditions"]) == 15
        for condition in suite["conditions"]:
            n = condition["episodes"]
            assert n == 128
            per = condition["per_episode"]
            lengths = np.array(per["length"])
            assert len(lengths) == n and ((lengths >= 1) & (lengths <= 1000)).all()
            assert condition["fall_count"] == sum(per["fell"])
            assert condition["nonfinite_count"] == sum(per["nonfinite"]) == 0
            assert condition["native_terminal_count_before_benchmark_stop"] == sum(per["native_terminal_before_benchmark_stop"])
            assert abs(float(lengths.mean()) - condition["mean_episode_length"]) < 1e-6
            moving = np.linalg.norm(condition["command_xy_m_s"]) > 1e-6
            expected = [expected_counts(int(l), moving, condition["transitions"]) for l in lengths]
            for group in GROUPS:
                raw = per[group]
                counts = np.array(raw["sample_count"])
                assert len(counts) == n
                assert np.array_equal(counts, [e[group] for e in expected])
                summary = condition["groups"][group]
                assert summary["sample_count"] == int(counts.sum())
                assert summary["episode_count"] == int((counts > 0).sum())
                for measurement in MEASUREMENTS:
                    values = raw[measurement]
                    assert len(values) == n
                    assert all((v is not None) == (c > 0) for v, c in zip(values, counts))
                    eligible = np.array([v for v in values if v is not None], dtype=np.float64)
                    assert np.isfinite(eligible).all()
                    if len(eligible):
                        mean_diff = abs(float(eligible.mean()) - summary[measurement]["mean"])
                        std_diff = abs(float(eligible.std()) - summary[measurement]["std"])
                        max_diff = max(max_diff, mean_diff, std_diff)
                        assert max(mean_diff, std_diff) < 1e-6
                    else:
                        assert summary[measurement] == {"mean": None, "std": None}
            condition_count += 1
            episodes += n
            nonfinite_count += condition["nonfinite_count"]
    return {"suites": len(suites), "conditions": condition_count, "episode_conditions": episodes,
            "nonfinite_episode_conditions": nonfinite_count, "max_summary_reconstruction_difference": max_diff,
            "group_counts_reconstruct_from_episode_length_and_schedule": True}


def progress_row(suite, profile):
    conditions = suite["conditions"]
    fixed = [c for c in conditions if not c["transitions"] and c["name"] != "standing"]
    standing = next(c for c in conditions if c["name"] == "standing")
    transitions = [c for c in conditions if c["transitions"]]
    transition_errors = [c["groups"]["walking"]["xy_error_m_s"]["mean"] for c in transitions
                         if c["groups"]["walking"]["episode_count"] > 0]
    metric = lambda c, key: c["groups"]["all"][key]["mean"]
    return {"env_steps": suite.get("env_steps", suite.get("num_steps")), "profile": profile,
            "fixed_xy_error_m_s": float(np.mean([metric(c, "xy_error_m_s") for c in fixed])),
            "fixed_leg_motor_rmse_rad": float(np.mean([metric(c, "leg_motor_rmse_rad") for c in fixed])),
            "fixed_falls": sum(c["fall_count"] for c in fixed), "fixed_episodes": sum(c["episodes"] for c in fixed),
            "standing_speed_m_s": metric(standing, "xy_speed_m_s"), "standing_falls": standing["fall_count"],
            "transition_falls": sum(c["fall_count"] for c in transitions),
            "transition_walking_xy_error_m_s": float(np.mean(transition_errors)) if transition_errors else None,
            "transition_walking_eligible_conditions": len(transition_errors),
            "transition_walking_eligible_episodes": sum(c["groups"]["walking"]["episode_count"] for c in transitions),
            "all_falls": sum(c["fall_count"] for c in conditions), "all_episodes": sum(c["episodes"] for c in conditions),
            "nonfinite_episode_conditions": sum(c["nonfinite_count"] for c in conditions),
            "mean_episode_length": float(np.mean([c["mean_episode_length"] for c in conditions]))}


def save_figure(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=180, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def plot_paper_progress():
    """Keep the existing paper-only figure current after the main transfer."""
    rows = list(csv.DictReader((OUT / "paper_progress.csv").open()))
    steps = [float(r["env_steps"]) / 1e6 for r in rows]
    panels = [("native_return", "Native evaluation return", "Paper reward units"),
              ("fixed_xy_error_m_s", "Fixed walking XY error", "m/s"),
              ("all_falls", "Falls across all 15 conditions", "% of 1,920 episodes"),
              ("standing_speed_m_s", "Reset-standing speed", "m/s"),
              ("fixed_leg_rmse_rad", "Leg error against paper reference", "rad"),
              ("elapsed_s", "Native progress callback elapsed time", "hours")]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for ax, (key, title, unit) in zip(axs.flat, panels):
        values = [float(r[key]) for r in rows]
        if key == "all_falls":
            values = [100 * v / float(r["all_episodes"]) for v, r in zip(values, rows)]
        if key == "elapsed_s":
            values = [v / 3600 for v in values]
        ax.plot(steps, values, color=COLORS["paper_native"], marker=".")
        ax.axvspan(180.08064, 200.0896, color=COLORS["paper_native"], alpha=0.12)
        ax.axvline(240.10752, color="black", linestyle="--", alpha=0.6)
        ax.set(title=title, xlabel="Collected transitions (M)", ylabel=unit)
        ax.grid(alpha=0.2)
    fig.suptitle("Verified paper seed 0: native variation through 300,134,400 transitions\n"
                 "Shading: 180–200M physical candidates; dashed line: 240M native return-best", fontsize=12)
    save_figure(fig, "paper_progress")


def main():
    cfg = read_json(MAIN / "train_config.json")
    env = read_json(MAIN / "env_config.json")
    args = read_json(MAIN / "args.json")
    provenance = read_json(MAIN / "provenance.json")
    for line in (MAIN / "source/toddlerbot/locomotion/main_compare.gin").read_text().splitlines():
        if not line.startswith("PPOConfig."):
            continue
        key, value = line.split("=", 1)
        expected = ast.literal_eval(value.strip())
        if isinstance(expected, tuple):
            expected = list(expected)
        assert cfg[key.split(".")[1].strip()] == expected, key
    assert args["brax"] and not args["torch"] and args["restore"] == ""
    assert args["gin_file"] == "main_compare" and cfg["seed"] == 0
    assert env == read_json(SMOKE / "env_config.json")
    source_checks = {}
    local_source_differences = []
    for name, expected in provenance["source_sha256"].items():
        assert sha256(MAIN / "source" / name) == expected, name
        source_checks[name] = expected
        if (ROOT / name).is_file() and sha256(ROOT / name) != expected:
            local_source_differences.append(name)
    for name, expected in provenance["model_reference_sha256"].items():
        assert sha256(ROOT / name) == expected, name
    assert sha256(ROOT / provenance["lock"]["name"]) == provenance["lock"]["sha256"]
    assert any(d["platform"] == "gpu" and d["kind"] == "NVIDIA GeForce RTX 5070" for d in provenance["devices"])

    checkpoints = read_jsonl(MAIN / "checkpoints.jsonl")
    assert [r["env_steps"] for r in checkpoints] == EXPECTED_STEPS
    hash_steps = {}
    for r in checkpoints:
        f = MAIN / r["file"]
        assert sha256(f) == r["sha256"] and r["parameters_finite"]
        params = model.load_params(str(f))
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params))
        hash_steps[r["sha256"]] = r["env_steps"]
    best_step = hash_steps[sha256(MAIN / "model_best")]
    last_step = hash_steps[sha256(MAIN / "model_last")]
    assert last_step == EXPECTED_STEPS[-1]
    print("Checkpoint hashes and all parameters passed; final steps:", last_step, flush=True)

    metrics = read_jsonl(MAIN / "metrics.jsonl")
    metric_groups = defaultdict(list)
    for row in metrics:
        assert all(np.isfinite(v) for v in row["metrics"].values())
        metric_groups[row["env_steps"]].append(row)
    assert sorted(metric_groups) == EXPECTED_STEPS[1:]
    training_rows = []
    for step, rows in sorted(metric_groups.items()):
        assert len(rows) == 2
        episode = next(r for r in rows if "episode/sum_reward" in r["metrics"])
        training = next(r for r in rows if "training/total_loss" in r["metrics"])
        ep = episode["metrics"]
        decomposition = {name: 0.02 * ep[f"episode/{name}"] for name, weight in env["reward_scales"].items() if weight}
        # Separate float32 episode accumulators introduce a small rounding gap.
        component_gap = abs(sum(decomposition.values()) - ep["episode/sum_reward"])
        assert np.isclose(sum(decomposition.values()), ep["episode/sum_reward"], rtol=1e-6, atol=1e-6)
        training_rows.append({"env_steps": step, "native_training_return": ep["episode/sum_reward"],
                              "native_training_episode_length": ep["episode/length"],
                              "episode_callback_walltime_s": episode["walltime_s"],
                              "epoch_callback_walltime_s": training["walltime_s"], **training["metrics"],
                              "reward_component_reconstruction_gap": component_gap,
                              **{f"return_component/{k}": v for k, v in decomposition.items()}})
    assert max(training_rows, key=lambda r: r["native_training_return"])["env_steps"] == best_step
    write_csv("main_training_progress.csv", training_rows)

    suites = read_jsonl(MAIN / "xy_evaluation.jsonl")
    assert len(suites) == 32
    assert sorted((r["env_steps"], r["profile"]) for r in suites) == sorted(
        (step, profile) for step in EXPECTED_STEPS for profile in ("nominal", "native"))
    eval_audit = audit_evaluations(suites)
    baseline_audit = read_json(OUT / "paper_audit.json")
    for name, expected in baseline_audit["input_sha256"].items():
        assert sha256(PAPER / name) == expected, name
    paper_suites = read_jsonl(PAPER / "xy_evaluation.jsonl")
    assert [r["num_steps"] for r in paper_suites] == EXPECTED_STEPS[1:]
    all_suites = [(r, f"main_{r['profile']}") for r in suites] + [(r, "paper_native") for r in paper_suites]
    progress = sorted([progress_row(r, profile) for r, profile in all_suites], key=lambda r: (r["env_steps"], r["profile"]))
    write_csv("main_paper_matched_progress.csv", [r for r in progress if r["env_steps"] > 0])
    finals = [(r, profile) for r, profile in all_suites if r.get("env_steps", r.get("num_steps")) == last_step]
    conditions = []
    for suite, profile in finals:
        for c in suite["conditions"]:
            for group, summary in c["groups"].items():
                row = {"profile": profile, "env_steps": last_step, "condition": c["name"], "group": group,
                       "command_x_m_s": c["command_xy_m_s"][0], "command_y_m_s": c["command_xy_m_s"][1],
                       "falls": c["fall_count"], "episodes": c["episodes"], "mean_episode_length": c["mean_episode_length"],
                       "nonfinite": c["nonfinite_count"], "sample_count": summary["sample_count"], "eligible_episodes": summary["episode_count"]}
                for measurement in MEASUREMENTS:
                    for statistic in ("mean", "std"):
                        row[f"{measurement}/{statistic}"] = summary[measurement][statistic]
                conditions.append(row)
    write_csv("main_paper_final_conditions.csv", conditions)

    fixed_commands = [c["command_xy_m_s"] for c in finals[0][0]["conditions"]
                      if not c["transitions"] and c["name"] != "standing"]
    zero_velocity_baseline = float(np.mean(np.linalg.norm(fixed_commands, axis=1)))
    main_below_baseline = {profile: sum(r["fixed_xy_error_m_s"] < zero_velocity_baseline
                                       for r in progress if r["profile"] == profile and r["env_steps"] > 0)
                           for profile in ("main_nominal", "main_native")}
    gains = []
    for suite, profile in finals:
        cs = {c["name"]: c for c in suite["conditions"]}
        measured = lambda name, key: cs[name]["groups"]["all"][key]["mean"]
        gains.append({"profile": profile,
                      "forward_backward_command_response_ratio": (measured("forward_010", "velocity_x_m_s") - measured("backward_010", "velocity_x_m_s")) / 0.2,
                      "left_right_command_response_ratio": (measured("left_005", "velocity_y_m_s") - measured("right_005", "velocity_y_m_s")) / 0.1})
    final_progress = [r for r in progress if r["env_steps"] == last_step]
    inputs = [MAIN / name for name in ("args.json", "train_config.json", "env_config.json", "provenance.json",
                                      "metrics.jsonl", "checkpoints.jsonl", "xy_evaluation.jsonl", "output.log", "model_best", "model_last")]
    audit = {"analysis_date": "2026-10-17", "status": "ready_for_physical_result_analysis",
             "main_run": str(MAIN.relative_to(ROOT)), "smoke_run": str(SMOKE.relative_to(ROOT)),
             "source_revision": provenance["head"], "main_base": provenance["main_base"], "recorded_git_status": provenance["git_status"],
             "devices": provenance["devices"], "packages": {k: provenance["packages"][k] for k in ("brax", "jax", "jaxlib", "flax", "mujoco", "mujoco-mjx")},
             "requested_steps": cfg["num_timesteps"], "actual_steps": last_step, "seed": cfg["seed"],
             "checkpoint_records": len(checkpoints), "checkpoint_hashes_verified": True, "all_loaded_checkpoint_parameters_finite": True,
             "native_return_best_step": best_step, "native_return_best": max(r["native_training_return"] for r in training_rows),
             "metric_records": len(metrics), "metric_records_per_trained_step": 2, "all_native_scalar_metrics_finite": True,
             "metric_callback_note": "episode-window callback and epoch training callback have distinct metrics and times; preserve both",
             "evaluation_audit": eval_audit, "source_snapshots_verified": len(source_checks),
             "local_model_reference_hashes_verified": len(provenance["model_reference_sha256"]),
             "local_source_files_differing_from_run": local_source_differences,
             "source_difference_note": "local train entry point moves preflight before Robot creation; dependency helper rejects unexpected CUDA plugins. Reward, environment, evaluator and PPO call are unchanged.",
             "fresh_effective_config_matches_prepared_overlay": True, "paper_inputs_match_prior_audit": True,
             "main_wandb_offline_directory_available": False,
             "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
             "checkpoint_sha256": {r["file"]: r["sha256"] for r in checkpoints},
             "final_progress": final_progress, "final_signed_response_ratios": gains,
             "analytic_zero_velocity_fixed_xy_error_m_s": zero_velocity_baseline,
             "trained_main_checkpoints_below_analytic_zero_velocity_error": main_below_baseline,
             "median_main_epoch_training_sps_after_first": float(np.median([r["training/sps"] for r in training_rows[1:]])),
             "final_native_training_timer_s": training_rows[-1]["training/walltime"],
             "final_epoch_callback_walltime_s": training_rows[-1]["epoch_callback_walltime_s"],
             "max_reward_component_reconstruction_gap": max(r["reward_component_reconstruction_gap"] for r in training_rows),
             "final_return_components": {k.removeprefix("return_component/"): v for k, v in training_rows[-1].items() if k.startswith("return_component/")}}
    (OUT / "main_run_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    physical = (("fixed_xy_error_m_s", "Fixed walking XY error", "m/s"),
                ("all_falls", "Falls across all 15 conditions", "% of 1,920 episodes"),
                ("fixed_leg_motor_rmse_rad", "Leg error against each recipe's reference", "rad"),
                ("standing_speed_m_s", "Reset-standing mean instantaneous speed", "m/s"))
    for ax, (key, title, unit) in zip([axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0]], physical):
        for profile in COLORS:
            rows = [r for r in progress if r["profile"] == profile and r["env_steps"] > 0]
            values = [100 * r[key] / r["all_episodes"] if key == "all_falls" else r[key] for r in rows]
            ax.plot([r["env_steps"] / 1e6 for r in rows], values, marker=".", color=COLORS[profile], label=LABELS[profile])
        ax.set(title=title, xlabel="Collected transitions (M)", ylabel=unit)
        if key in ("fixed_xy_error_m_s", "standing_speed_m_s"):
            ax.set_yscale("log")
        if key == "all_falls":
            ax.set_yscale("symlog", linthresh=1)
            ax.set_yticks([0, 1, 10, 100], ["0", "1", "10", "100"])
        ax.grid(alpha=0.2)
    axs[0, 0].legend(fontsize=9)
    axs[1, 1].plot([r["env_steps"] / 1e6 for r in training_rows], [r["native_training_return"] for r in training_rows], color=COLORS["main_native"], marker=".")
    axs[1, 1].axvline(best_step / 1e6, color="black", linestyle=":")
    axs[1, 1].set(title="Main native training episode return", xlabel="Collected transitions (M)", ylabel="Main reward units")
    paper_metrics = read_jsonl(PAPER / "metrics.jsonl")
    axs[1, 2].plot([r["num_steps"] / 1e6 for r in paper_metrics], [r["metrics"]["eval/episode_reward"] for r in paper_metrics], color=COLORS["paper_native"], marker=".")
    axs[1, 2].axvline(240.10752, color="black", linestyle=":")
    axs[1, 2].set(title="Paper native random-command evaluation return", xlabel="Collected transitions (M)", ylabel="Paper reward units")
    for ax in axs[1, 1:]:
        ax.grid(alpha=0.2)
    fig.suptitle("Verified 300M runs: main learns survival with weak signed command response\nNative profiles use different variation; raw returns use different objectives and sampling", fontsize=12)
    save_figure(fig, "main_paper_progress")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for suite, profile in finals:
        cs = {c["name"]: c for c in suite["conditions"]}
        for ax, names, command_index, key in ((axs[0], ["backward_010", "backward_005", "forward_005", "forward_010"], 0, "velocity_x_m_s"),
                                              (axs[1], ["right_005", "left_005"], 1, "velocity_y_m_s")):
            commands = [cs[n]["command_xy_m_s"][command_index] for n in names]
            means = [cs[n]["groups"]["all"][key]["mean"] for n in names]
            std = [cs[n]["groups"]["all"][key]["std"] for n in names]
            ax.errorbar(commands, means, yerr=std, color=COLORS[profile], marker="o", capsize=3, label=LABELS[profile])
    for ax, extent, title in ((axs[0], 0.11, "Forward / backward command following"), (axs[1], 0.06, "Left / right command following")):
        ax.plot([-extent, extent], [-extent, extent], color="black", linestyle="--", label="Perfect mean tracking")
        ax.axhline(0, color="black", alpha=0.25)
        ax.set(xlim=(-extent, extent), xlabel="Commanded local velocity (m/s)", ylabel="Mean measured local velocity (m/s)", title=title)
        ax.grid(alpha=0.2)
    axs[0].legend(fontsize=9)
    fig.suptitle("Final 300,134,400-step policies: main remains close to zero velocity\nError bars: population SD of episode means; nominal fixes reset/model settings; SD is not training-seed uncertainty", fontsize=11)
    save_figure(fig, "main_paper_signed_response")
    plot_paper_progress()
    print("Evaluation audit:", eval_audit, flush=True)
    print("Final measurements:", final_progress, flush=True)
    print("Signed response ratios:", gains, flush=True)
    print("Native return best:", best_step, audit["native_return_best"], flush=True)


if __name__ == "__main__":
    main()
