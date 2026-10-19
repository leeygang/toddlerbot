"""Recheck copied results directly from episode records and retained policies.

Run with .venv-main-compare/bin/python. Produces review_20261018.json without
running training or physics. This independently checks the saved summaries,
console, checkpoint aliases and the measurements used to rank next steps.
"""

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
from brax.io import model


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MAIN = ROOT / "results/toddlerbot_2xc_walk_brax_20261016_191310"
PAPER = ROOT / "results/toddlerbot_walk_ppo_20261015_194347"
GROUPS = ("all", "standing", "walking", "start", "stop")
MEASUREMENTS = ("velocity_x_m_s", "velocity_y_m_s", "xy_error_m_s",
                "xy_speed_m_s", "leg_motor_rmse_rad")
STEPS = [i * 20008960 for i in range(16)]


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    assert path.read_bytes().endswith(b"\n"), path
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_policies(folder, files, aliases):
    steps_by_hash = {}
    leaves = 0
    for step, file in files:
        params = model.load_params(str(file))
        arrays = [np.asarray(x) for x in jax.tree.leaves(params)]
        assert arrays and all(np.isfinite(x).all() for x in arrays), file
        leaves += len(arrays)
        steps_by_hash[sha256(file)] = step
    matches = {name: steps_by_hash[sha256(folder / name)] for name in aliases}
    return {"retained_files": len(files), "all_parameters_finite": True,
            "finite_parameter_leaves": leaves, "alias_steps": matches}


def expected_counts(lengths, moving, transitions):
    def window(start, stop):
        return np.maximum(np.minimum(lengths, stop) - start, 0)

    walking = (window(150, 450) + window(600, 850)) if transitions else lengths
    walking = walking if moving else np.zeros_like(lengths)
    return {"all": lengths, "walking": walking, "standing": lengths - walking,
            "start": window(150, 175) + window(600, 625) if transitions and moving else np.zeros_like(lengths),
            "stop": window(450, 475) + window(850, 875) if transitions and moving else np.zeros_like(lengths)}


def audit_suites(suites):
    maximum_difference = 0.0
    episodes = 0
    for suite in suites:
        assert suite["seed"] == 12102 and suite["episode_length"] == 1000
        assert suite["control_period_s"] == 0.02 and suite["deterministic_policy"]
        assert len(suite["conditions"]) == 15
        for condition in suite["conditions"]:
            raw = condition["per_episode"]
            lengths = np.asarray(raw["length"])
            assert condition["episodes"] == len(lengths) == 128
            assert ((lengths >= 1) & (lengths <= 1000)).all()
            assert condition["fall_count"] == sum(raw["fell"])
            assert condition["nonfinite_count"] == sum(raw["nonfinite"]) == 0
            assert np.isclose(lengths.mean(), condition["mean_episode_length"], rtol=0, atol=1e-6)
            expected = expected_counts(lengths, np.linalg.norm(condition["command_xy_m_s"]) > 1e-6,
                                       condition["transitions"])
            for group in GROUPS:
                counts = np.asarray(raw[group]["sample_count"])
                summary = condition["groups"][group]
                assert np.array_equal(counts, expected[group])
                assert summary["sample_count"] == int(counts.sum())
                assert summary["episode_count"] == int((counts > 0).sum())
                for name in MEASUREMENTS:
                    values = raw[group][name]
                    assert len(values) == 128
                    assert all((v is not None) == (n > 0) for v, n in zip(values, counts))
                    eligible = np.asarray([v for v in values if v is not None], dtype=np.float64)
                    assert np.isfinite(eligible).all()
                    if len(eligible):
                        differences = [abs(eligible.mean() - summary[name]["mean"]),
                                       abs(eligible.std() - summary[name]["std"])]
                        maximum_difference = max(maximum_difference, *differences)
                        assert max(differences) < 1e-6
                    else:
                        assert summary[name] == {"mean": None, "std": None}
            episodes += len(lengths)
    return {"suites": len(suites), "episode_conditions": episodes,
            "nonfinite_episode_conditions": 0,
            "max_summary_difference": float(maximum_difference)}


def condition_mean(condition, measurement):
    return float(np.mean(condition["per_episode"]["all"][measurement]))


def physical_row(suite, profile):
    conditions = {c["name"]: c for c in suite["conditions"]}
    fixed = [c for c in conditions.values() if not c["transitions"] and c["name"] != "standing"]
    longitudinal = (condition_mean(conditions["forward_010"], "velocity_x_m_s") -
                    condition_mean(conditions["backward_010"], "velocity_x_m_s")) / 0.20
    lateral = (condition_mean(conditions["left_005"], "velocity_y_m_s") -
               condition_mean(conditions["right_005"], "velocity_y_m_s")) / 0.10
    return {"steps": suite.get("env_steps", suite.get("num_steps")), "profile": profile,
            "fixed_xy_error_m_s": float(np.mean([condition_mean(c, "xy_error_m_s") for c in fixed])),
            "own_reference_leg_rmse_rad": float(np.mean([condition_mean(c, "leg_motor_rmse_rad") for c in fixed])),
            "standing_speed_m_s": condition_mean(conditions["standing"], "xy_speed_m_s"),
            "falls": sum(sum(c["per_episode"]["fell"]) for c in conditions.values()), "episodes": 1920,
            "mean_episode_length": float(np.mean([c["per_episode"]["length"] for c in conditions.values()])),
            "forward_backward_response_ratio": longitudinal, "left_right_response_ratio": lateral,
            "command_axis_velocity_m_s": {
                name: condition_mean(conditions[name], axis) for name, axis in (
                    ("forward_010", "velocity_x_m_s"), ("backward_010", "velocity_x_m_s"),
                    ("right_005", "velocity_y_m_s"), ("left_005", "velocity_y_m_s"))}}


def archived_method(path, name):
    """Evaluate only the arithmetic of a reviewed reward method with NumPy."""
    tree = ast.parse(path.read_text())
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(methods) == 1
    method = methods[0]
    for arg in method.args.args:
        arg.annotation = None
    method.returns = None
    namespace = {"jnp": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def main():
    main_audit = read_json(OUT / "main_run_audit.json")
    paper_audit = read_json(OUT / "paper_audit.json")
    input_hashes = {}
    for audit, folder in ((main_audit, ROOT), (paper_audit, PAPER)):
        for relative, expected in audit["input_sha256"].items():
            file = folder / relative
            assert sha256(file) == expected, file
            input_hashes[str(file.relative_to(ROOT))] = expected

    main_provenance = read_json(MAIN / "provenance.json")
    paper_provenance = read_json(PAPER / "provenance.json")
    for relative, expected in main_provenance["source_sha256"].items():
        assert sha256(MAIN / "source" / relative) == expected, relative
    for relative, expected in main_provenance["model_reference_sha256"].items():
        assert sha256(ROOT / relative) == expected, relative
    paper_source_files = []
    revision = paper_provenance["source_revision"]
    for file in sorted((PAPER / "locomotion").iterdir()):
        if file.is_file():
            relative = f"toddlerbot/locomotion/{file.name}"
            recorded = subprocess.run(["git", "show", f"{revision}:{relative}"], check=True, capture_output=True)
            assert recorded.stdout == file.read_bytes(), relative
            paper_source_files.append(relative)
    for relative, expected in paper_provenance["model_reference_sha256"].items():
        recorded = subprocess.run(["git", "show", f"{revision}:{relative}"], check=True, capture_output=True)
        assert hashlib.sha256(recorded.stdout).hexdigest() == expected, relative

    checkpoints = read_jsonl(MAIN / "checkpoints.jsonl")
    assert [r["env_steps"] for r in checkpoints] == STEPS
    for record in checkpoints:
        assert sha256(MAIN / record["file"]) == record["sha256"]
    main_policies = verify_policies(MAIN, [(r["env_steps"], MAIN / r["file"]) for r in checkpoints],
                                    ("model_best", "model_last"))
    paper_policies = verify_policies(PAPER, [(step, PAPER / str(step) / "policy") for step in STEPS[1:]],
                                     ("best_policy", "policy"))
    print("Verified all 31 retained policies and best/last aliases.", flush=True)

    main_metrics = read_jsonl(MAIN / "metrics.jsonl")
    paper_metrics = read_jsonl(PAPER / "metrics.jsonl")
    assert len(main_metrics) == 30 and len(paper_metrics) == 15
    assert [r["num_steps"] for r in paper_metrics] == STEPS[1:]
    for record in main_metrics + paper_metrics:
        assert all(np.isfinite(value) for value in record["metrics"].values())
    main_suites = read_jsonl(MAIN / "xy_evaluation.jsonl")
    paper_suites = read_jsonl(PAPER / "xy_evaluation.jsonl")
    assert sorted((r["env_steps"], r["profile"]) for r in main_suites) == sorted(
        (step, profile) for step in STEPS for profile in ("nominal", "native"))
    assert [r["num_steps"] for r in paper_suites] == STEPS[1:]
    signature = lambda r: [(c["name"], c["command_xy_m_s"], c["transitions"]) for c in r["conditions"]]
    assert all(signature(r) == signature(paper_suites[0]) for r in main_suites + paper_suites)
    evaluation_audits = {"main": audit_suites(main_suites), "paper": audit_suites(paper_suites)}
    rows = [physical_row(r, "main_" + r["profile"]) for r in main_suites]
    rows += [physical_row(r, "paper_native") for r in paper_suites]
    finals = [r for r in rows if r["steps"] == STEPS[-1]]
    for final in finals:
        previous = next(r for r in main_audit["final_progress"] if r["profile"] == final["profile"])
        assert np.isclose(final["fixed_xy_error_m_s"], previous["fixed_xy_error_m_s"], rtol=0, atol=1e-6)
        assert final["falls"] == previous["all_falls"]
    print("Reconstructed all 90,240 episode-condition summaries; no nonfinite evaluations.", flush=True)

    console = re.findall(r"XY (nominal|native), step (\d+): falls=(\d+)/(\d+), forward error=([^\n]+)",
                         (MAIN / "output.log").read_text())
    assert len(console) == len(main_suites)
    for profile, step, falls, episodes, error in console:
        suite = next(r for r in main_suites if r["profile"] == profile and r["env_steps"] == int(step))
        row = next(r for r in rows if r["profile"] == "main_" + profile and r["steps"] == int(step))
        forward = next(c for c in suite["conditions"] if c["name"] == "forward_010")
        assert row["falls"] == int(falls) and row["episodes"] == int(episodes)
        assert np.isclose(condition_mean(forward, "xy_error_m_s"), float(error), rtol=0, atol=1e-6)

    fixed = [c for c in paper_suites[-1]["conditions"] if not c["transitions"] and c["name"] != "standing"]
    zero_baseline = float(np.mean([np.linalg.norm(c["command_xy_m_s"]) for c in fixed]))
    main_better = {profile: sum(r["fixed_xy_error_m_s"] < zero_baseline for r in rows
                               if r["profile"] == profile and r["steps"] > 0)
                   for profile in ("main_nominal", "main_native")}
    main_cfg, paper_cfg = read_json(MAIN / "env_config.json"), read_json(PAPER / "env_config.json")
    main_method = archived_method(MAIN / "source/toddlerbot/locomotion/mjx_env.py", "_reward_penalty_action_rate")
    paper_method = archived_method(PAPER / "locomotion/mjx_env.py", "_reward_leg_action_rate")
    info = {"last_act": np.zeros(12)}
    action = np.full(12, 0.1)
    main_penalty = main_cfg["reward_scales"]["penalty_action_rate"] * main_method(None, None, info, action)
    paper_penalty = paper_cfg["reward_scales"]["leg_action_rate"] * paper_method(
        SimpleNamespace(leg_actuator_indices=np.arange(12)), None, info, action)
    assert np.isclose(main_penalty / paper_penalty, 480)
    last_episode = next(r["metrics"] for r in main_metrics if r["env_steps"] == STEPS[-1]
                        and "episode/sum_reward" in r["metrics"])
    components = {name: 0.02 * last_episode[f"episode/{name}"]
                  for name, weight in main_cfg["reward_scales"].items() if weight}
    assert np.isclose(sum(components.values()), last_episode["episode/sum_reward"], rtol=1e-6, atol=1e-6)

    review = {"review_date": "2026-10-18", "status": "prior_physical_findings_confirmed_priorities_revised",
              "input_sha256_unchanged": input_hashes, "final_steps_both_runs": STEPS[-1],
              "main_policies": main_policies, "paper_policies": paper_policies,
              "main_archived_source_files_verified": len(main_provenance["source_sha256"]),
              "main_model_reference_hashes_verified": len(main_provenance["model_reference_sha256"]),
              "paper_archived_source_files_match_recorded_head": paper_source_files,
              "paper_model_reference_hashes_match_recorded_head": len(paper_provenance["model_reference_sha256"]),
              "paper_recorded_source_dirty": paper_provenance["source_dirty"],
              "paper_dirty_flag_limit": "Archived locomotion source and recorded model/reference hashes match HEAD; scope of other dirty files was not recorded.",
              "finite_metric_records": {"main": len(main_metrics), "paper": len(paper_metrics)},
              "evaluation_audits": evaluation_audits, "main_console_evaluation_records_confirmed": len(console),
              "final_results_from_per_episode_records": finals,
              "analytic_zero_velocity_error_m_s": zero_baseline,
              "trained_main_checkpoints_below_zero_velocity_baseline": main_better,
              "paper_shortlist": [r for r in rows if r["profile"] == "paper_native" and r["steps"] in (180080640, 200089600, 240107520, STEPS[-1])],
              "main_shortlist": [r for r in rows if r["profile"].startswith("main_") and r["steps"] in (80035840, 220098560, STEPS[-1])],
              "action_change_penalties_before_dt_at_rms_0_1": {"main": float(main_penalty), "paper": float(paper_penalty), "ratio": float(main_penalty / paper_penalty)},
              "main_final_native_return_components": components,
              "main_foot_to_xy_return_component_ratio": components["feet_phase"] / components["lin_vel_xy"],
              "causal_limit": "No reward ablation or shared robustness reevaluation has been performed; individual causes are hypotheses.",
              "priorities": ["Recover main signed tracking: width-only 1000 to 100 at XY weight 2; independent action-change weight 2 to 0.05/12",
                             "Preserve paper 180M/200M and reevaluate under common nominal and fresh draws",
                             "Screen physical outcomes at 20M/40M/60M/80M before extending training"],
              "historical_instability_role": "Monitor finite policies/actions; fresh main run shows no historical NaN failure."}
    (OUT / "review_20261018.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"final_results": finals, "review": str(OUT / "review_20261018.json")}, indent=2))


if __name__ == "__main__":
    main()
