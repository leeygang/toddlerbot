#!/usr/bin/env python3
"""ToddlerBot walk-reference geometry gate.

Replays the precomputed ``walk_zmp`` lookup-table references on the free-base
MuJoCo model at the ``home`` root pose and checks the same stance/swing foot
geometry gate used for WildRobot:

  - stance foot bottom z must be <= ``_STANCE_TOL_M``
  - swing foot bottom z must be > ``_SWING_MIN_M``

This gives ToddlerBot and WildRobot one shared, reference-only gate for
foot-ground consistency before any policy-side metrics are considered.

Usage::

    python3 tests/test_walk_reference_geometry.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import mujoco
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
_STANCE_TOL_M = 0.003
_SWING_MIN_M = -0.002
_VX_BINS = (0.10, 0.15, 0.20, 0.25)
_VX_BINS_INSCOPE = (0.10, 0.15)
_PROBE_FRAMES = (0, 5, 10, 16, 22, 28, 32, 48, 64)
_ROBOT_VARIANTS = ("toddlerbot_2xc", "toddlerbot_2xm")


def _box_min_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    """Return the lowest world-z of a box geom."""
    pos = data.geom_xpos[geom_id]
    rot = data.geom_xmat[geom_id].reshape(3, 3)
    sx, sy, sz = model.geom_size[geom_id]
    return float(
        pos[2]
        - abs(rot[2, 0]) * sx
        - abs(rot[2, 1]) * sy
        - abs(rot[2, 2]) * sz
    )


def _setup(robot_name: str):
    """Load model, lookup table, and foot collision geoms for one robot."""
    scene_path = _REPO_ROOT / "toddlerbot" / "descriptions" / robot_name / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    q_home = np.array(model.keyframe("home").qpos, dtype=np.float64)

    suffix = "_2xc" if "2xc" in robot_name else "_2xm"
    lookup_path = _REPO_ROOT / "motion" / f"walk_zmp{suffix}.lz4"
    lookup_keys, motion_ref_list = joblib.load(lookup_path)
    lookup_keys = np.array(lookup_keys, dtype=np.float32)

    left_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "left_ankle_roll_link_collision"
    )
    right_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "right_ankle_roll_link_collision"
    )

    return model, data, q_home, lookup_keys, motion_ref_list, left_geom, right_geom


def _find_nominal_traj(
    lookup_keys: np.ndarray, motion_ref_list: list[dict], vx: float
) -> tuple[np.ndarray, dict]:
    """Return the nearest lookup-table trajectory for ``[vx, 0, 0]``."""
    target = np.array([vx, 0.0, 0.0], dtype=np.float32)
    idx = int(np.argmin(np.linalg.norm(lookup_keys - target, axis=1)))
    return lookup_keys[idx], motion_ref_list[idx]


def _apply_frame(
    data: mujoco.MjData, q_home: np.ndarray, ref_qpos: np.ndarray
) -> None:
    """Replay one lookup-table frame on the free-base model."""
    data.qpos[:] = q_home
    data.qpos[7:] = ref_qpos
    mujoco.mj_forward(data.model, data)


def _collect_failures(
    robot_name: str, vx_bins: tuple[float, ...] = _VX_BINS
) -> list[str]:
    """Run the geometry gate and return human-readable failures."""
    (
        model,
        data,
        q_home,
        lookup_keys,
        motion_ref_list,
        left_geom,
        right_geom,
    ) = _setup(robot_name)

    failures: list[str] = []
    for vx in vx_bins:
        _matched_key, traj = _find_nominal_traj(lookup_keys, motion_ref_list, vx)
        for frame_idx in _PROBE_FRAMES:
            if frame_idx >= len(traj["qpos"]):
                continue
            _apply_frame(data, q_home, np.asarray(traj["qpos"][frame_idx], np.float64))
            left_z = _box_min_z(model, data, left_geom)
            right_z = _box_min_z(model, data, right_geom)

            for side, z, contact in (
                ("L", left_z, float(traj["contact"][frame_idx][0])),
                ("R", right_z, float(traj["contact"][frame_idx][1])),
            ):
                role = "stance" if contact > 0.5 else "swing"
                if role == "stance" and z > _STANCE_TOL_M:
                    failures.append(
                        f"{robot_name} vx={vx:.2f} frame={frame_idx} {side} "
                        f"stance z={z:+.4f} > {_STANCE_TOL_M}"
                    )
                if role == "swing" and z < _SWING_MIN_M:
                    failures.append(
                        f"{robot_name} vx={vx:.2f} frame={frame_idx} {side} "
                        f"swing z={z:+.4f} < {_SWING_MIN_M}"
                    )

    return failures


def test_geometry_gate_passes():
    """Shared in-scope geometry gate for ToddlerBot nominal walking."""
    failures: list[str] = []
    for robot_name in _ROBOT_VARIANTS:
        failures.extend(_collect_failures(robot_name, _VX_BINS_INSCOPE))

    assert not failures, (
        f"ToddlerBot walk geometry gate failed in {len(failures)} in-scope cases "
        f"(vx in {_VX_BINS_INSCOPE}). First few: " + "; ".join(failures[:5])
    )


def main() -> int:
    """CLI entry point with the full diagnostic matrix."""
    print("ToddlerBot walk geometry gate")
    print(f"  stance foot bottom z must be <= {_STANCE_TOL_M:.3f} m above floor")
    print(f"  swing  foot bottom z must be >  {_SWING_MIN_M:.3f} m above floor")

    all_failures: list[str] = []

    for robot_name in _ROBOT_VARIANTS:
        (
            model,
            data,
            q_home,
            lookup_keys,
            motion_ref_list,
            left_geom,
            right_geom,
        ) = _setup(robot_name)

        print()
        print(robot_name)
        print(
            f"{'vx':>5} {'frame':>5} {'Lc':>2} {'Rc':>2} "
            f"{'L_min_z':>9} {'R_min_z':>9} {'verdict':>7}"
        )
        print("-" * 52)

        robot_failures: list[str] = []
        for vx in _VX_BINS:
            matched_key, traj = _find_nominal_traj(lookup_keys, motion_ref_list, vx)
            worst_stance_z = -1e9
            worst_swing_z = 1e9

            for frame_idx in _PROBE_FRAMES:
                if frame_idx >= len(traj["qpos"]):
                    continue
                _apply_frame(
                    data, q_home, np.asarray(traj["qpos"][frame_idx], np.float64)
                )
                left_z = _box_min_z(model, data, left_geom)
                right_z = _box_min_z(model, data, right_geom)
                left_contact = float(traj["contact"][frame_idx][0])
                right_contact = float(traj["contact"][frame_idx][1])

                verdict = "OK"
                for side, z, contact in (
                    ("L", left_z, left_contact),
                    ("R", right_z, right_contact),
                ):
                    if contact > 0.5:
                        worst_stance_z = max(worst_stance_z, z)
                        if z > _STANCE_TOL_M:
                            verdict = "FAIL"
                            robot_failures.append(
                                f"{robot_name} vx={vx:.2f} frame={frame_idx} {side} "
                                f"stance z={z:+.4f} > {_STANCE_TOL_M}"
                            )
                    else:
                        worst_swing_z = min(worst_swing_z, z)
                        if z < _SWING_MIN_M:
                            verdict = "FAIL"
                            robot_failures.append(
                                f"{robot_name} vx={vx:.2f} frame={frame_idx} {side} "
                                f"swing z={z:+.4f} < {_SWING_MIN_M}"
                            )

                print(
                    f"{vx:5.2f} {frame_idx:5d} {int(left_contact):2d} {int(right_contact):2d} "
                    f"{left_z:+9.4f} {right_z:+9.4f} {verdict:>7}"
                )

            print(
                f"  summary req_vx={vx:.2f} matched_key={matched_key.tolist()} "
                f"worst_stance_z={worst_stance_z:+.4f} "
                f"worst_swing_z={worst_swing_z:+.4f}"
            )

        all_failures.extend(robot_failures)
        print(f"  failures: {len(robot_failures)}")
        for failure in robot_failures[:10]:
            print(f"    - {failure}")
        if len(robot_failures) > 10:
            print(f"    ... and {len(robot_failures) - 10} more")

    in_scope_failures = [
        failure
        for failure in all_failures
        if any(f"vx={vx:.2f}" in failure for vx in _VX_BINS_INSCOPE)
    ]
    print()
    print(
        f"In-scope failures (vx in {_VX_BINS_INSCOPE}): {len(in_scope_failures)} "
        f"- this is what the pytest gate asserts on."
    )

    if in_scope_failures:
        print(
            f"\nToddlerBot walk geometry gate (in-scope): FAIL "
            f"({len(in_scope_failures)} cases)"
        )
        return 1

    if all_failures:
        print(
            f"\nToddlerBot walk geometry gate (in-scope): PASS; "
            f"{len(all_failures)} deferred-bin failures noted above"
        )
        return 0

    print("\nToddlerBot walk geometry gate: PASS (full matrix)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
