"""Verify the pinned main comparison runtime and record its source provenance."""

import argparse
import hashlib
import importlib.metadata as metadata
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


MAIN_REVISION = "f81679b72cca2cb227e190beeea18eed7894c10c"
ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def installed_packages():
    return {canonicalize_name(dist.metadata["Name"]): dist.version
            for dist in metadata.distributions()}


def dependency_lock():
    name = ("requirements-main-compare.lock" if sys.platform == "linux"
            else "requirements-main-compare-cpu.lock")
    return ROOT / name


def check_dependencies():
    versions = installed_packages()
    lock = dependency_lock()
    errors = []
    checked = 0
    expected_cuda_packages = set()
    for line in lock.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        req = Requirement(line)
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        name = canonicalize_name(req.name)
        if name.startswith("jax-cuda"):
            expected_cuda_packages.add(name)
        version = versions.get(name)
        if version is None or version not in req.specifier:
            errors.append(f"{req}: installed {version or 'missing'}")
        checked += 1
    for name, version in versions.items():
        if name.startswith("jax-cuda") and name not in expected_cuda_packages:
            errors.append(f"Unexpected CUDA package {name}=={version} for {lock.name}")
    for dist in metadata.distributions():
        for text in dist.requires or ():
            req = Requirement(text)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            version = versions.get(canonicalize_name(req.name))
            if version is None or version not in req.specifier:
                errors.append(f"{dist.metadata['Name']} requires {req}: {version}")
    if errors:
        raise RuntimeError(
            f"Dependency check failed for {sys.executable}.\n"
            f"Sync {lock.name} into .venv-main-compare, then set "
            "UV_PROJECT_ENVIRONMENT=.venv-main-compare UV_PYTHON=3.11.\n"
            + "\n".join(errors)
        )
    print(f"Verified {checked} active pins in {lock.name}.")


def verify_runtime(require_gpu=True):
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"The main comparison lock requires Python 3.11: {sys.version}")
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"Run from the comparison worktree: {ROOT}")
    subprocess.run(["git", "merge-base", "--is-ancestor", MAIN_REVISION, "HEAD"],
                   cwd=ROOT, check=True)
    check_dependencies()
    # Import the native trainer first, so the preflight uses its XLA settings.
    from toddlerbot.locomotion import train_mjx
    import brax
    import jax
    import jax.numpy as jnp
    import numpy as np
    from brax.training.agents.ppo import networks, train

    if Path(train_mjx.__file__).resolve() != ROOT / "toddlerbot/locomotion/train_mjx.py":
        raise RuntimeError(f"Wrong trainer source: {train_mjx.__file__}")
    if not Path(brax.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Brax imported outside the selected environment: {brax.__file__}")
    for function, required in (
        (train.train, {"run_evals", "restore_params", "training_metrics_steps"}),
        (networks.make_ppo_networks, {"value_obs_key", "distribution_type", "state_dependent_std"}),
    ):
        missing = required - set(inspect.signature(function).parameters)
        if missing:
            raise RuntimeError(f"Unsupported Brax API: {function.__name__}: {missing}")
    print("ToddlerBot HEAD:", subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    print("Trainer:", train_mjx.__file__)
    print("Brax:", brax.__version__, brax.__file__)
    print("Python:", sys.executable)
    print("JAX:", jax.__version__, jax.devices())
    if require_gpu and (sys.platform != "linux" or jax.default_backend() != "gpu"):
        raise RuntimeError("Long comparison training requires a verified Linux GPU.")
    x = jnp.arange(64, dtype=jnp.float32).reshape(8, 8)
    actual = jax.jit(lambda a: a @ a.T)(x).block_until_ready()
    expected = np.asarray(x) @ np.asarray(x).T
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)
    print(f"{jax.default_backend().upper()} JIT compilation passed.")


def assert_finite(tree, label):
    import jax
    import numpy as np
    for leaf in jax.tree.leaves(jax.device_get(tree)):
        value = np.asarray(leaf)
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite {label}; stopping before checkpoint promotion.")


def write_provenance(folder, env):
    import brax
    import jax
    import numpy as np

    folder = Path(folder)
    paths = list((ROOT / "toddlerbot/locomotion").glob("*.py"))
    paths += list((ROOT / "toddlerbot/locomotion").glob("*.gin"))
    for directory in ("reference", "utils", "sim/terrain"):
        paths += list((ROOT / "toddlerbot" / directory).glob("*.py"))
    paths += [ROOT / "toddlerbot/sim/robot.py", ROOT / "toddlerbot/sim/motor_control.py"]
    paths += list(ROOT.glob("requirements-main-compare*"))
    source_hashes = {}
    for path in sorted(set(paths)):
        relative = path.relative_to(ROOT)
        target = folder / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[str(relative)] = sha256(path)
    model_paths = list((ROOT / "toddlerbot/descriptions" / env.robot.name).glob("*.xml"))
    model_paths += list((ROOT / "toddlerbot/descriptions" / env.robot.name).glob("*.yml"))
    model_paths += [ROOT / "motion/walk_zmp_2xc.lz4", ROOT / "motion/arm_pose_dataset.lz4"]
    provenance = {
        "main_base": MAIN_REVISION,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "python": sys.version,
        "executable": sys.executable,
        "packages": installed_packages(),
        "lock": {"name": dependency_lock().name, "sha256": sha256(dependency_lock())},
        "brax_source": brax.__file__,
        "devices": [{"kind": d.device_kind, "platform": d.platform,
                     "compute_capability": str(getattr(d, "compute_capability", "n/a"))}
                    for d in jax.devices()],
        "source_sha256": source_hashes,
        "model_reference_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in model_paths},
        "robot_config": env.robot.config,
        "control_period_s": float(env.dt),
        "physics_timestep_s": float(env.sys.opt.timestep),
        "physics_solver": int(env.sys.opt.solver),
        "physics_iterations": int(env.sys.opt.iterations),
        "physics_ls_iterations": int(env.sys.opt.ls_iterations),
        "nominal_torso_height_m": float(env.motion_ref.default_qpos[2]),
        "ppo": {"backend": "native_brax", "state_dependent_std": False,
                "run_evals": False, "bootstrap_on_timeout": False,
                "learning_rate_schedule": "none"},
    }
    # Measure COM in the actual assembled model, at its home pose.
    import mujoco
    data = mujoco.MjData(env.sys.mj_model)
    data.qpos[:] = env.sys.mj_model.keyframe("home").qpos
    mujoco.mj_forward(env.sys.mj_model, data)
    provenance["nominal_com_height_m"] = float(data.subtree_com[0, 2])
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Check the Mac CPU projection only.")
    args = parser.parse_args()
    verify_runtime(require_gpu=not args.cpu)


if __name__ == "__main__":
    main()
