"""Check the paper runtime, including the explicit Blackwell overrides."""

import argparse
import importlib.metadata as metadata
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


BRAX_REVISION = "14147806f6c25724ba830ba36764df3243889b0d"
PREPARED_REVISION = "586d4dd65bed4d5e8066a98b6210b2b3a85f7144"
CORE_PACKAGES = {
    "jax", "jaxlib", "mujoco", "mujoco-mjx", "flax", "optax",
    "chex", "orbax-checkpoint", "numpy", "scipy",
}
BLACKWELL_OVERRIDES = {"jax", "jaxlib", "orbax-checkpoint"}


def locked_requirements(path):
    """Read the exact package pins, excluding the editable Brax source."""
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-e ")):
            yield Requirement(line)


def installed_packages():
    return {
        canonicalize_name(dist.metadata["Name"]): dist.version
        for dist in metadata.distributions()
    }


def matching_dependency_lock():
    """Select a known lock by the installed core versions, on CPU or GPU."""
    versions = installed_packages()
    for name in ("requirements-paper.lock", "requirements-paper-blackwell.lock"):
        path = Path(name)
        if not path.exists():
            continue
        core = [req for req in locked_requirements(path)
                if canonicalize_name(req.name) in CORE_PACKAGES]
        if {canonicalize_name(req.name) for req in core} != CORE_PACKAGES:
            continue
        if all(versions.get(canonicalize_name(req.name)) is not None
               and versions[canonicalize_name(req.name)] in req.specifier
               for req in core):
            return path
    return None


def check_dependencies(lock, blackwell=False):
    versions = installed_packages()
    errors = []
    if versions.get("brax") != "0.10.5":
        errors.append(f"brax==0.10.5: installed {versions.get('brax', 'missing')}")
    checked = 0
    for req in locked_requirements(lock):
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        name = canonicalize_name(req.name)
        version = versions.get(name)
        if version is None or version not in req.specifier:
            errors.append(f"{req}: installed {version or 'missing'}")
        checked += 1

    overridden = []
    for dist in metadata.distributions():
        parent = canonicalize_name(dist.metadata["Name"])
        for text in dist.requires or ():
            req = Requirement(text)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            name = canonicalize_name(req.name)
            version = versions.get(name)
            if version is not None and version in req.specifier:
                continue
            if blackwell and parent == "brax" and name in BLACKWELL_OVERRIDES:
                overridden.append(f"brax: {req} -> {version}")
            else:
                errors.append(f"{parent} requires {req}: installed {version or 'missing'}")
    if errors:
        raise RuntimeError("Dependency check failed:\n" + "\n".join(errors))
    print(f"Verified {checked} active lock pins.")
    for text in overridden:
        print(f"Explicit Blackwell override: {text}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blackwell", action="store_true",
                        help="Check the Linux Python 3.10 Blackwell lock.")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"Python 3.10 is required, got {sys.version}.")
    if args.blackwell and sys.platform != "linux":
        raise RuntimeError("The Blackwell GPU lock targets Linux Python 3.10.")
    root = Path(__file__).resolve().parents[2]
    if Path.cwd().resolve() != root:
        raise RuntimeError(f"Run this check from {root}.")
    lock = root / ("requirements-paper-blackwell.lock" if args.blackwell
                   else "requirements-paper.lock")
    check_dependencies(lock, args.blackwell)
    subprocess.run(["git", "merge-base", "--is-ancestor", PREPARED_REVISION, "HEAD"],
                   check=True)

    revision = subprocess.check_output(
        ["git", "-C", "toddlerbot/brax", "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != BRAX_REVISION:
        raise RuntimeError(f"Brax must be at {BRAX_REVISION}, got {revision}.")
    if subprocess.check_output(
        ["git", "-C", "toddlerbot/brax", "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("The pinned Brax source has local changes.")
    print("ToddlerBot HEAD:", subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip())

    # Import the native trainer before JAX to use the same XLA settings.
    from toddlerbot.locomotion import train_mjx
    import brax
    import jax
    import jax.numpy as jnp
    import numpy as np

    if not Path(brax.__file__).resolve().is_relative_to(root / "toddlerbot/brax"):
        raise RuntimeError(f"Brax imported outside this checkout: {brax.__file__}")
    print("Trainer:", train_mjx.__file__)
    print("Brax:", brax.__file__, revision)
    print("Python:", sys.executable)
    print("JAX:", jax.__version__)
    for device in jax.devices():
        print("Device:", device, device.device_kind,
              "compute capability", getattr(device, "compute_capability", "n/a"))
    if jax.default_backend() != "gpu":
        raise RuntimeError("A GPU backend is required for the long training run.")
    x = jnp.arange(64, dtype=jnp.float32).reshape(8, 8)
    actual = jax.jit(lambda a: a @ a.T)(x).block_until_ready()
    reference = np.asarray(x) @ np.asarray(x).T
    np.testing.assert_allclose(np.asarray(actual), reference, rtol=1e-5, atol=1e-5)
    print("GPU JIT compilation passed. Run the native PPO smoke check next.")


if __name__ == "__main__":
    main()
