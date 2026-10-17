"""Startup regression checks for the isolated main comparison runtime."""

import sys

import pytest

from toddlerbot.locomotion import main_environment


def test_rejects_extra_cuda_plugin(tmp_path, monkeypatch):
    lock = tmp_path / "requirements-main-compare.lock"
    lock.write_text("jax==0.6.2\njaxlib==0.6.2\njax-cuda12-plugin==0.6.2\n")
    monkeypatch.setattr(main_environment, "dependency_lock", lambda: lock)
    monkeypatch.setattr(main_environment, "installed_packages", lambda: {
        "jax": "0.6.2", "jaxlib": "0.6.2", "jax-cuda12-plugin": "0.6.2",
        "jax-cuda13-plugin": "0.8.0",
    })
    monkeypatch.setattr(main_environment.metadata, "distributions", lambda: [])
    with pytest.raises(RuntimeError, match="Unexpected CUDA package jax-cuda13-plugin") as error:
        main_environment.check_dependencies()
    assert "UV_PROJECT_ENVIRONMENT=.venv-main-compare" in str(error.value)
    assert sys.executable in str(error.value)


def test_comparison_rejects_environment_before_robot_initialization(monkeypatch):
    import gin
    from toddlerbot.locomotion import train_mjx

    def reject_runtime(**kwargs):
        raise RuntimeError("Comparison environment is incompatible")

    def unexpected_robot(*args, **kwargs):
        pytest.fail("Robot initialized before the comparison environment check")

    monkeypatch.setattr(main_environment, "verify_runtime", reject_runtime)
    monkeypatch.setattr(train_mjx, "Robot", unexpected_robot)
    monkeypatch.setattr(sys, "argv", [
        "train_mjx", "--brax", "--gin-file", "main_compare", "--skip-video",
    ])
    gin.clear_config()
    try:
        with pytest.raises(RuntimeError, match="Comparison environment is incompatible"):
            train_mjx.main()
    finally:
        gin.clear_config()
