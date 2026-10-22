"""Smoke test verifying package import, version, config loading, and seeding."""

import os
import pwdstrength
from pwdstrength.utils import load_config, set_seed, setup_logger


def test_package_import():
    """Verify package import and metadata."""
    assert pwdstrength.__version__ == "0.1.0"
    assert hasattr(pwdstrength, "load_config")
    assert hasattr(pwdstrength, "set_seed")
    assert hasattr(pwdstrength, "setup_logger")


def test_config_loader():
    """Verify configs/default.yaml loads properly."""
    cfg = load_config("configs/default.yaml")
    assert "paths" in cfg
    assert "model" in cfg
    assert "data" in cfg
    assert cfg["data"]["max_len"] == 32
    assert cfg["data"]["class_thresholds"] == [3.0, 6.0, 8.0, 10.0]
    assert cfg["model"]["lstm_layers"] == 2
    assert cfg["model"]["lstm_units"] == 128


def test_seeding():
    """Verify reproducible seeding."""
    import random
    set_seed(1234)
    r1 = random.random()
    set_seed(1234)
    r2 = random.random()
    assert r1 == r2


def test_logger():
    """Verify logger setup."""
    logger = setup_logger("test_logger")
    assert logger.name == "test_logger"
