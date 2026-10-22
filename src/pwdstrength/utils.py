"""Utility functions for seeding, configuration, and logging."""

import os
import random
import logging
from typing import Any, Dict


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across random, numpy, and PyTorch for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _simple_yaml_parse(text: str) -> Dict[str, Any]:
    """Fallback recursive-free lightweight YAML parser for key-value and nested dicts."""
    result: Dict[str, Any] = {}
    current_section = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.strip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        if ":" in stripped:
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()

            if indent == 0:
                if not val:
                    result[key] = {}
                    current_section = key
                else:
                    result[key] = _parse_val(val)
                    current_section = None
            elif indent >= 2 and current_section:
                result[current_section][key] = _parse_val(val)

    return result


def _parse_val(val: str) -> Any:
    """Parse scalar or array YAML string."""
    if not val:
        return None
    if val.startswith("[") and val.endswith("]"):
        items = val[1:-1].split(",")
        return [_parse_val(item.strip()) for item in items if item.strip()]
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() == "none" or val.lower() == "null":
        return None
    try:
        if "." in val or "e" in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val.strip("\"'")


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file, falling back to built-in parser if PyYAML is unavailable."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        import yaml
        return yaml.safe_load(content)
    except ImportError:
        return _simple_yaml_parse(content)


def setup_logger(name: str = "pwdstrength", level: str = "INFO") -> logging.Logger:
    """Configure and return a standardized console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    return logger
