"""YAML configuration loading with beginner-friendly validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a configuration file cannot be understood."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from *path*.

    Keeping configuration as a plain mapping makes each command's required
    settings visible at the point where they are used.
    """

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {config_path}")
    return raw


def config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one mapping-valued configuration section."""

    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration section '{name}' must be a mapping")
    return value
