"""Configuration loading.

Resolution order for every config key:
1. Environment variable (e.g., SDM_RAW_DATA_PATH)
2. configs/paths.local.yaml if present (gitignored, per-user)
3. configs/paths.yaml (committed default)

For non-path configs (frozen_design.yaml, task1_gates.yaml), only file-based
resolution is used.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

# Project root: parent of src/
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _PROJECT_ROOT / "configs"


def project_root() -> Path:
    """Return the project root directory."""
    return _PROJECT_ROOT


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file into a dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_paths() -> dict[str, Any]:
    """Load paths config with local override and env var support."""
    base = load_yaml(_CONFIG_DIR / "paths.yaml")
    local_file = _CONFIG_DIR / "paths.local.yaml"
    if local_file.exists():
        base.update(load_yaml(local_file))

    # Env var overrides
    env_map = {
        "SDM_RAW_DATA_PATH": "raw_data_path",
    }
    for env_key, cfg_key in env_map.items():
        if env_key in os.environ:
            base[cfg_key] = os.environ[env_key]
    return base


def load_frozen_design() -> dict[str, Any]:
    """Load the frozen design decisions."""
    return load_yaml(_CONFIG_DIR / "frozen_design.yaml")


def load_task1_gates() -> dict[str, Any]:
    """Load Task 1 gate thresholds."""
    return load_yaml(_CONFIG_DIR / "task1_gates.yaml")


def config_hash(cfg: dict[str, Any]) -> str:
    """Deterministic hash of a config dict — logged for reproducibility."""
    serialised = yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(serialised.encode()).hexdigest()[:12]


def resolve_path(path_str: str) -> Path:
    """Resolve a path from config — absolute as-is, relative to project root."""
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p
