"""Reproducibility helpers.

Each stress-test replicate carries a deterministic seed derived from a master
seed + the (species, axis, level, replicate_idx, algorithm) tuple. This means
results are reproducible without storing a separate seed per row, and cross-
replicate independence is preserved.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np


def derive_seed(master_seed: int, *components: str | int) -> int:
    """Derive a deterministic 32-bit seed from master seed + components.

    Example
    -------
    >>> derive_seed(20260416, "Austropotamobius_torrentium", "snapping", 20, 7, "RF")
    1845720133   # deterministic
    """
    key = f"{master_seed}::" + "::".join(str(c) for c in components)
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def rng(master_seed: int, *components: str | int) -> np.random.Generator:
    """Return a numpy Generator with a derived seed."""
    return np.random.default_rng(derive_seed(master_seed, *components))


def get_git_commit(repo_root: Path | str | None = None) -> str | None:
    """Return the current git commit SHA, or None if not a git repo."""
    cwd = Path(repo_root) if repo_root else None
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_dirty(repo_root: Path | str | None = None) -> bool:
    """Return True if the working tree has uncommitted changes."""
    cwd = Path(repo_root) if repo_root else None
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
