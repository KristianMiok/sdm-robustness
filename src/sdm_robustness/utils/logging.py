"""Logging setup — loguru with file + console handlers."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_dir: Path | str | None = None, level: str = "INFO") -> None:
    """Configure loguru for a run.

    Parameters
    ----------
    log_dir : path, optional
        Directory to write run.log. If None, only console logging is configured.
    level : str
        Minimum log level (DEBUG, INFO, WARNING, ERROR).
    """
    logger.remove()  # reset defaults
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    )
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "run.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {name}:{function}:{line} | {message}",
            rotation="50 MB",
            retention=10,
        )


__all__ = ["logger", "setup_logging"]
