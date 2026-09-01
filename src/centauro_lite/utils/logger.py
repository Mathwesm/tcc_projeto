"""Centralized loguru configuration.

Importing this module is not enough; call :func:`setup_logging` once from the
entrypoint. Structured JSON output matters for unattended runs, where the log file
is the only evidence available when something fails at three in the morning.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


def setup_logging(
    *,
    log_dir: Path,
    level: str = "INFO",
    serialize: bool = False,
) -> None:
    """Configure the console and file sinks.

    Args:
        log_dir: Directory for rotated log files. Created if missing.
        level: Minimum level shown on the console. The file sink always keeps DEBUG,
            so a failure can be investigated after the fact without reproducing it.
        serialize: Emit the file sink as JSON. Enable for scheduled runs, where logs
            are read by tooling rather than by a human scrolling a terminal.
    """
    logger.remove()
    logger.add(sys.stderr, format=_CONSOLE_FORMAT, level=level, colorize=True)

    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "centauro_lite.log",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        serialize=serialize,
        enqueue=True,
        backtrace=True,
        diagnose=False,  # keep variable values out of logs: they leak credentials
    )
