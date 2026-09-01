"""Entrypoint. Run with ``python -m centauro_lite`` or ``poetry run poe eda``."""

from __future__ import annotations

import sys

from loguru import logger

from centauro_lite.cli import app
from centauro_lite.services.notifier import send_alert


def main() -> int:
    """Run the CLI and translate failure into a non-zero exit code.

    Returns:
        ``0`` on success, ``1`` on failure. A long unattended run can only be
        supervised through its exit code, so swallowing the exception here would
        hide the failure from whatever launched it.
    """
    try:
        app()
    except SystemExit as exc:  # Typer's own exit path, already reported to the user.
        return int(exc.code or 0)
    # Broad on purpose: the entrypoint is the last line of defense, and an unattended
    # run must convert any failure into an alert plus a non-zero exit code.
    except Exception as exc:
        logger.exception("Run failed")
        send_alert(f"*centauro-lite failed*\n```\n{exc}\n```")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
