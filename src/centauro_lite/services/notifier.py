"""Failure alerting through the Telegram Bot API.

An unattended job that fails silently is worse than one that never ran, because
nobody goes looking for it. This module is the escape hatch that makes failure loud.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from centauro_lite.config import settings

_API_URL = "https://api.telegram.org/bot{token}/{method}"
_TIMEOUT = 10.0


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def send_alert(message: str, *, attachment: Path | None = None) -> None:
    """Push an alert to the configured Telegram chat.

    Silently returns when credentials are absent, so local development does not
    fail merely because the developer has no bot configured.

    Args:
        message: Text body. Markdown is supported.
        attachment: Optional file sent alongside, such as a log excerpt or a chart.

    Raises:
        httpx.HTTPStatusError: If the Telegram API rejects the request.
    """
    if not settings.telegram_enabled:
        logger.warning("Telegram credentials missing, alert not sent | message={}", message)
        return

    token = settings.telegram_bot_token.get_secret_value()  # type: ignore[union-attr]
    payload: dict[str, Any] = {
        "chat_id": settings.telegram_chat_id,
        "parse_mode": "Markdown",
    }

    if attachment is None:
        payload["text"] = message
        url = _API_URL.format(token=token, method="sendMessage")
        response = httpx.post(url, data=payload, timeout=_TIMEOUT)
    else:
        payload["caption"] = message
        url = _API_URL.format(token=token, method="sendDocument")
        with attachment.open("rb") as handle:
            response = httpx.post(
                url,
                data=payload,
                files={"document": handle},
                timeout=_TIMEOUT,
            )

    response.raise_for_status()
    logger.info("Alert delivered to Telegram | attachment={}", attachment)
