"""Application settings, validated at import time.

Reading configuration through pydantic means a missing or malformed variable fails
loudly at boot instead of surfacing as a confusing ``None`` deep inside a pipeline run.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = "INFO"
    log_dir: Path = Path("logs")
    log_serialize: bool = False

    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None

    @property
    def telegram_enabled(self) -> bool:
        """Whether both Telegram credentials are present."""
        return self.telegram_bot_token is not None and self.telegram_chat_id is not None


settings = Settings()
