"""Application configuration.

All runtime settings are sourced from environment variables (optionally via a
local ``.env`` file). No secrets are hardcoded here. Every field added must also
be documented in ``.env.example``.
"""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings, populated from the environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Human-readable application name, surfaced in the /health payload and logs.
    APP_NAME: str = "HireLens"

    # Deployment environment: "dev" or "prod". Drives environment-specific behavior.
    APP_ENV: str = "dev"

    # URL prefix for all versioned v1 API routes.
    API_V1_PREFIX: str = "/api/v1"

    # CORS allow-list. Comma-separated in the env, parsed into a list here.
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ]

    # Root logging level: DEBUG / INFO / WARNING / ERROR / CRITICAL.
    LOG_LEVEL: str = "INFO"

    # Database connection string. Defaults to a local SQLite file for zero-setup dev.
    DATABASE_URL: str = "sqlite:///./hirelens.db"

    # Freemium scan cap for the anonymous seeker /score path. 0 (or negative) means
    # UNLIMITED — the default: there is no scan limit. Set a positive integer to
    # re-enable a freemium tier without code changes.
    FREEMIUM_SCAN_LIMIT: int = 0

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Allow ALLOWED_ORIGINS to be provided as a comma-separated string."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


settings = Settings()
