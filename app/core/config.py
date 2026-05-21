"""Central configuration via Pydantic-Settings — reads from .env / environment."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, AnyUrl, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field("gpt-4o", description="Vision-language model name")
    openai_max_tokens: int = Field(4096, ge=256, le=16384)
    openai_temperature: float = Field(0.2, ge=0.0, le=2.0)

    # ── Database ──────────────────────────────────────────────────────────────
    postgres_user: str = "tempovis"
    postgres_password: str = "changeme"
    postgres_db: str = "tempovis"
    postgres_host: str = "localhost"
    postgres_port: int = Field(5432, ge=1, le=65535)
    # Optional full DSN — overrides component vars when provided
    database_url_override: str | None = Field(None, alias="DATABASE_URL")

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field("redis://localhost:6379/0", description="Redis connection URL")

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = Field(8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent_max_iterations: int = Field(10, ge=1, le=50)
    agent_timeout_seconds: int = Field(120, ge=10, le=600)

    # ── Renderer ──────────────────────────────────────────────────────────────
    plot_dpi: int = Field(150, ge=72, le=300)
    plot_width_in: float = Field(12.0, ge=4.0, le=24.0)
    plot_height_in: float = Field(5.0, ge=2.0, le=12.0)

    # ── Storage (local | minio | s3) ─────────────────────────────────────────
    storage_backend: Literal["local", "minio", "s3"] = "local"
    local_plots_dir: str = "data/plots"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "tempovis-plots"
    minio_secure: bool = False
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_s3_bucket: str = "tempovis-plots"
    aws_region: str = "us-east-1"

    # ── Celery ────────────────────────────────────────────────────────────────
    celery_broker_url: str = Field(
        "redis://localhost:6379/1", description="Celery broker (Redis)"
    )
    celery_result_backend: str = Field(
        "redis://localhost:6379/2", description="Celery result backend (Redis)"
    )

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_limit_analyze: str = Field(
        "10/minute", description="slowapi rate limit string for /analyze"
    )

    # ── DRY_RUN / testing ─────────────────────────────────────────────────────
    dry_run: bool = Field(
        False,
        description="Skip real OpenAI calls; return a hardcoded mock response",
    )
    max_calls_per_day: int = Field(
        20,
        ge=1,
        le=10_000,
        description="Max OpenAI API calls per UTC day; enforced via Redis counter",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @model_validator(mode="after")
    def _validate_production_settings(self) -> "Settings":
        if self.app_env == "production" and self.postgres_password == "changeme":
            raise ValueError("Default postgres_password must not be used in production")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()  # type: ignore[call-arg]
