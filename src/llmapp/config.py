"""Typed application configuration.

Configuration is data: it is read once at startup,
validated, and never mutated. Nothing else in the
application reads os.environ directly.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "ci", "production"]
Provider = Literal["fake", "openai", "anthropic"]


class Settings(BaseSettings):
    """Every value the application needs to run."""

    model_config = SettingsConfigDict(
        env_prefix="LLMAPP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    environment: Environment = "local"
    provider: Provider = "fake"

    # Model identifiers live in .env, never in code.
    model: str = ""
    api_key: SecretStr | None = None

    request_timeout_s: float = Field(default=30.0, gt=0)
    max_output_tokens: int = Field(default=512, gt=0)
    daily_cost_limit_usd: float = Field(default=5.0, ge=0)

    # Point at any OpenAI-compatible endpoint; None uses
    # the provider's default host.
    base_url: str | None = None

    # Serving
    port: int = Field(default=8000, gt=0, le=65535)
    rate_limit_per_minute: int = Field(default=60, gt=0)
    jwt_secret: SecretStr | None = None

    # Retrieval
    embedding_model: str = ""
    embedding_dimensions: int = Field(default=1536, gt=0)

    corpus_dir: Path = Path("corpus")
    data_dir: Path = Path("data")

    @model_validator(mode="after")
    def check_provider_requirements(self) -> "Settings":
        if self.provider == "fake":
            return self
        if not self.api_key:
            raise ValueError(
                f"provider={self.provider} requires "
                "LLMAPP_API_KEY"
            )
        if not self.model:
            raise ValueError(
                f"provider={self.provider} requires LLMAPP_MODEL"
            )
        return self

    @model_validator(mode="after")
    def check_production_rules(self) -> "Settings":
        if self.environment != "production":
            return self
        if self.provider == "fake":
            raise ValueError(
                "the fake provider must not run in production"
            )
        if not self.jwt_secret:
            raise ValueError(
                "production requires LLMAPP_JWT_SECRET"
            )
        if len(self.jwt_secret.get_secret_value()) < 32:
            raise ValueError(
                "LLMAPP_JWT_SECRET must be at least 32 bytes"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process."""
    return Settings()
