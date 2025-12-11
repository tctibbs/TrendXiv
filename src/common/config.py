"""Configuration management using Pydantic settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="TRENDXIV_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    database_path: str = "data/trendxiv.db"
    arxiv_delay_seconds: float = 3.0
    arxiv_page_size: int = 500
    arxiv_max_results: int = 10000
    cache_ttl_hours: int = 24
    default_lookback_years: int = 5

    @property
    def database_file(self) -> Path:
        """Return database path as Path object."""
        return Path(self.database_path)


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
