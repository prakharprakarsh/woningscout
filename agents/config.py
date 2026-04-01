"""
Centralized configuration — loaded from .env or environment variables.
Uses pydantic-settings so we get type validation for free.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application-wide configuration.

    All values can be overridden via environment variables or .env file.
    Blank FUNDA_API_KEY triggers demo mode (uses fixture data).
    """

    # Funda
    funda_api_key: str = ""
    funda_rate_limit_rps: float = 2.0  # their ToS limit

    # Database
    database_url: str = "postgresql://scout:scout_dev_pass@localhost:5432/woningscout"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Model storage
    model_path: str = "./models/artifacts"

    # Pipeline
    scan_interval: int = 300  # seconds between scans
    target_regions: str = "amsterdam"  # comma-separated

    # Alerting — blank means log-only
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_pass: str = ""
    telegram_bot_token: str = ""

    # Monitoring
    prometheus_port: int = 9090

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # json or console

    @property
    def is_demo_mode(self) -> bool:
        """Demo mode = no Funda key, use fixtures instead."""
        return not self.funda_api_key

    @property
    def regions_list(self) -> list[str]:
        return [r.strip() for r in self.target_regions.split(",") if r.strip()]

    @property
    def model_dir(self) -> Path:
        p = Path(self.model_path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton — import this everywhere
settings = Settings()
