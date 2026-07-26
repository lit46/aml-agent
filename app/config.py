"""Centralized application configuration.

All environment-dependent values (API keys, model names, data paths) are
loaded here once and imported elsewhere via the `settings` singleton.
Avoid calling os.getenv() directly anywhere else in the codebase.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, populated from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-6"

    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    transactions_path: Path = data_dir / "transactions_sample.csv"
    accounts_path: Path = data_dir / "accounts_sample.csv"


settings = Settings()
