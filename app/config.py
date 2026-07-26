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

    anthropic_api_key: str | None = None
    claude_model: str = "claude-sonnet-5"

    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    transactions_path: Path = data_dir / "transactions_sample.csv"
    accounts_path: Path = data_dir / "accounts_sample.csv"

    def require_api_key(self) -> str:
        """Fetch the API key, raising a clear error only at the point of use.

        Data loading, tools, and tests that never call the LLM should not be
        able to crash just because .env is missing — only the orchestrator,
        which actually needs the key, calls this.
        """
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
                "add your key before running the orchestrator."
            )
        return self.anthropic_api_key


settings = Settings()
