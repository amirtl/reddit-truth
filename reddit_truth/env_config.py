import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Django
    secret_key: str = "dev-insecure-key-change-in-production"
    debug: bool = False

    # Database (Supabase PostgreSQL)
    db_password: str = ""
    db_host: str = "localhost"
    db_port: int = 5432

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Reddit API
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "reddit-truth/0.1"

    # LLM API keys (only needed for non-Ollama providers)
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""


env = EnvConfig()


def _export_llm_keys(config: EnvConfig) -> None:
    """Bridge LLM keys into os.environ so LiteLLM (called from the pure pipeline)
    can authenticate. Pydantic loads them onto `env`, but LiteLLM reads the
    provider-conventional environment variables directly. Only set non-empty keys
    so we never clobber a real env var with a blank default.
    """
    for var, value in (
        ("GEMINI_API_KEY", config.gemini_api_key),
        ("OPENAI_API_KEY", config.openai_api_key),
        ("ANTHROPIC_API_KEY", config.anthropic_api_key),
    ):
        if value:
            os.environ.setdefault(var, value)


_export_llm_keys(env)
