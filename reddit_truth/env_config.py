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
