from pathlib import Path
from pydantic import BaseModel
import yaml


class LLMConfig(BaseModel):
    product_understanding: str
    aspect_extraction: str
    summarization: str


class EmbeddingConfig(BaseModel):
    provider: str
    model: str


class AppConfig(BaseModel):
    llms: LLMConfig
    embeddings: EmbeddingConfig


def load_config(path: str = "config.yml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)
