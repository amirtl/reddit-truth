import pytest
from pipeline.config import load_config, AppConfig


def test_load_config_returns_typed_object(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("""
llms:
  product_understanding: "ollama/llama3.2"
  aspect_extraction: "gemini/gemini-2.0-flash"
  summarization: "ollama/llama3.2"
embeddings:
  provider: "local"
  model: "all-MiniLM-L6-v2"
""")
    config = load_config(str(config_file))
    assert isinstance(config, AppConfig)
    assert config.llms.product_understanding == "ollama/llama3.2"
    assert config.llms.aspect_extraction == "gemini/gemini-2.0-flash"
    assert config.embeddings.model == "all-MiniLM-L6-v2"


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yml")


def test_load_config_all_llm_fields():
    import tempfile, os
    content = """
llms:
  product_understanding: "openai/gpt-4o"
  aspect_extraction: "openai/gpt-4o"
  summarization: "openai/gpt-4o"
embeddings:
  provider: "local"
  model: "all-MiniLM-L6-v2"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        config = load_config(path)
        assert config.llms.summarization == "openai/gpt-4o"
    finally:
        os.unlink(path)
