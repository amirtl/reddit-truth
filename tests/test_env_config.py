import os
from reddit_truth.env_config import EnvConfig, _export_llm_keys


def test_exports_nonempty_key_to_environ(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg = EnvConfig(gemini_api_key="test-key", openai_api_key="", anthropic_api_key="")
    _export_llm_keys(cfg)
    assert os.environ["GEMINI_API_KEY"] == "test-key"


def test_does_not_export_empty_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = EnvConfig(gemini_api_key="x", openai_api_key="", anthropic_api_key="")
    _export_llm_keys(cfg)
    assert "OPENAI_API_KEY" not in os.environ


def test_does_not_clobber_existing_environ(monkeypatch):
    # A real shell-exported key must win over a stale .env default.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "real-shell-value")
    cfg = EnvConfig(anthropic_api_key="from-dotenv")
    _export_llm_keys(cfg)
    assert os.environ["ANTHROPIC_API_KEY"] == "real-shell-value"
