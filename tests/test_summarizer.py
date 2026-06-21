import json
import pytest
from unittest.mock import MagicMock
from pipeline.config import AppConfig, LLMConfig, EmbeddingConfig
from pipeline.summarizer import Summarizer
from pipeline.types import Cluster, QuantifiedAspect, AspectClaim, AspectSummary


@pytest.fixture
def config():
    return AppConfig(
        llms=LLMConfig(
            product_understanding="ollama/llama3.2",
            aspect_extraction="gemini/gemini-2.0-flash",
            summarization="ollama/llama3.2",
        ),
        embeddings=EmbeddingConfig(provider="local", model="all-MiniLM-L6-v2"),
    )


def mock_llm_response(mocker, content: dict):
    mock = mocker.patch("pipeline.summarizer.litellm.completion")
    mock.return_value.choices[0].message.content = json.dumps(content)
    return mock


# ── robustness to imperfect LLM JSON ──────────────────────────────────────────

def test_missing_headline_falls_back_to_aspect_label(config, mocker):
    # real Ollama sometimes omits keys entirely — must not crash
    mock_llm_response(mocker, {"detail": "Users report degradation."})
    aspects = [QuantifiedAspect("battery life", 87.0, 71.0, 29.0, "declining")]
    clusters = [Cluster("battery life", [], 0, 0)]

    result = Summarizer(config).run(aspects, clusters)

    assert result[0].headline == "battery life"
    assert result[0].detail == "Users report degradation."


def test_detail_returned_as_list_is_joined(config, mocker):
    # the LLM occasionally returns detail as a JSON array of sentences
    mock_llm_response(mocker, {
        "headline": "Mixed sound",
        "detail": ["Bass is deep.", "Mics are weak."],
    })
    aspects = [QuantifiedAspect("sound", 50.0, 50.0, 50.0, "stable")]
    clusters = [Cluster("sound", [], 0, 0)]

    result = Summarizer(config).run(aspects, clusters)

    assert result[0].detail == "Bass is deep. Mics are weak."


def test_missing_detail_and_trend_note_default_to_empty(config, mocker):
    mock_llm_response(mocker, {"headline": "Great"})
    aspects = [QuantifiedAspect("comfort", 30.0, 90.0, 10.0, "stable")]
    clusters = [Cluster("comfort", [], 0, 0)]

    result = Summarizer(config).run(aspects, clusters)

    assert result[0].detail == ""
    assert result[0].trend_note == ""


# ── structural tests ──────────────────────────────────────────────────────────

def test_returns_list_of_aspect_summaries(config, mocker):
    mock_llm_response(mocker, {"headline": "Great battery", "detail": "Users love it.", "trend_note": ""})
    aspects = [QuantifiedAspect("battery life", 87.0, 71.0, 29.0, "declining")]
    clusters = [Cluster("battery life", [AspectClaim("c1", "battery life", "positive", "lasts 3 days")], 1, 0)]
    result = Summarizer(config).run(aspects, clusters)
    assert len(result) == 1
    assert isinstance(result[0], AspectSummary)


def test_one_summary_per_aspect(config, mocker):
    mock_llm_response(mocker, {"headline": "X", "detail": "Y", "trend_note": ""})
    aspects = [
        QuantifiedAspect("battery", 87.0, 71.0, 29.0, "stable"),
        QuantifiedAspect("ANC", 60.0, 50.0, 50.0, "stable"),
    ]
    clusters = [
        Cluster("battery", [], 0, 0),
        Cluster("ANC", [], 0, 0),
    ]
    result = Summarizer(config).run(aspects, clusters)
    assert len(result) == 2


# ── correctness tests ─────────────────────────────────────────────────────────

def test_preserves_quantified_numbers(config, mocker):
    mock_llm_response(mocker, {"headline": "Battery fades over time", "detail": "Users note degradation.", "trend_note": "Worse recently."})
    aspects = [QuantifiedAspect("battery life", 87.0, 71.0, 29.0, "declining")]
    clusters = [Cluster("battery life", [], 0, 0)]

    result = Summarizer(config).run(aspects, clusters)

    assert result[0].mention_pct == 87.0
    assert result[0].positive_pct == 71.0
    assert result[0].negative_pct == 29.0
    assert result[0].recent_trend == "declining"


def test_uses_llm_text_for_headline_and_detail(config, mocker):
    mock_llm_response(mocker, {
        "headline": "Battery lasts days not hours",
        "detail": "Most users report 30-hour real-world battery life.",
        "trend_note": "",
    })
    aspects = [QuantifiedAspect("battery life", 87.0, 71.0, 29.0, "stable")]
    clusters = [Cluster("battery life", [], 0, 0)]

    result = Summarizer(config).run(aspects, clusters)

    assert result[0].headline == "Battery lasts days not hours"
    assert result[0].detail == "Most users report 30-hour real-world battery life."
    assert result[0].trend_note == ""


def test_passes_quantified_data_in_prompt(config, mocker):
    mock = mock_llm_response(mocker, {"headline": "X", "detail": "Y", "trend_note": ""})
    aspects = [QuantifiedAspect("ANC", 91.0, 78.0, 22.0, "improving")]
    clusters = [Cluster("ANC", [], 0, 0)]

    Summarizer(config).run(aspects, clusters)

    prompt = mock.call_args.kwargs["messages"][0]["content"]
    assert "91.0" in prompt
    assert "78.0" in prompt
    assert "improving" in prompt


def test_passes_quotes_from_cluster_in_prompt(config, mocker):
    mock = mock_llm_response(mocker, {"headline": "X", "detail": "Y", "trend_note": ""})
    claims = [
        AspectClaim("c1", "battery life", "positive", "lasts three full days"),
        AspectClaim("c2", "battery life", "negative", "dies after 8 months"),
    ]
    aspects = [QuantifiedAspect("battery life", 87.0, 71.0, 29.0, "declining")]
    clusters = [Cluster("battery life", claims, 1, 1)]

    Summarizer(config).run(aspects, clusters)

    prompt = mock.call_args.kwargs["messages"][0]["content"]
    assert "lasts three full days" in prompt
    assert "dies after 8 months" in prompt


def test_uses_configured_model(config, mocker):
    mock = mock_llm_response(mocker, {"headline": "X", "detail": "Y", "trend_note": ""})
    aspects = [QuantifiedAspect("battery", 87.0, 71.0, 29.0, "stable")]
    clusters = [Cluster("battery", [], 0, 0)]

    Summarizer(config).run(aspects, clusters)

    assert mock.call_args.kwargs["model"] == "ollama/llama3.2"
