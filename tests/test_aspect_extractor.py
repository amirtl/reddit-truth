import json
import pytest
from datetime import datetime
from pipeline.config import AppConfig, LLMConfig, EmbeddingConfig
from pipeline.aspect_extractor import AspectExtractor
from pipeline.types import RawComment, AspectClaim


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


def make_comment(id, text):
    return RawComment(id, text, 10, datetime(2024, 1, 1), "headphones", "https://reddit.com")


def test_extracts_claims_from_batch(config, mocker):
    mock_response = {"claims": [
        {"comment_id": "c1", "aspect": "battery life", "sentiment": "positive", "quote": "lasts 3 days"},
        {"comment_id": "c2", "aspect": "ANC", "sentiment": "negative", "quote": "ANC is weak"},
    ]}
    mock_completion = mocker.patch("pipeline.aspect_extractor.litellm.completion")
    mock_completion.return_value.choices[0].message.content = json.dumps(mock_response)

    extractor = AspectExtractor(config)
    comments = [
        make_comment("c1", "Battery life is great, lasts 3 days easily"),
        make_comment("c2", "ANC is weak compared to Bose QC45"),
    ]
    result = extractor.run(comments)

    assert len(result) == 2
    assert all(isinstance(r, AspectClaim) for r in result)
    assert result[0].aspect == "battery life"
    assert result[0].sentiment == "positive"


def test_handles_multi_aspect_comment(config, mocker):
    mock_response = {"claims": [
        {"comment_id": "c1", "aspect": "battery life", "sentiment": "positive", "quote": "great battery"},
        {"comment_id": "c1", "aspect": "ANC", "sentiment": "negative", "quote": "ANC disappoints"},
    ]}
    mock_completion = mocker.patch("pipeline.aspect_extractor.litellm.completion")
    mock_completion.return_value.choices[0].message.content = json.dumps(mock_response)

    extractor = AspectExtractor(config)
    result = extractor.run([make_comment("c1", "Great battery but ANC disappoints")])

    assert len(result) == 2
    assert all(c.comment_id == "c1" for c in result)


def test_batches_large_input(config, mocker):
    mock_completion = mocker.patch("pipeline.aspect_extractor.litellm.completion")
    mock_completion.return_value.choices[0].message.content = json.dumps({"claims": []})

    extractor = AspectExtractor(config)
    comments = [make_comment(f"c{i}", f"Comment number {i} about product quality and performance") for i in range(40)]
    extractor.run(comments)

    # 40 comments / batch size 15 → ceil(40/15) = 3 calls
    assert mock_completion.call_count == 3


def test_includes_comment_ids_in_prompt(config, mocker):
    mock_completion = mocker.patch("pipeline.aspect_extractor.litellm.completion")
    mock_completion.return_value.choices[0].message.content = json.dumps({"claims": []})

    extractor = AspectExtractor(config)
    extractor.run([make_comment("abc123", "Sound quality is really good for the price")])

    prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
    assert "abc123" in prompt
