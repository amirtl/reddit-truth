import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock
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


def _mock_claims(mocker, claims):
    m = mocker.patch("pipeline.aspect_extractor.litellm.completion")
    m.return_value.choices[0].message.content = json.dumps({"claims": claims})
    return m


def test_skips_claim_missing_aspect(config, mocker):
    _mock_claims(mocker, [
        {"comment_id": "c1", "sentiment": "positive", "quote": "x"},  # no aspect
        {"comment_id": "c1", "aspect": "battery", "sentiment": "positive", "quote": "y"},
    ])
    result = AspectExtractor(config).run([make_comment("c1", "Battery is great and lasts long")])
    assert [c.aspect for c in result] == ["battery"]


def test_normalizes_unknown_sentiment_to_mixed(config, mocker):
    _mock_claims(mocker, [{"comment_id": "c1", "aspect": "battery", "sentiment": "amazing", "quote": "y"}])
    result = AspectExtractor(config).run([make_comment("c1", "Battery is great and lasts long")])
    assert result[0].sentiment == "mixed"


def test_drops_hallucinated_comment_id(config, mocker):
    # the model invents an id not in the batch — must not pollute the results
    _mock_claims(mocker, [
        {"comment_id": "zzz", "aspect": "battery", "sentiment": "positive", "quote": "y"},
        {"comment_id": "c1", "aspect": "sound", "sentiment": "positive", "quote": "y"},
    ])
    result = AspectExtractor(config).run([make_comment("c1", "Battery is great and the sound is clear")])
    assert [c.comment_id for c in result] == ["c1"]


def test_missing_quote_defaults_to_empty(config, mocker):
    _mock_claims(mocker, [{"comment_id": "c1", "aspect": "battery", "sentiment": "positive"}])
    result = AspectExtractor(config).run([make_comment("c1", "Battery is great and lasts long")])
    assert result[0].quote == ""


def test_capitalized_sentiment_is_normalized(config, mocker):
    # some models (llama3.2) return "Negative" — must map to "negative", not "mixed"
    _mock_claims(mocker, [{"comment_id": "c1", "aspect": "battery", "sentiment": "Negative", "quote": "y"}])
    result = AspectExtractor(config).run([make_comment("c1", "Battery is great and lasts long")])
    assert result[0].sentiment == "negative"


def test_integer_comment_id_is_matched(config, mocker):
    # some models return the id as an int when the prompt numbering looks numeric
    _mock_claims(mocker, [{"comment_id": 1, "aspect": "battery", "sentiment": "positive", "quote": "y"}])
    result = AspectExtractor(config).run([make_comment("1", "Battery is great and lasts long")])
    assert [c.comment_id for c in result] == ["1"]


def test_non_dict_claims_are_skipped(config, mocker):
    _mock_claims(mocker, ["not a dict", 123, {"comment_id": "c1", "aspect": "battery", "sentiment": "positive", "quote": "y"}])
    result = AspectExtractor(config).run([make_comment("c1", "Battery is great and lasts long")])
    assert len(result) == 1


def test_prompt_asks_for_canonical_aspect_labels(config, mocker):
    mock_completion = mocker.patch("pipeline.aspect_extractor.litellm.completion")
    mock_completion.return_value.choices[0].message.content = json.dumps({"claims": []})

    AspectExtractor(config).run([make_comment("c1", "Sound is great and the battery lasts a long time")])

    prompt = mock_completion.call_args.kwargs["messages"][0]["content"].lower()
    # steer toward consistent canonical labels and expanded abbreviations
    assert "canonical" in prompt or "consistent" in prompt
    assert "abbreviation" in prompt or "anc" in prompt


def test_comment_ids_correctly_attributed_across_batches(config, mocker):
    # simulate two batches: first returns claims for batch-1 comments,
    # second returns claims for batch-2 comments
    batch1_response = {"claims": [
        {"comment_id": "c0", "aspect": "battery", "sentiment": "positive", "quote": "great battery"},
    ]}
    batch2_response = {"claims": [
        {"comment_id": "c15", "aspect": "ANC", "sentiment": "negative", "quote": "weak ANC"},
    ]}
    mock_completion = mocker.patch("pipeline.aspect_extractor.litellm.completion")
    mock_completion.side_effect = [
        MagicMock(**{"choices": [MagicMock(**{"message.content": json.dumps(batch1_response)})]}),
        MagicMock(**{"choices": [MagicMock(**{"message.content": json.dumps(batch2_response)})]}),
    ]

    extractor = AspectExtractor(config)
    comments = [make_comment(f"c{i}", f"Comment {i} about this product quality") for i in range(16)]
    result = extractor.run(comments)

    ids = [r.comment_id for r in result]
    assert "c0" in ids
    assert "c15" in ids
