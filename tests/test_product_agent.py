import json
import pytest
from pipeline.config import AppConfig, LLMConfig, EmbeddingConfig
from pipeline.product_agent import ProductUnderstandingAgent, parse_product_info
from pipeline.types import ProductInfo


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


def test_run_returns_product_info(config, mocker):
    mock_response = {
        "canonical_id": "sony-wh-1000xm5",
        "canonical_name": "Sony WH-1000XM5",
        "category": "headphones",
        "search_terms": ["WH-1000XM5", "Sony XM5"],
        "subreddits": ["headphones", "audiophile"],
    }
    mock_completion = mocker.patch("pipeline.product_agent.litellm.completion")
    mock_completion.return_value.choices[0].message.content = json.dumps(mock_response)

    agent = ProductUnderstandingAgent(config)
    result = agent.run("Sony WH-1000XM5")

    assert isinstance(result, ProductInfo)
    assert result.canonical_id == "sony-wh-1000xm5"
    assert "headphones" in result.subreddits
    mock_completion.assert_called_once()


def test_run_passes_query_in_prompt(config, mocker):
    mock_completion = mocker.patch("pipeline.product_agent.litellm.completion")
    mock_completion.return_value.choices[0].message.content = json.dumps({
        "canonical_id": "test-product",
        "canonical_name": "Test Product",
        "category": "electronics",
        "search_terms": ["test"],
        "subreddits": ["gadgets"],
    })

    agent = ProductUnderstandingAgent(config)
    agent.run("some product name")

    call_args = mock_completion.call_args
    prompt = call_args.kwargs["messages"][0]["content"]
    assert "some product name" in prompt


def _mock_llm(mocker, payload):
    m = mocker.patch("pipeline.product_agent.litellm.completion")
    m.return_value.choices[0].message.content = json.dumps(payload)
    return m


def test_missing_keys_do_not_crash(config, mocker):
    _mock_llm(mocker, {})  # LLM returned nothing useful
    result = ProductUnderstandingAgent(config).run("Dyson V15")
    assert isinstance(result, ProductInfo)
    assert result.canonical_name  # falls back, never empty
    assert result.search_terms    # never empty


def test_empty_search_terms_falls_back_to_query(config, mocker):
    _mock_llm(mocker, {"canonical_name": "Dyson V15", "search_terms": [], "subreddits": ["dyson"]})
    result = ProductUnderstandingAgent(config).run("Dyson V15")
    assert "Dyson V15" in result.search_terms


def test_raw_query_always_included_as_search_term(config, mocker):
    # guards against the model returning terms for the wrong product
    _mock_llm(mocker, {
        "canonical_id": "x", "canonical_name": "X", "category": "headphones",
        "search_terms": ["Sony WH-1000XM5"], "subreddits": ["headphones"],
    })
    result = ProductUnderstandingAgent(config).run("Baldur's Gate 3")
    assert "Baldur's Gate 3" in result.search_terms


def test_prompt_demands_specific_terms_not_generic(config, mocker):
    mock_completion = mocker.patch("pipeline.product_agent.litellm.completion")
    mock_completion.return_value.choices[0].message.content = json.dumps({
        "canonical_id": "x", "canonical_name": "X", "category": "headphones",
        "search_terms": ["x"], "subreddits": ["y"],
    })

    ProductUnderstandingAgent(config).run("Sony WH-1000XM5")

    prompt = mock_completion.call_args.kwargs["messages"][0]["content"].lower()
    # must steer the model toward specific identifiers and away from category terms
    assert "specific" in prompt
    assert "generic" in prompt or "category" in prompt


def test_run_uses_configured_model(config, mocker):
    mock_completion = mocker.patch("pipeline.product_agent.litellm.completion")
    mock_completion.return_value.choices[0].message.content = json.dumps({
        "canonical_id": "test-product",
        "canonical_name": "Test Product",
        "category": "electronics",
        "search_terms": ["test"],
        "subreddits": ["gadgets"],
    })

    agent = ProductUnderstandingAgent(config)
    agent.run("test product")

    call_args = mock_completion.call_args
    assert call_args.kwargs["model"] == "ollama/llama3.2"


def test_parse_product_info_keeps_query_and_drops_empty():
    info = parse_product_info(
        {"canonical_name": "X", "search_terms": [], "subreddits": ["a", ""]}, "My Query")
    assert "My Query" in info.search_terms          # raw query always present
    assert info.subreddits == ["a"]                  # empties dropped
