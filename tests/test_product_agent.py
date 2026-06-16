import json
import pytest
from pipeline.config import AppConfig, LLMConfig, EmbeddingConfig
from pipeline.product_agent import ProductUnderstandingAgent
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
