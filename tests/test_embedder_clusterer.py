import pytest
from pipeline.config import AppConfig, LLMConfig, EmbeddingConfig
from pipeline.embedder_clusterer import EmbedderClusterer
from pipeline.types import AspectClaim, Cluster


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


@pytest.fixture
def mixed_claims():
    return [
        AspectClaim("c1", "battery life", "positive", "lasts 3 days"),
        AspectClaim("c2", "battery life", "positive", "great battery"),
        AspectClaim("c3", "battery life", "negative", "dies fast"),
        AspectClaim("c4", "ANC quality", "positive", "best ANC"),
        AspectClaim("c5", "ANC quality", "positive", "kills background noise"),
    ]


def test_returns_list_of_clusters(config, mixed_claims):
    ec = EmbedderClusterer(config)
    result = ec.run(mixed_claims)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(c, Cluster) for c in result)


def test_clusters_have_positive_negative_counts(config, mixed_claims):
    ec = EmbedderClusterer(config)
    result = ec.run(mixed_claims)
    for cluster in result:
        assert cluster.positive_count >= 0
        assert cluster.negative_count >= 0
        assert cluster.positive_count + cluster.negative_count <= len(cluster.claims)


def test_returns_empty_for_empty_input(config):
    ec = EmbedderClusterer(config)
    assert ec.run([]) == []


def test_sorted_by_claim_count(config, mixed_claims):
    ec = EmbedderClusterer(config)
    result = ec.run(mixed_claims)
    counts = [len(c.claims) for c in result]
    assert counts == sorted(counts, reverse=True)


def test_groups_semantically_similar_claims_together(config, mixed_claims):
    ec = EmbedderClusterer(config)
    result = ec.run(mixed_claims)

    battery_cluster = next(c for c in result if "battery" in c.label.lower())
    anc_cluster = next(c for c in result if "anc" in c.label.lower())

    battery_ids = {claim.comment_id for claim in battery_cluster.claims}
    anc_ids = {claim.comment_id for claim in anc_cluster.claims}

    assert {"c1", "c2", "c3"} == battery_ids
    assert {"c4", "c5"} == anc_ids
