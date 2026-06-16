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
def battery_and_anc_claims():
    return [
        AspectClaim("c1", "battery life", "positive", "lasts 3 days"),
        AspectClaim("c2", "battery life", "positive", "great battery"),
        AspectClaim("c3", "battery life", "negative", "dies fast"),
        AspectClaim("c4", "ANC quality", "positive", "best ANC"),
        AspectClaim("c5", "ANC quality", "positive", "kills background noise"),
    ]


# ── structural tests ────────────────────────────────────────────────────────

def test_returns_list_of_clusters(config, battery_and_anc_claims):
    result = EmbedderClusterer(config).run(battery_and_anc_claims)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(c, Cluster) for c in result)


def test_returns_empty_for_empty_input(config):
    assert EmbedderClusterer(config).run([]) == []


def test_sorted_by_claim_count(config, battery_and_anc_claims):
    result = EmbedderClusterer(config).run(battery_and_anc_claims)
    counts = [len(c.claims) for c in result]
    assert counts == sorted(counts, reverse=True)


def test_no_claim_is_lost(config, battery_and_anc_claims):
    result = EmbedderClusterer(config).run(battery_and_anc_claims)
    all_ids = [claim.comment_id for cluster in result for claim in cluster.claims]
    assert sorted(all_ids) == sorted(c.comment_id for c in battery_and_anc_claims)


def test_positive_negative_counts_are_correct(config, battery_and_anc_claims):
    result = EmbedderClusterer(config).run(battery_and_anc_claims)
    for cluster in result:
        expected_pos = sum(1 for c in cluster.claims if c.sentiment == "positive")
        expected_neg = sum(1 for c in cluster.claims if c.sentiment == "negative")
        assert cluster.positive_count == expected_pos
        assert cluster.negative_count == expected_neg


# ── correctness tests ───────────────────────────────────────────────────────

def test_groups_semantically_similar_claims_together(config, battery_and_anc_claims):
    result = EmbedderClusterer(config).run(battery_and_anc_claims)

    battery_cluster = next(c for c in result if "battery" in c.label.lower())
    anc_cluster = next(c for c in result if "anc" in c.label.lower())

    assert {claim.comment_id for claim in battery_cluster.claims} == {"c1", "c2", "c3"}
    assert {claim.comment_id for claim in anc_cluster.claims} == {"c4", "c5"}


def test_single_topic_returns_one_cluster(config):
    claims = [
        AspectClaim("c1", "sound quality", "positive", "amazing sound"),
        AspectClaim("c2", "sound quality", "positive", "great audio"),
        AspectClaim("c3", "sound quality", "negative", "tinny sound"),
    ]
    result = EmbedderClusterer(config).run(claims)
    assert len(result) == 1
    assert len(result[0].claims) == 3


def test_three_distinct_topics_cluster_separately(config):
    claims = [
        AspectClaim("c1", "battery life", "positive", "lasts all day"),
        AspectClaim("c2", "battery life", "positive", "long battery"),
        AspectClaim("c3", "ANC", "positive", "silence everything"),
        AspectClaim("c4", "ANC", "negative", "ANC is weak"),
        AspectClaim("c5", "comfort", "positive", "very comfortable"),
        AspectClaim("c6", "comfort", "positive", "fits perfectly"),
    ]
    result = EmbedderClusterer(config).run(claims)

    labels = {c.label.lower() for c in result}
    assert any("battery" in l for l in labels)
    assert any("anc" in l for l in labels)
    assert any("comfort" in l for l in labels)


def test_single_claim_returns_one_cluster(config):
    claims = [AspectClaim("c1", "battery life", "positive", "lasts 3 days")]
    result = EmbedderClusterer(config).run(claims)
    assert len(result) == 1
    assert result[0].claims[0].comment_id == "c1"
