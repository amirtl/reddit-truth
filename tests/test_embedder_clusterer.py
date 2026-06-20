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


@pytest.fixture
def realistic_claims():
    # Long, discursive quotes that share opinion vocabulary — the set that
    # collapsed the whole pipeline into one cluster in the live smoke run. The
    # "price" claims are the connective tissue that bridges every aspect.
    return [
        AspectClaim("c1", "noise cancelling", "positive", "blocks out the entire airplane engine drone completely"),
        AspectClaim("c2", "noise cancelling", "positive", "best I have ever experienced, far better than the Bose QC45"),
        AspectClaim("c3", "noise cancelling", "positive", "subway noise basically disappears when I put them on"),
        AspectClaim("c4", "battery life", "negative", "barely makes it two days now after eight months of use"),
        AspectClaim("c5", "battery life", "negative", "drains much faster than when I first bought them"),
        AspectClaim("c6", "battery life", "negative", "dying after only a year, really disappointing"),
        AspectClaim("c7", "battery life", "positive", "easily lasted thirty hours per charge on long trips"),
        AspectClaim("c8", "comfort", "positive", "earcups are soft and they never clamp too hard on my head"),
        AspectClaim("c9", "comfort", "positive", "I forget I am wearing them during an entire eight hour work day"),
        AspectClaim("c10", "sound quality", "positive", "rich and detailed with deep bass, slightly warm signature"),
        AspectClaim("c11", "sound quality", "positive", "vocals are clear and the soundstage feels surprisingly wide"),
        AspectClaim("c12", "call quality", "negative", "people told me I sounded muffled and distant during calls"),
        AspectClaim("c13", "call quality", "negative", "friends say my voice sounds thin and processed on calls"),
        AspectClaim("c14", "build quality", "negative", "plastic hinges feel a little fragile to me"),
        AspectClaim("c15", "price", "positive", "worth every penny, comfort sound and noise cancelling justify the cost"),
        AspectClaim("c16", "price", "positive", "steep but the multipoint bluetooth and app features make them complete"),
    ]


def test_realistic_quotes_keep_distinct_aspects_separate(config, realistic_claims):
    result = EmbedderClusterer(config).run(realistic_claims)

    # The whole value prop depends on aspects NOT collapsing into one bucket.
    assert len(result) >= 4, f"aspects collapsed into {len(result)} cluster(s)"

    battery = next(c for c in result if "battery" in c.label.lower())
    anc = next(c for c in result if "noise" in c.label.lower())
    battery_ids = {cl.comment_id for cl in battery.claims}
    anc_ids = {cl.comment_id for cl in anc.claims}

    assert battery_ids.isdisjoint(anc_ids), "battery and ANC claims merged together"
    assert {"c4", "c5", "c6", "c7"}.issubset(battery_ids)
    assert {"c1", "c2", "c3"}.issubset(anc_ids)
