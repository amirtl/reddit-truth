"""Clustering must generalize across product types — not just headphones.

Each case lists realistic aspect labels for a product, the variant pairs that
must MERGE into one cluster, and distinct aspects that must NOT merge. Behaviour
verified empirically against the embedding model at the tuned threshold.
"""
import pytest

from pipeline.config import AppConfig, LLMConfig, EmbeddingConfig
from pipeline.embedder_clusterer import EmbedderClusterer
from pipeline.types import AspectClaim


@pytest.fixture(scope="module")
def clusterer():
    # module-scoped so the embedding model loads once for all cases
    cfg = AppConfig(
        llms=LLMConfig(product_understanding="ollama/x", aspect_extraction="gemini/x", summarization="ollama/x"),
        embeddings=EmbeddingConfig(provider="local", model="all-MiniLM-L6-v2"),
    )
    return EmbedderClusterer(cfg)


def aspect_clusters(clusterer, aspects):
    """Return {aspect -> frozenset of comment_ids in its cluster}."""
    claims = []
    for asp in aspects:
        claims.append(AspectClaim(f"{asp}#0", asp, "positive", "good"))
        claims.append(AspectClaim(f"{asp}#1", asp, "negative", "bad"))
    result = clusterer.run(claims)
    mapping = {}
    for asp in aspects:
        for c in result:
            ids = {cl.comment_id for cl in c.claims}
            if f"{asp}#0" in ids:
                mapping[asp] = frozenset(ids)
                break
    return mapping


# (name, aspects, merge_pairs, distinct_aspects)
CASES = [
    ("headphones",
     ["noise cancellation", "battery life", "battery", "comfort", "sound quality", "build quality", "price"],
     [("battery life", "battery")],
     ["noise cancellation", "comfort", "sound quality", "price"]),
    ("phone",
     ["battery life", "camera", "camera quality", "display", "screen", "performance", "price"],
     [("camera", "camera quality"), ("display", "screen")],
     ["battery life", "performance", "price"]),
    ("laptop",
     ["battery life", "keyboard", "display", "screen", "trackpad", "performance", "price"],
     [("display", "screen")],
     ["keyboard", "battery life", "trackpad", "performance", "price"]),
    ("car",
     ["range", "battery range", "interior", "autopilot", "price", "ride comfort"],
     [("range", "battery range")],
     ["interior", "autopilot", "price"]),
    ("vacuum",
     ["suction", "suction power", "battery life", "weight", "price"],
     [("suction", "suction power")],
     ["battery life", "weight", "price"]),
    ("e-reader",
     ["screen", "display", "battery life", "lighting", "backlight", "weight", "price"],
     [("screen", "display"), ("lighting", "backlight")],
     ["battery life", "weight", "price"]),
    ("mouse",
     ["ergonomics", "scroll wheel", "scrolling", "battery life", "buttons", "price"],
     [("scroll wheel", "scrolling")],
     ["ergonomics", "battery life", "buttons", "price"]),
    ("game",
     ["story", "graphics", "combat", "combat system", "music", "performance"],
     [("combat", "combat system")],
     ["story", "graphics", "music", "performance"]),
    ("console (no variants)",
     ["battery life", "screen", "performance", "joycon drift", "game library", "price"],
     [],
     ["battery life", "screen", "performance", "joycon drift", "game library", "price"]),
    ("earbuds",
     ["sound quality", "fit", "comfort", "battery life", "call quality", "price"],
     [],
     ["sound quality", "fit", "comfort", "battery life", "call quality", "price"]),
]


@pytest.mark.parametrize("name,aspects,merge,distinct", CASES, ids=[c[0] for c in CASES])
def test_clustering_generalizes_across_products(clusterer, name, aspects, merge, distinct):
    m = aspect_clusters(clusterer, aspects)

    for a, b in merge:
        assert m[a] == m[b], f"{name}: '{a}' and '{b}' should have merged"

    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            a, b = distinct[i], distinct[j]
            assert m[a] != m[b], f"{name}: distinct aspects '{a}' and '{b}' were wrongly merged"
