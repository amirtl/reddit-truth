from datetime import datetime
from pipeline.types import ProductInfo, RawComment, AspectClaim, Cluster, QuantifiedAspect, AspectSummary


def test_product_info_creation():
    p = ProductInfo(
        canonical_id="sony-wh-1000xm5",
        canonical_name="Sony WH-1000XM5",
        category="headphones",
        search_terms=["WH-1000XM5", "Sony XM5"],
        subreddits=["headphones", "audiophile"],
    )
    assert p.canonical_id == "sony-wh-1000xm5"
    assert len(p.search_terms) == 2


def test_raw_comment_creation():
    c = RawComment(
        id="abc123",
        text="Battery life is incredible, lasts 3 days easily",
        score=42,
        created_at=datetime(2024, 6, 1),
        subreddit="headphones",
        post_url="https://reddit.com/r/headphones/comments/abc",
    )
    assert c.score == 42


def test_aspect_claim_creation():
    claim = AspectClaim(
        comment_id="abc123",
        aspect="battery life",
        sentiment="positive",
        quote="lasts 3 days easily",
    )
    assert claim.sentiment == "positive"


def test_cluster_counts():
    claims = [
        AspectClaim("c1", "battery life", "positive", "great battery"),
        AspectClaim("c2", "battery life", "negative", "dies fast"),
    ]
    cluster = Cluster(label="battery life", claims=claims, positive_count=1, negative_count=1)
    assert len(cluster.claims) == 2
    assert cluster.positive_count + cluster.negative_count == len(cluster.claims)


def test_quantified_aspect_percentages():
    aspect = QuantifiedAspect(
        label="battery life",
        mention_pct=87.0,
        positive_pct=71.0,
        negative_pct=29.0,
        recent_trend="declining",
    )
    assert aspect.mention_pct == 87.0
    assert aspect.recent_trend == "declining"


def test_aspect_summary_has_all_fields():
    summary = AspectSummary(
        label="battery life",
        mention_pct=87.0,
        positive_pct=71.0,
        negative_pct=29.0,
        recent_trend="declining",
        headline="Battery fades after 8 months",
        detail="Most users praise battery life initially but report degradation.",
    )
    assert summary.headline == "Battery fades after 8 months"
    assert summary.trend_note == ""
