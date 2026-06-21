import pytest
from datetime import datetime, timedelta, timezone
from pipeline.types import RawComment, AspectClaim, Cluster, QuantifiedAspect
from pipeline.quantifier import Quantifier


def make_comment(id, days_ago=30):
    return RawComment(
        id, "some text", 10,
        datetime.now(timezone.utc) - timedelta(days=days_ago),
        "headphones", "https://reddit.com"
    )


def make_cluster(label, claims, positive_count, negative_count):
    return Cluster(label=label, claims=claims, positive_count=positive_count, negative_count=negative_count)


# ── structural tests ─────────────────────────────────────────────────────────

def test_returns_list_of_quantified_aspects():
    comments = [make_comment(f"c{i}") for i in range(5)]
    cluster = make_cluster("battery", [AspectClaim(f"c{i}", "battery", "positive", "good") for i in range(3)], 3, 0)
    result = Quantifier().run([cluster], comments)
    assert isinstance(result, list)
    assert all(isinstance(a, QuantifiedAspect) for a in result)


def test_sorted_by_mention_pct():
    comments = [make_comment(f"c{i}") for i in range(10)]
    big = make_cluster("battery", [AspectClaim(f"c{i}", "battery", "positive", "good") for i in range(8)], 8, 0)
    small = make_cluster("ANC", [AspectClaim("c8", "ANC", "positive", "good"), AspectClaim("c9", "ANC", "positive", "ok")], 2, 0)
    result = Quantifier().run([small, big], comments)
    assert result[0].label == "battery"


# ── correctness tests ─────────────────────────────────────────────────────────

def test_mention_pct_calculation():
    comments = [make_comment(f"c{i}") for i in range(10)]
    claims = [AspectClaim(f"c{i}", "battery", "positive", "great") for i in range(8)]
    cluster = make_cluster("battery", claims, 8, 0)
    result = Quantifier().run([cluster], comments)
    assert result[0].mention_pct == 80.0


def test_mention_pct_counts_unique_comments_not_claims():
    # c1 has two claims — should count as one mention; c2 is a second mention
    comments = [make_comment("c1"), make_comment("c2"), make_comment("c3")]
    claims = [
        AspectClaim("c1", "battery", "positive", "great battery"),
        AspectClaim("c1", "battery", "positive", "lasts long"),  # same comment, second claim
        AspectClaim("c2", "battery", "positive", "good"),
    ]
    cluster = make_cluster("battery", claims, 3, 0)
    result = Quantifier().run([cluster], comments)
    assert result[0].mention_pct == pytest.approx(66.7, abs=0.1)  # 2 unique comments / 3, not 3/3


# ── min-mention filter (drops the single-mention tail) ────────────────────────

def test_drops_single_mention_aspects():
    comments = [make_comment(f"c{i}") for i in range(5)]
    big = make_cluster("battery", [AspectClaim(f"c{i}", "battery", "positive", "g") for i in range(3)], 3, 0)
    singleton = make_cluster("warranty", [AspectClaim("c4", "warranty", "negative", "bad")], 0, 1)
    result = Quantifier().run([big, singleton], comments)
    labels = [a.label for a in result]
    assert "battery" in labels
    assert "warranty" not in labels


def test_aspect_with_two_claims_from_one_comment_is_dropped():
    # mentioned twice but by a single commenter — not a signal
    comments = [make_comment("c1"), make_comment("c2")]
    claims = [
        AspectClaim("c1", "hinge", "negative", "broke"),
        AspectClaim("c1", "hinge", "negative", "cracked"),
    ]
    cluster = make_cluster("hinge", claims, 0, 2)
    result = Quantifier().run([cluster], comments)
    assert result == []


def test_positive_negative_pct():
    comments = [make_comment(f"c{i}") for i in range(3)]
    claims = [
        AspectClaim("c1", "ANC", "positive", "great"),
        AspectClaim("c2", "ANC", "negative", "weak"),
        AspectClaim("c3", "ANC", "positive", "excellent"),
    ]
    cluster = make_cluster("ANC", claims, 2, 1)
    result = Quantifier().run([cluster], comments)
    assert result[0].positive_pct == pytest.approx(66.7, abs=0.1)
    assert result[0].negative_pct == pytest.approx(33.3, abs=0.1)


def test_trend_declining_when_recent_more_negative():
    old_comments  = [make_comment(f"old{i}", days_ago=200) for i in range(5)]
    recent_comments = [make_comment(f"new{i}", days_ago=10) for i in range(5)]
    all_comments = old_comments + recent_comments

    old_claims    = [AspectClaim(f"old{i}", "durability", "positive", "good") for i in range(5)]
    recent_claims = [AspectClaim(f"new{i}", "durability", "negative", "broke") for i in range(5)]
    cluster = make_cluster("durability", old_claims + recent_claims, 5, 5)

    result = Quantifier().run([cluster], all_comments)
    assert result[0].recent_trend == "declining"


def test_trend_improving_when_recent_more_positive():
    old_comments    = [make_comment(f"old{i}", days_ago=200) for i in range(5)]
    recent_comments = [make_comment(f"new{i}", days_ago=10) for i in range(5)]
    all_comments = old_comments + recent_comments

    old_claims    = [AspectClaim(f"old{i}", "battery", "negative", "bad") for i in range(5)]
    recent_claims = [AspectClaim(f"new{i}", "battery", "positive", "great") for i in range(5)]
    cluster = make_cluster("battery", old_claims + recent_claims, 5, 5)

    result = Quantifier().run([cluster], all_comments)
    assert result[0].recent_trend == "improving"


def test_trend_stable_when_no_significant_change():
    comments = [make_comment(f"c{i}", days_ago=10) for i in range(4)]
    claims = [
        AspectClaim("c0", "comfort", "positive", "comfy"),
        AspectClaim("c1", "comfort", "positive", "fits well"),
        AspectClaim("c2", "comfort", "negative", "tight"),
        AspectClaim("c3", "comfort", "positive", "great fit"),
    ]
    cluster = make_cluster("comfort", claims, 3, 1)
    result = Quantifier().run([cluster], comments)
    assert result[0].recent_trend == "stable"


def test_trend_stable_when_no_recent_comments():
    comments = [make_comment(f"c{i}", days_ago=200) for i in range(5)]
    claims = [AspectClaim(f"c{i}", "battery", "positive", "good") for i in range(5)]
    cluster = make_cluster("battery", claims, 5, 0)
    result = Quantifier().run([cluster], comments)
    assert result[0].recent_trend == "stable"
