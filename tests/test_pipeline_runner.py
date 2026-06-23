import pytest
from datetime import datetime, timezone
from pipeline.runner import PipelineRunner
from pipeline.types import (
    ProductInfo, RawComment, AspectClaim, Cluster, QuantifiedAspect, AspectSummary,
    PipelineResult,
)


# ── builders for mock components ────────────────────────────────────────────────

def make_product():
    return ProductInfo("sony-xm5", "Sony WH-1000XM5", "headphones", ["XM5"], ["headphones"])


def make_comment(id):
    return RawComment(id, "text", 10, datetime.now(timezone.utc), "headphones", "url")


def make_runner(mocker, *, claims=None, summaries=None):
    """Build a PipelineRunner with all 7 components mocked."""
    product = make_product()
    comments = [make_comment("c1"), make_comment("c2")]
    filtered = [make_comment("c1")]
    claims = [AspectClaim("c1", "battery", "positive", "good")] if claims is None else claims
    clusters = [Cluster("battery", claims, 1, 0)]
    aspects = [QuantifiedAspect("battery", 50.0, 100.0, 0.0, "stable")]
    summaries = [AspectSummary("battery", 50.0, 100.0, 0.0, "stable", "h", "d")] if summaries is None else summaries

    product_agent = mocker.Mock(); product_agent.run.return_value = product
    scraper = mocker.Mock(); scraper.run.return_value = comments
    noise_filter = mocker.Mock(); noise_filter.run.return_value = filtered
    aspect_extractor = mocker.Mock(); aspect_extractor.run.return_value = claims
    embedder_clusterer = mocker.Mock(); embedder_clusterer.run.return_value = clusters
    quantifier = mocker.Mock(); quantifier.run.return_value = aspects
    summarizer = mocker.Mock(); summarizer.run.return_value = summaries

    runner = PipelineRunner(
        product_agent=product_agent, scraper=scraper, noise_filter=noise_filter,
        aspect_extractor=aspect_extractor, embedder_clusterer=embedder_clusterer,
        quantifier=quantifier, summarizer=summarizer,
    )
    return runner, {
        "product": product, "comments": comments, "filtered": filtered,
        "claims": claims, "clusters": clusters, "aspects": aspects, "summaries": summaries,
        "product_agent": product_agent, "scraper": scraper, "noise_filter": noise_filter,
        "aspect_extractor": aspect_extractor, "embedder_clusterer": embedder_clusterer,
        "quantifier": quantifier, "summarizer": summarizer,
    }


# ── structural / orchestration tests ────────────────────────────────────────────

def test_returns_pipeline_result_with_product_and_summaries(mocker):
    runner, ctx = make_runner(mocker)
    result = runner.run("Sony WH-1000XM5")
    assert isinstance(result, PipelineResult)
    assert result.product is ctx["product"]
    assert result.summaries is ctx["summaries"]
    assert result.comment_count == 1  # len(filtered)


def test_data_flows_between_stages(mocker):
    runner, ctx = make_runner(mocker)
    runner.run("Sony WH-1000XM5")

    ctx["product_agent"].run.assert_called_once_with("Sony WH-1000XM5")
    ctx["scraper"].run.assert_called_once_with(ctx["product"])
    ctx["noise_filter"].run.assert_called_once_with(ctx["comments"])
    # extractor gets the product too, so it can ignore comparisons to other products
    ctx["aspect_extractor"].run.assert_called_once_with(ctx["filtered"], ctx["product"])
    ctx["embedder_clusterer"].run.assert_called_once_with(ctx["claims"])
    # quantifier needs both clusters and the FILTERED comments (for trend dates)
    ctx["quantifier"].run.assert_called_once_with(ctx["clusters"], ctx["filtered"])
    ctx["summarizer"].run.assert_called_once_with(ctx["aspects"], ctx["clusters"])


# ── correctness: short-circuit when there are no opinions ───────────────────────

def test_short_circuits_when_no_claims(mocker):
    runner, ctx = make_runner(mocker, claims=[])
    result = runner.run("Obscure Product 9000")

    # still returns the product so the caller can say "no opinions for <product>"
    assert isinstance(result, PipelineResult)
    assert result.product is ctx["product"]
    assert result.summaries == []
    assert result.comment_count == 1  # len(filtered)
    # the expensive / pointless downstream stages must NOT run
    ctx["embedder_clusterer"].run.assert_not_called()
    ctx["quantifier"].run.assert_not_called()
    ctx["summarizer"].run.assert_not_called()


# ── progress reporting ──────────────────────────────────────────────────────────

def test_progress_callback_invoked_per_stage(mocker):
    runner, ctx = make_runner(mocker)
    stages = []
    runner.run("Sony WH-1000XM5", progress_callback=stages.append)

    assert stages == [
        "understanding", "scraping", "filtering", "extracting",
        "clustering", "quantifying", "summarizing",
    ]


def test_progress_callback_stops_at_extracting_when_no_claims(mocker):
    runner, ctx = make_runner(mocker, claims=[])
    stages = []
    runner.run("Obscure Product 9000", progress_callback=stages.append)

    assert stages == ["understanding", "scraping", "filtering", "extracting"]


def test_works_without_progress_callback(mocker):
    runner, ctx = make_runner(mocker)
    # should not raise
    result = runner.run("Sony WH-1000XM5")
    assert result.summaries is ctx["summaries"]
