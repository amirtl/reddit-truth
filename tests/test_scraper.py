import pytest
from unittest.mock import MagicMock
from pipeline.scraper import RedditScraper
from pipeline.types import ProductInfo, RawComment


@pytest.fixture
def product_info():
    return ProductInfo(
        canonical_id="sony-wh-1000xm5",
        canonical_name="Sony WH-1000XM5",
        category="headphones",
        search_terms=["WH-1000XM5"],
        subreddits=["headphones"],
    )


def _make_mock_comment(id, body, score=10, created_utc=1700000000):
    c = MagicMock()
    c.id = id
    c.body = body
    c.score = score
    c.created_utc = created_utc
    c.subreddit = MagicMock()
    c.subreddit.__str__ = lambda self: "headphones"
    return c


def test_run_returns_raw_comments(product_info, mocker):
    mock_submission = MagicMock()
    mock_submission.url = "https://reddit.com/r/headphones/comments/abc"
    mock_comment = _make_mock_comment("c1", "Battery life is incredible on these headphones")
    mock_submission.comments.list.return_value = [mock_comment]

    mock_reddit = mocker.patch("pipeline.scraper.praw.Reddit")
    mock_reddit.return_value.subreddit.return_value.search.return_value = [mock_submission]

    scraper = RedditScraper("fake_id", "fake_secret", "fake_agent")
    results = scraper.run(product_info, limit=10)

    assert len(results) == 1
    assert isinstance(results[0], RawComment)
    assert results[0].id == "c1"
    assert results[0].subreddit == "headphones"


def test_run_deduplicates_comments(product_info, mocker):
    mock_submission = MagicMock()
    mock_submission.url = "https://reddit.com/r/headphones/comments/abc"
    same_comment = _make_mock_comment("dup1", "This comment appears in multiple search results")
    mock_submission.comments.list.return_value = [same_comment, same_comment]

    mock_reddit = mocker.patch("pipeline.scraper.praw.Reddit")
    mock_reddit.return_value.subreddit.return_value.search.return_value = [mock_submission, mock_submission]

    scraper = RedditScraper("fake_id", "fake_secret", "fake_agent")
    results = scraper.run(product_info, limit=10)

    assert [r.id for r in results].count("dup1") == 1


def test_run_maps_comment_fields_correctly(product_info, mocker):
    mock_submission = MagicMock()
    mock_submission.url = "https://reddit.com/r/headphones/comments/abc"
    mock_comment = _make_mock_comment("c1", "Great sound quality overall", score=55, created_utc=1700000000)
    mock_submission.comments.list.return_value = [mock_comment]

    mock_reddit = mocker.patch("pipeline.scraper.praw.Reddit")
    mock_reddit.return_value.subreddit.return_value.search.return_value = [mock_submission]

    scraper = RedditScraper("fake_id", "fake_secret", "fake_agent")
    results = scraper.run(product_info, limit=10)

    assert results[0].text == "Great sound quality overall"
    assert results[0].score == 55
    assert results[0].post_url == "https://reddit.com/r/headphones/comments/abc"
