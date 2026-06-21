from datetime import timezone
from unittest.mock import MagicMock

import pytest

from pipeline.arctic_shift_scraper import ArcticShiftScraper
from pipeline.types import ProductInfo, RawComment


def make_product(terms=None, subs=None):
    return ProductInfo(
        canonical_id="sony-xm5",
        canonical_name="Sony WH-1000XM5",
        category="headphones",
        search_terms=terms if terms is not None else ["WH-1000XM5"],
        subreddits=subs if subs is not None else ["headphones"],
    )


def post(pid, title="XM5 review thread"):
    # Default title contains the product alias so the focus filter keeps it.
    return {"id": pid, "title": title}


def comment(cid, body="great battery life on these headphones", score=5,
            created=1750000000, sub="headphones", permalink="/r/headphones/comments/x/c/"):
    return {
        "id": cid, "body": body, "score": score, "created_utc": created,
        "subreddit": sub, "permalink": permalink,
    }


def fake_http(posts: dict, comments: dict):
    """Build a requests.get replacement driven by canned data.

    posts:    {(subreddit, query): [ post dicts ]}
    comments: {link_id: [comment dicts]}
    """
    def _get(url, params=None, headers=None, timeout=None):
        resp = MagicMock()
        if "/posts/search" in url:
            key = (params["subreddit"], params["query"])
            resp.json.return_value = {"data": posts.get(key, [])}
        elif "/comments/search" in url:
            resp.json.return_value = {"data": comments.get(params["link_id"], [])}
        else:
            resp.json.return_value = {"data": []}
        return resp
    return _get


def scraper():
    return ArcticShiftScraper(request_delay=0)


# ── search fan-out ────────────────────────────────────────────────────────────

def test_searches_each_term_subreddit_combo(mocker):
    get = mocker.patch("pipeline.arctic_shift_scraper.requests.get",
                       side_effect=fake_http({}, {}))
    product = make_product(terms=["XM5", "Sony"], subs=["headphones", "audiophile"])

    scraper().run(product)

    searched = {
        (c.kwargs["params"]["subreddit"], c.kwargs["params"]["query"])
        for c in get.call_args_list if "/posts/search" in c.args[0]
    }
    assert searched == {
        ("headphones", "XM5"), ("audiophile", "XM5"),
        ("headphones", "Sony"), ("audiophile", "Sony"),
    }


# ── mapping ───────────────────────────────────────────────────────────────────

def test_maps_comments_to_rawcomment(mocker):
    mocker.patch("pipeline.arctic_shift_scraper.requests.get", side_effect=fake_http(
        posts={("headphones", "XM5"): [post("sub1", "XM5 long-term review")]},
        comments={"sub1": [comment("c1", score=12, created=1750000000)]},
    ))
    result = scraper().run(make_product(terms=["XM5"]))

    assert len(result) == 1
    rc = result[0]
    assert isinstance(rc, RawComment)
    assert rc.id == "c1"
    assert rc.text == "great battery life on these headphones"
    assert rc.score == 12
    assert rc.subreddit == "headphones"
    assert rc.post_url == "https://reddit.com/r/headphones/comments/x/c/"
    assert rc.created_at.tzinfo == timezone.utc
    assert rc.created_at.year == 2025


# ── precision: only harvest title-focused threads ─────────────────────────────

def test_keeps_only_title_focused_posts(mocker):
    mocker.patch("pipeline.arctic_shift_scraper.requests.get", side_effect=fake_http(
        posts={("headphones", "XM5"): [
            post("focused", "My XM5 after six months"),       # alias in title -> keep
            post("offtopic", "Best headphones under $400?"),  # no alias -> drop
        ]},
        comments={"focused": [comment("c_on")], "offtopic": [comment("c_off")]},
    ))
    result = scraper().run(make_product(terms=["XM5"]))
    ids = {c.id for c in result}
    assert "c_on" in ids
    assert "c_off" not in ids


def test_alias_match_is_word_bounded(mocker):
    # "XM5" must not match "XM500" or be a loose substring
    mocker.patch("pipeline.arctic_shift_scraper.requests.get", side_effect=fake_http(
        posts={("headphones", "XM5"): [post("p", "AKG XM500 unboxing")]},
        comments={"p": [comment("c")]},
    ))
    result = scraper().run(make_product(terms=["XM5"]))
    assert result == []


# ── dedup / limit / filtering / resilience ────────────────────────────────────

def test_deduplicates_comments_across_submissions(mocker):
    mocker.patch("pipeline.arctic_shift_scraper.requests.get", side_effect=fake_http(
        posts={("headphones", "XM5"): [post("sub1"), post("sub2")]},
        comments={"sub1": [comment("dup")], "sub2": [comment("dup"), comment("c2")]},
    ))
    result = scraper().run(make_product(terms=["XM5"]))
    assert sorted(c.id for c in result) == ["c2", "dup"]


def test_honors_limit(mocker):
    mocker.patch("pipeline.arctic_shift_scraper.requests.get", side_effect=fake_http(
        posts={("headphones", "XM5"): [post("sub1")]},
        comments={"sub1": [comment(f"c{i}") for i in range(10)]},
    ))
    result = scraper().run(make_product(terms=["XM5"]), limit=3)
    assert len(result) == 3


def test_skips_deleted_removed_and_empty(mocker):
    mocker.patch("pipeline.arctic_shift_scraper.requests.get", side_effect=fake_http(
        posts={("headphones", "XM5"): [post("sub1")]},
        comments={"sub1": [
            comment("c1", body="[deleted]"),
            comment("c2", body="[removed]"),
            comment("c3", body=""),
            comment("c4", body="this one is a real opinion about the sound"),
        ]},
    ))
    result = scraper().run(make_product(terms=["XM5"]))
    assert [c.id for c in result] == ["c4"]


def test_failing_request_is_skipped_not_fatal(mocker):
    import requests

    def flaky_get(url, params=None, headers=None, timeout=None):
        if params.get("subreddit") == "broken" and "/posts/search" in url:
            raise requests.RequestException("boom")
        return fake_http(
            posts={("headphones", "XM5"): [post("sub1")]},
            comments={"sub1": [comment("c1")]},
        )(url, params=params)

    mocker.patch("pipeline.arctic_shift_scraper.requests.get", side_effect=flaky_get)
    result = scraper().run(make_product(terms=["XM5"], subs=["broken", "headphones"]))
    assert [c.id for c in result] == ["c1"]


def test_arctic_shift_error_payload_is_skipped(mocker):
    def err_get(url, params=None, headers=None, timeout=None):
        resp = MagicMock()
        resp.json.return_value = {"data": None, "error": "Timeout. Maybe slow down a bit"}
        return resp

    mocker.patch("pipeline.arctic_shift_scraper.requests.get", side_effect=err_get)
    result = scraper().run(make_product(terms=["XM5"]))
    assert result == []
