"""Unit tests for the *pure* logic of the smoke-test harness.

The harness itself (HTTP submit/poll/fetch) IS an integration test and is not
unit-tested — mocking the very network it exists to exercise would be circular.
But its shape-validation logic is pure and deterministic, so we pin it here.
"""
from scripts.smoke_test import is_terminal, summary_problems


def valid_summary():
    return {
        "aspect": "battery life",
        "mention_pct": 80.0,
        "positive_pct": 75.0,
        "negative_pct": 25.0,
        "recent_trend": "improving",
        "headline": "Battery lasts days",
        "detail": "Most users report multi-day battery life.",
        "trend_note": "Up recently.",
    }


# ── is_terminal ─────────────────────────────────────────────────────────────────

def test_done_and_failed_are_terminal():
    assert is_terminal("done")
    assert is_terminal("failed")


def test_pending_and_running_are_not_terminal():
    assert not is_terminal("pending")
    assert not is_terminal("running")


# ── summary_problems ────────────────────────────────────────────────────────────

def test_valid_summary_has_no_problems():
    assert summary_problems(valid_summary()) == []


def test_missing_field_is_reported():
    s = valid_summary()
    del s["headline"]
    problems = summary_problems(s)
    assert any("headline" in p for p in problems)


def test_empty_headline_is_reported():
    s = valid_summary()
    s["headline"] = "   "
    assert any("headline" in p for p in summary_problems(s))


def test_out_of_range_percentage_is_reported():
    s = valid_summary()
    s["mention_pct"] = 150.0
    assert any("mention_pct" in p for p in summary_problems(s))


def test_invalid_trend_value_is_reported():
    s = valid_summary()
    s["recent_trend"] = "skyrocketing"
    assert any("recent_trend" in p for p in summary_problems(s))
