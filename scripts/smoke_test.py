#!/usr/bin/env python
"""End-to-end smoke test for reddit-truth.

Nothing is mocked. This drives the REAL running stack through the HTTP API:
submit a query, watch the job go pending -> running -> done, then fetch the
summaries. It asserts the pipes connect and the output is well-shaped — NOT
exact values, since a real LLM is non-deterministic.

Prerequisites (all must be running): Postgres, Redis, an LLM backend (Ollama
and/or Gemini key), valid Reddit API creds, the Django server, and a Celery
worker. See the "Smoke test" section of the README.

Usage:
    python scripts/smoke_test.py "Sony WH-1000XM5"
    SMOKE_BASE_URL=http://localhost:8000 python scripts/smoke_test.py
"""
import argparse
import os
import sys
import time

import requests

DEFAULT_BASE_URL = os.environ.get("SMOKE_BASE_URL", "http://localhost:8000")
DEFAULT_QUERY = "Sony WH-1000XM5"
POLL_INTERVAL_SECONDS = 3
TIMEOUT_SECONDS = 600

TERMINAL_STATUSES = {"done", "failed"}
VALID_TRENDS = {"improving", "declining", "stable"}
PERCENTAGE_FIELDS = ("mention_pct", "positive_pct", "negative_pct")
TEXT_FIELDS = ("aspect", "headline", "detail")
REQUIRED_SUMMARY_FIELDS = (
    "aspect", "mention_pct", "positive_pct", "negative_pct",
    "recent_trend", "headline", "detail", "trend_note",
)


# ── pure logic (unit-tested) ────────────────────────────────────────────────────

def is_terminal(status: str) -> bool:
    """A job is terminal once the worker can no longer change it."""
    return status in TERMINAL_STATUSES


def summary_problems(summary: dict) -> list[str]:
    """Return a list of human-readable shape problems with one summary dict.

    Empty list == well-formed. Validates structure/ranges only, never the
    semantic content (which an LLM produces non-deterministically).
    """
    problems: list[str] = []

    for field in REQUIRED_SUMMARY_FIELDS:
        if field not in summary:
            problems.append(f"missing field: {field}")

    for field in PERCENTAGE_FIELDS:
        value = summary.get(field)
        if isinstance(value, (int, float)):
            if not (0.0 <= value <= 100.0):
                problems.append(f"{field} out of range: {value}")
        elif field in summary:
            problems.append(f"{field} is not a number: {value!r}")

    for field in TEXT_FIELDS:
        value = summary.get(field)
        if field in summary and (not isinstance(value, str) or not value.strip()):
            problems.append(f"{field} is empty")

    trend = summary.get("recent_trend")
    if "recent_trend" in summary and trend not in VALID_TRENDS:
        problems.append(f"recent_trend invalid: {trend!r}")

    return problems


# ── IO orchestration (the integration test itself) ──────────────────────────────

def submit(base_url: str, query: str) -> str:
    resp = requests.post(f"{base_url}/api/jobs/", json={"query": query}, timeout=30)
    if resp.status_code != 202:
        raise SystemExit(f"submit failed: {resp.status_code} {resp.text}")
    job_id = resp.json()["job_id"]
    print(f"→ submitted '{query}' → job {job_id}")
    return job_id


def poll(base_url: str, job_id: str) -> dict:
    deadline = time.monotonic() + TIMEOUT_SECONDS
    last_message = None
    while True:
        resp = requests.get(f"{base_url}/api/jobs/{job_id}/", timeout=30)
        resp.raise_for_status()
        job = resp.json()

        message = (job["status"], job.get("status_message", ""), job.get("progress", 0))
        if message != last_message:
            print(f"  [{job['status']:<8}] {job.get('progress', 0):>3}%  {job.get('status_message', '')}")
            last_message = message

        if is_terminal(job["status"]):
            return job
        if time.monotonic() > deadline:
            raise SystemExit(f"timed out after {TIMEOUT_SECONDS}s (last status: {job['status']})")
        time.sleep(POLL_INTERVAL_SECONDS)


def fetch_summaries(base_url: str, canonical_id: str) -> list[dict]:
    resp = requests.get(f"{base_url}/api/products/{canonical_id}/summaries/", timeout=30)
    resp.raise_for_status()
    return resp.json()


def report(job: dict, summaries: list[dict]) -> int:
    """Print a human report and return a process exit code."""
    if job["status"] == "failed":
        print(f"\n✗ job FAILED: {job.get('status_message')}")
        return 1

    if not summaries:
        # The pipeline ran and completed, but found no opinions — downstream
        # stages (cluster/quantify/summarize) were skipped. Valid, but weak.
        print(f"\n⚠ job done but no summaries: {job.get('status_message')}")
        print("  Try a more popular product to exercise the full pipeline.")
        return 0

    all_problems: list[str] = []
    print(f"\n✓ job done — {len(summaries)} aspect summaries:")
    for s in summaries:
        problems = summary_problems(s)
        all_problems.extend(problems)
        flag = "  ✓" if not problems else "  ✗"
        print(f"{flag} {s.get('aspect')}: {s.get('headline')}")
        for p in problems:
            print(f"      ! {p}")

    if all_problems:
        print(f"\n✗ {len(all_problems)} shape problem(s) found")
        return 1
    print("\n✓ smoke test passed — all summaries well-formed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end smoke test for reddit-truth.")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY, help="product to analyse")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args(argv)

    try:
        job_id = submit(args.base_url, args.query)
        job = poll(args.base_url, job_id)

        summaries = []
        if job["status"] == "done" and job.get("canonical_id"):
            summaries = fetch_summaries(args.base_url, job["canonical_id"])
    except requests.ConnectionError:
        print(f"✗ could not reach {args.base_url} — is the server running? (make run)")
        return 1

    return report(job, summaries)


if __name__ == "__main__":
    sys.exit(main())
