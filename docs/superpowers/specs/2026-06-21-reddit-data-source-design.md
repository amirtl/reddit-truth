# Reddit data source — pluggable scraper backend

**Date:** 2026-06-21
**Status:** Approved

## Problem

The official Reddit API path is unusable for this project right now:

1. **Credential gate** — Reddit's "Responsible Builder Policy" blocks creating a
   script app for API credentials, even with a verified one-year-old account.
2. **Network unreachable** — from the dev/runtime environment, Reddit's servers
   return `http=000` (connection reset) while other hosts return `200`. So even
   *with* credentials, PRAW (which hits `oauth.reddit.com`) would likely fail.

The pipeline needs, per product, a few hundred recent comments across subreddits
with `body / score / created_utc / subreddit / permalink` → `RawComment`.

## Investigation (tested live)

| Source | No creds | Reachable | Recency | Comment fields | Keyword search |
|---|---|---|---|---|---|
| **Arctic-Shift** | yes | yes | **today** | yes | per-subreddit only |
| PullPush | yes | yes | frozen ~May 2025 | yes | all-Reddit |
| Official PRAW | no (gated) | no (`http=000`) | live | yes | all-Reddit |

PullPush is frozen at May 2025, which silently breaks the recent-vs-old trend
feature. Arctic-Shift is current to today and reachable; its only limitation —
no all-Reddit keyword search — is a non-issue because the product-understanding
agent already emits candidate subreddits. The full chain was verified live:
`posts/search?subreddit=X&query=term` → submission ids →
`comments/search?link_id=id` → comments. This mirrors the existing PRAW flow.

## Decision

Add **Arctic-Shift as a swappable scraper backend, default**, alongside the
existing PRAW `RedditScraper` (kept as portfolio evidence and a future option).
A config flag selects the backend. No PullPush fallback (YAGNI).

## Design

**Port.** A `Scraper` `Protocol`: `run(product: ProductInfo, limit: int) ->
list[RawComment]`. Both backends conform structurally; `pipeline/` is unchanged;
`build_runner` depends on the protocol, not a concrete class.

**ArcticShiftScraper.** Base `https://arctic-shift.photon-reddit.com/api`.
- For each search term × candidate subreddit (capped): `GET /posts/search?
  subreddit={sub}&query={term}&sort=desc&limit=N` → submission ids (deduped).
- For each submission: `GET /comments/search?link_id={id}&limit=M` → comments.
- Map → `RawComment(id, body, score, created_utc→tz-aware datetime, subreddit,
  "https://reddit.com"+permalink)`; skip `[deleted]`/`[removed]`/empty; dedup by
  id; stop at `limit`.
- Private methods: `_collect_submission_ids`, `_search_submissions`,
  `_fetch_comments`, `_to_raw_comment`.

**Config & wiring.** `config.yml` gains:
```yaml
scraper:
  backend: arctic_shift   # or "praw"
```
A `ScraperConfig` pydantic model (default `backend="arctic_shift"`, optional on
`AppConfig` so existing fixtures keep working). `build_runner` constructs
`ArcticShiftScraper` or `RedditScraper(env…)` on that flag.

**Error handling & rate limiting.** Arctic-Shift 422s ("slow down") under bursts:
sequential requests with a small configurable delay, conservative caps
(~5 subreddits, ~3 terms, ~10 posts/query), a few retries with backoff on
timeout/error payloads, and per-request failures skipped (not fatal). Zero
comments → existing "no opinions found" path handles it.

## Testing

- Unit tests mock the HTTP boundary (`requests`): correct URL/params, JSON →
  `RawComment` mapping, dedup, limit honored, erroring subreddit skipped,
  `[deleted]` filtered. `request_delay=0` in tests.
- The live `make smoke` over HTTP exercises real Arctic-Shift end to end; the
  `smoke_offline` fixture path remains for offline runs.

## Out of scope (YAGNI)

PullPush fallback, response caching, concurrency, retry-storm logic beyond simple
backoff — all addable later behind the same `Scraper` port.
