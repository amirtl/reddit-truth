import re
import time
from datetime import datetime, timezone

import requests

from .types import ProductInfo, RawComment

_SKIP_BODIES = {"", "[deleted]", "[removed]"}


class ArcticShiftScraper:
    """Scraper backend over the Arctic-Shift public archive (no Reddit creds).

    Mirrors the PRAW flow: find submissions about the product within the
    candidate subreddits, then pull their comments. Conforms to the Scraper port
    (run(product, limit) -> list[RawComment]).
    """

    def __init__(
        self,
        base_url: str = "https://arctic-shift.photon-reddit.com/api",
        user_agent: str = "reddit-truth/0.1",
        max_subreddits: int = 5,
        max_terms: int = 3,
        posts_per_query: int = 10,
        comments_per_post: int = 100,
        request_delay: float = 0.5,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = {"User-Agent": user_agent}
        self.max_subreddits = max_subreddits
        self.max_terms = max_terms
        self.posts_per_query = posts_per_query
        self.comments_per_post = comments_per_post
        self.request_delay = request_delay
        self.max_retries = max_retries

    def run(self, product: ProductInfo, limit: int = 100) -> list[RawComment]:
        matcher = self._alias_matcher(product.search_terms)
        comments: list[RawComment] = []
        seen: set[str] = set()
        for link_id in self._collect_submission_ids(product, matcher):
            if len(comments) >= limit:
                break
            for raw in self._fetch_comments(link_id):
                if raw.id in seen:
                    continue
                seen.add(raw.id)
                comments.append(raw)
                if len(comments) >= limit:
                    break
        return comments

    def _alias_matcher(self, aliases: list[str]) -> re.Pattern | None:
        """Word-bounded matcher for the product's aliases, so "XM5" matches a
        standalone mention but not "XM500"."""
        parts = [re.escape(a) for a in aliases if a]
        if not parts:
            return None
        return re.compile(rf"\b(?:{'|'.join(parts)})\b", re.IGNORECASE)

    def _collect_submission_ids(self, product: ProductInfo, matcher: re.Pattern | None) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for term in product.search_terms[: self.max_terms]:
            for subreddit in product.subreddits[: self.max_subreddits]:
                for sid in self._search_submissions(term, subreddit, matcher):
                    if sid not in seen:
                        seen.add(sid)
                        ids.append(sid)
        return ids

    def _search_submissions(self, term: str, subreddit: str, matcher: re.Pattern | None) -> list[str]:
        data = self._get("/posts/search", {
            "subreddit": subreddit, "query": term,
            "sort": "desc", "limit": self.posts_per_query,
        })
        # Precision: keep only threads whose TITLE names the product. Threads that
        # merely mention it in the body are ~99% off-topic (measured), so their
        # comments are dropped wholesale.
        return [
            p["id"] for p in data
            if p.get("id") and matcher and matcher.search(p.get("title", ""))
        ]

    def _fetch_comments(self, link_id: str) -> list[RawComment]:
        data = self._get("/comments/search", {
            "link_id": link_id, "limit": self.comments_per_post,
        })
        return [self._to_raw_comment(c) for c in data if self._is_usable(c)]

    def _is_usable(self, comment: dict) -> bool:
        return bool(comment.get("id")) and comment.get("created_utc") is not None \
            and comment.get("body", "") not in _SKIP_BODIES

    def _to_raw_comment(self, comment: dict) -> RawComment:
        return RawComment(
            id=comment["id"],
            text=comment["body"],
            score=int(comment.get("score", 0) or 0),
            created_at=datetime.fromtimestamp(float(comment["created_utc"]), tz=timezone.utc),
            subreddit=comment.get("subreddit", ""),
            post_url="https://reddit.com" + comment.get("permalink", ""),
        )

    def _get(self, path: str, params: dict) -> list[dict]:
        """GET with backoff. Returns the data list, or [] on persistent failure —
        a flaky subreddit must not kill the whole scrape."""
        url = self.base_url + path
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, headers=self.headers, timeout=30)
                payload = response.json()
                if isinstance(payload, dict) and payload.get("error"):
                    time.sleep(self.request_delay * (attempt + 1))  # rate-limited; back off
                    continue
                data = payload.get("data") if isinstance(payload, dict) else payload
                time.sleep(self.request_delay)
                return data or []
            except (requests.RequestException, ValueError):
                time.sleep(self.request_delay * (attempt + 1))
        return []
