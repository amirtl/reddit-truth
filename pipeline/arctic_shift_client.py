import re
import time

import requests


class ArcticShiftClient:
    """Low-level Arctic-Shift search. Returns post titles, or None on error so
    callers can tell 'genuinely empty' (drop the subreddit) from 'couldn't
    check' (don't punish it)."""

    def __init__(self, base_url: str = "https://arctic-shift.photon-reddit.com/api",
                 user_agent: str = "reddit-truth/0.1",
                 request_delay: float = 0.5, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.headers = {"User-Agent": user_agent}
        self.request_delay = request_delay
        self.max_retries = max_retries

    def post_titles(self, subreddit: str, query: str, limit: int = 10) -> list[str] | None:
        for attempt in range(self.max_retries):
            try:
                r = requests.get(self.base_url + "/posts/search", params={
                    "subreddit": subreddit, "query": query, "sort": "desc", "limit": limit,
                }, headers=self.headers, timeout=30)
                payload = r.json()
                if isinstance(payload, dict) and payload.get("error"):
                    time.sleep(self.request_delay * (attempt + 1))
                    continue
                data = payload.get("data") if isinstance(payload, dict) else payload
                time.sleep(self.request_delay)
                return [p.get("title", "") for p in (data or [])]
            except (requests.RequestException, ValueError):
                time.sleep(self.request_delay * (attempt + 1))
        return None

    @staticmethod
    def alias_matcher(terms: list[str]) -> re.Pattern | None:
        parts = [re.escape(t) for t in terms if t]
        if not parts:
            return None
        return re.compile(rf"\b(?:{'|'.join(parts)})\b", re.IGNORECASE)
