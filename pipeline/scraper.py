import praw
from datetime import datetime, timezone
from .types import ProductInfo, RawComment


class RedditScraper:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

    def run(self, product: ProductInfo, limit: int = 100) -> list[RawComment]:
        comments: list[RawComment] = []
        seen_ids: set[str] = set()
        for term in product.search_terms[:3]:
            results = self.reddit.subreddit("all").search(
                term, limit=limit, sort="relevance", time_filter="year"
            )
            for submission in results:
                comments.extend(self._comments_from_submission(submission, seen_ids))
        return comments

    def _comments_from_submission(self, submission, seen_ids: set[str]) -> list[RawComment]:
        submission.comments.replace_more(limit=0)
        comments = []
        for comment in submission.comments.list():
            if comment.id not in seen_ids:
                seen_ids.add(comment.id)
                comments.append(RawComment(
                    id=comment.id,
                    text=comment.body,
                    score=comment.score,
                    created_at=datetime.fromtimestamp(comment.created_utc, tz=timezone.utc),
                    subreddit=str(comment.subreddit),
                    post_url=submission.url,
                ))
        return comments
