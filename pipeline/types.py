from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class ProductInfo:
    canonical_id: str
    canonical_name: str
    category: str
    search_terms: list[str]
    subreddits: list[str]


@dataclass
class RawComment:
    id: str
    text: str
    score: int
    created_at: datetime
    subreddit: str
    post_url: str


@dataclass
class AspectClaim:
    comment_id: str
    aspect: str
    sentiment: Literal["positive", "negative", "mixed"]
    quote: str


@dataclass
class Cluster:
    label: str
    claims: list[AspectClaim]
    positive_count: int
    negative_count: int


@dataclass
class QuantifiedAspect:
    label: str
    mention_pct: float
    positive_pct: float
    negative_pct: float
    recent_trend: Literal["improving", "declining", "stable"]


@dataclass
class AspectSummary:
    label: str
    mention_pct: float
    positive_pct: float
    negative_pct: float
    recent_trend: str
    headline: str
    detail: str
    trend_note: str = ""
