import re

from pipeline.arctic_shift_client import ArcticShiftClient


def validate_subreddit(client, subreddit: str, terms: list[str], max_terms: int = 3) -> int | None:
    """How many distinct posts in r/<subreddit> have a TITLE matching the product.
    None if every query errored (unknown — caller must not treat as 0)."""
    matcher = ArcticShiftClient.alias_matcher(terms)
    if matcher is None:
        return 0
    matched: set[str] = set()
    saw_response = False
    for term in terms[:max_terms]:
        titles = client.post_titles(subreddit, term)
        if titles is None:
            continue
        saw_response = True
        for t in titles:
            if matcher.search(t):
                matched.add(t)
    return len(matched) if saw_response else None


def check_term_precision(client, term: str, anchor: str, subreddits: list[str], max_subs: int = 3) -> float:
    """Of the titles a term matches, what fraction also contain the specific
    anchor. Low = the term is generic ('Pro 2' matches Surface/iPad too).
    Returns 1.0 when there's no evidence, to avoid dropping terms blindly."""
    term_re = re.compile(re.escape(term), re.IGNORECASE)
    anchor_re = re.compile(re.escape(anchor), re.IGNORECASE)
    total = on_topic = 0
    for sub in subreddits[:max_subs]:
        titles = client.post_titles(sub, term)
        if not titles:
            continue
        for t in titles:
            if term_re.search(t):
                total += 1
                if anchor_re.search(t):
                    on_topic += 1
    return 1.0 if total == 0 else on_topic / total

