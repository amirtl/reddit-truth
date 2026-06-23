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
