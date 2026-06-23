from pipeline.agent_tools import validate_subreddit


class FakeClient:
    def __init__(self, by_query):  # {query: [titles] or None}
        self.by_query = by_query

    def post_titles(self, subreddit, query, limit=10):
        return self.by_query.get(query)


def test_validate_subreddit_counts_title_matches():
    client = FakeClient({"AirPods Pro 2": ["AirPods Pro 2 review", "off topic"]})
    assert validate_subreddit(client, "airpods", ["AirPods Pro 2"]) == 1


def test_validate_subreddit_returns_none_when_all_queries_error():
    client = FakeClient({"AirPods Pro 2": None})
    assert validate_subreddit(client, "airpods", ["AirPods Pro 2"]) is None


def test_validate_subreddit_zero_when_no_titles_match():
    client = FakeClient({"AirPods Pro 2": ["random unrelated thread"]})
    assert validate_subreddit(client, "airpods", ["AirPods Pro 2"]) == 0
