from pipeline.agent_tools import validate_subreddit, check_term_precision


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


def test_term_precision_low_for_generic_term():
    # "Pro 2" matches all three; only one also names the anchor "AirPods Pro 2"
    client = FakeClient({"Pro 2": ["AirPods Pro 2 thread", "Surface Pro 2", "iPad Pro 2"]})
    p = check_term_precision(client, "Pro 2", "AirPods Pro 2", ["gadgets"])
    assert p < 0.5


def test_term_precision_high_for_specific_term():
    client = FakeClient({"AirPods Pro 2": ["AirPods Pro 2 review", "AirPods Pro 2 vs Sony"]})
    assert check_term_precision(client, "AirPods Pro 2", "AirPods Pro 2", ["gadgets"]) == 1.0


def test_term_precision_returns_1_when_no_data():
    client = FakeClient({"Ghost": None})
    assert check_term_precision(client, "Ghost", "Ghost", ["gadgets"]) == 1.0
