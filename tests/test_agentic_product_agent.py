from pipeline.agentic_product_agent import decide, MAX_ITERS


def _state(subs, noisy=None, iters=0):
    return {"raw_query": "q", "draft": None,
            "validation": {"subreddits": subs, "noisy_terms": noisy or []},
            "iterations": iters, "history": []}


def test_decide_finalizes_when_enough_productive_and_clean():
    assert decide(_state({"apple": 8, "headphones": 10})) == "finalize"


def test_decide_revises_when_too_few_productive():
    assert decide(_state({"iOS": 3, "fake": 0})) == "revise"


def test_decide_revises_when_terms_noisy():
    assert decide(_state({"apple": 8, "headphones": 10}, noisy=["Pro 2"])) == "revise"


def test_decide_finalizes_at_iteration_cap_even_if_weak():
    assert decide(_state({"fake": 0}, iters=MAX_ITERS)) == "finalize"


def test_unknown_subreddits_dont_count_as_productive():
    # None == couldn't check; must not be treated as productive
    assert decide(_state({"a": None, "b": None})) == "revise"
