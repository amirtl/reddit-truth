import json

from pipeline.agentic_product_agent import decide, MAX_ITERS, AgenticProductAgent
from pipeline.types import ProductInfo


class _FakeConfig:
    class llms:
        product_understanding = "ollama/gemma2:9b"


class FakeClient:
    """Models Arctic-Shift: dead subreddits return [] for any term; productive
    ones return N on-topic titles for a specific term, and a caller-supplied
    mixed list for a 'noisy' generic term."""

    def __init__(self, productive, noisy=None):
        self.productive = productive          # {subreddit: n_threads}
        self.noisy = noisy or {}              # {term: [titles]}

    def post_titles(self, subreddit, query, limit=10):
        if self.productive.get(subreddit, 0) == 0:
            return []
        if query in self.noisy:
            return self.noisy[query]
        return [f"{query} thread {i}" for i in range(self.productive[subreddit])]


def _agent(client):
    return AgenticProductAgent(config=None, client=client)


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


def test_validate_flags_dead_subreddits_and_noisy_terms():
    draft = ProductInfo("x", "AirPods Pro 2", "earbuds",
                        ["AirPods Pro 2", "Pro 2"], ["apple", "fake"])
    client = FakeClient(
        productive={"apple": 5, "fake": 0},
        noisy={"Pro 2": ["AirPods Pro 2 x", "Surface Pro 2", "iPad Pro 2", "Galaxy Pro 2"]},
    )
    out = _agent(client)._validate({"draft": draft, "validation": {}, "iterations": 0,
                                    "raw_query": "AirPods Pro 2", "history": []})
    assert out["validation"]["subreddits"]["fake"] == 0
    assert out["validation"]["subreddits"]["apple"] > 0
    assert "Pro 2" in out["validation"]["noisy_terms"]


def test_specific_term_not_flagged_when_a_longer_variant_exists():
    # The raw query "AirPods Pro 2" is the anchor. A longer variant
    # "AirPods Pro 2 review" must NOT cause the good term "AirPods Pro 2" to be
    # flagged noisy (the longest-term-anchor bug burned revise loops live).
    draft = ProductInfo("x", "AirPods Pro 2", "earbuds",
                        ["AirPods Pro 2", "AirPods Pro 2 review"], ["apple"])
    client = FakeClient(productive={"apple": 3})
    out = _agent(client)._validate({"draft": draft, "validation": {}, "iterations": 0,
                                    "raw_query": "AirPods Pro 2", "history": []})
    assert "AirPods Pro 2" not in out["validation"]["noisy_terms"]


def test_finalize_keeps_productive_subreddits_first():
    draft = ProductInfo("x", "X", "c", ["X"], ["fake", "apple"])
    state = {"draft": draft,
             "validation": {"subreddits": {"fake": 0, "apple": 8}, "noisy_terms": []},
             "iterations": 1, "raw_query": "X", "history": []}
    out = _agent(FakeClient({}))._finalize(state)
    assert out["draft"].subreddits[0] == "apple"   # productive first
    assert "fake" not in out["draft"].subreddits    # dead dropped


def test_propose_uses_product_understanding_agent(mocker):
    mocker.patch("pipeline.product_agent.litellm.completion").return_value.choices[0].message.content = json.dumps(
        {"canonical_id": "p", "canonical_name": "P", "category": "c",
         "search_terms": ["P"], "subreddits": ["sub"]})
    agent = AgenticProductAgent(config=_FakeConfig(), client=FakeClient({}))
    out = agent._propose({"raw_query": "P", "history": []})
    assert out["draft"].subreddits == ["sub"]


def test_revise_increments_iterations_and_reparses(mocker):
    mocker.patch("pipeline.agentic_product_agent.litellm.completion").return_value.choices[0].message.content = json.dumps(
        {"canonical_id": "p", "canonical_name": "P", "category": "c",
         "search_terms": ["P"], "subreddits": ["good"]})
    agent = AgenticProductAgent(config=_FakeConfig(), client=FakeClient({}))
    state = {"raw_query": "P", "iterations": 0, "history": [],
             "draft": ProductInfo("p", "P", "c", ["P"], ["bad"]),
             "validation": {"subreddits": {"bad": 0}, "noisy_terms": []}}
    out = agent._revise(state)
    assert out["iterations"] == 1
    assert out["draft"].subreddits == ["good"]


def test_revise_preserves_productive_subreddits(mocker):
    # The LLM may return a fresh list that drops already-productive subreddits.
    # revise must keep validated-productive subs (don't lose progress) — the bug
    # behind Kindle's intermittent 0-comment runs.
    mocker.patch("pipeline.agentic_product_agent.litellm.completion").return_value.choices[0].message.content = json.dumps(
        {"canonical_id": "k", "canonical_name": "K", "category": "c",
         "search_terms": ["K"], "subreddits": ["amazondevices"]})
    agent = AgenticProductAgent(config=_FakeConfig(), client=FakeClient({}))
    state = {"raw_query": "Kindle", "iterations": 0, "history": [],
             "draft": ProductInfo("k", "K", "c", ["K"], ["kindle", "ebooks", "amazondevices"]),
             "validation": {"subreddits": {"kindle": 7, "ebooks": 4, "amazondevices": 0},
                            "noisy_terms": []}}
    out = agent._revise(state)
    assert "kindle" in out["draft"].subreddits      # productive kept
    assert "ebooks" in out["draft"].subreddits
    assert out["draft"].subreddits.index("kindle") < out["draft"].subreddits.index("amazondevices")


def test_revise_dedups_subreddits_case_insensitively(mocker):
    # Productive 'audiophile' + the LLM's 'Audiophile' must collapse to one.
    mocker.patch("pipeline.agentic_product_agent.litellm.completion").return_value.choices[0].message.content = json.dumps(
        {"canonical_id": "x", "canonical_name": "X", "category": "c",
         "search_terms": ["X"], "subreddits": ["Audiophile"]})
    agent = AgenticProductAgent(config=_FakeConfig(), client=FakeClient({}))
    state = {"raw_query": "X", "iterations": 0, "history": [],
             "draft": ProductInfo("x", "X", "c", ["X"], ["audiophile"]),
             "validation": {"subreddits": {"audiophile": 5}, "noisy_terms": []}}
    out = agent._revise(state)
    lowered = [s.lower() for s in out["draft"].subreddits]
    assert lowered.count("audiophile") == 1


def test_finalize_dedups_subreddits_case_insensitively():
    draft = ProductInfo("x", "X", "c", ["X"], ["NintendoSwitch", "nintendoswitch", "gaming"])
    state = {"draft": draft, "iterations": 1, "raw_query": "X", "history": [],
             "validation": {"subreddits": {"NintendoSwitch": 8, "nintendoswitch": 5, "gaming": 3},
                            "noisy_terms": []}}
    out = AgenticProductAgent(config=None, client=FakeClient({}))._finalize(state)
    lowered = [s.lower() for s in out["draft"].subreddits]
    assert lowered.count("nintendoswitch") == 1


def test_run_self_corrects_dead_subreddits(mocker):
    # propose gives a fake subreddit; revise gives two real ones; the graph should
    # loop once and finalize with the real subreddits, dropping the fake one.
    agent = AgenticProductAgent(config=_FakeConfig(), client=None)
    mocker.patch.object(agent, "_propose", return_value={
        "draft": ProductInfo("x", "X", "c", ["X"], ["FakeSub"]), "history": []})
    mocker.patch.object(agent, "_revise", return_value={
        "draft": ProductInfo("x", "X", "c", ["X"], ["realsub", "realsub2"]),
        "iterations": 1, "history": []})

    def fake_validate(state):
        subs = {s: (0 if s == "FakeSub" else 5) for s in state["draft"].subreddits}
        return {"validation": {"subreddits": subs, "noisy_terms": []}, "history": []}
    mocker.patch.object(agent, "_validate", side_effect=fake_validate)

    result = agent.run("X")
    assert "realsub" in result.subreddits
    assert "FakeSub" not in result.subreddits


def test_run_terminates_at_cap(mocker):
    # the LLM is stubborn — always the same dead subreddit. Must not hang; must
    # finalize non-empty (graceful degradation).
    agent = AgenticProductAgent(config=_FakeConfig(), client=None)
    mocker.patch.object(agent, "_propose", return_value={
        "draft": ProductInfo("x", "X", "c", ["X"], ["FakeSub"]), "history": []})
    mocker.patch.object(agent, "_revise", side_effect=lambda s: {
        "draft": ProductInfo("x", "X", "c", ["X"], ["FakeSub"]),
        "iterations": s["iterations"] + 1, "history": []})
    mocker.patch.object(agent, "_validate", return_value={
        "validation": {"subreddits": {"FakeSub": 0}, "noisy_terms": []}, "history": []})

    result = agent.run("X")
    assert result.subreddits   # never empty
