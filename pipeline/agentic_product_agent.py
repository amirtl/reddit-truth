import json
import operator
from typing import Annotated, TypedDict

import litellm
from langgraph.graph import END, START, StateGraph

from pipeline.agent_tools import check_term_precision, validate_subreddit
from pipeline.arctic_shift_client import ArcticShiftClient
from pipeline.product_agent import ProductUnderstandingAgent, parse_product_info
from pipeline.types import ProductInfo

MAX_ITERS = 2
MIN_PRODUCTIVE = 2
NOISY_TERM_THRESHOLD = 0.3


class AgentState(TypedDict):
    """The clipboard passed to every node. Every field overwrites on update,
    EXCEPT `history`, whose reducer (operator.add) appends instead — so the log
    accumulates across nodes rather than each node clobbering the last."""
    raw_query: str
    draft: ProductInfo
    validation: dict           # {"subreddits": {name: int|None}, "noisy_terms": [str]}
    iterations: int
    history: Annotated[list[str], operator.add]


def decide(state: AgentState) -> str:
    """Conditional-edge function: read state, return the next node's name.
    Pure and deterministic, so it's unit-tested without the graph or an LLM."""
    subs = state["validation"]["subreddits"]
    # `isinstance(n, int)` excludes None (couldn't-check) from "productive".
    productive = [s for s, n in subs.items() if isinstance(n, int) and n > 0]
    clean = not state["validation"]["noisy_terms"]
    if len(productive) >= MIN_PRODUCTIVE and clean:
        return "finalize"
    if state["iterations"] >= MAX_ITERS:     # termination guard → graceful degradation
        return "finalize"
    return "revise"


class AgenticProductAgent:
    """Self-correcting product understanding as a LangGraph agent.

    propose -> validate -> (decide) -> revise -> ... -> finalize. The LLM drafts;
    deterministic tools (Pattern B) ground the draft against Arctic-Shift; the
    conditional edge loops until it converges or hits the iteration cap.
    """

    def __init__(self, config, client: ArcticShiftClient | None = None):
        self.config = config
        self.client = client or ArcticShiftClient()
        self._graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(AgentState)
        # Lambda indirection so dispatch happens at runtime (and patches in tests
        # take effect) rather than capturing the bound method at build time.
        g.add_node("propose", lambda s: self._propose(s))
        g.add_node("validate", lambda s: self._validate(s))
        g.add_node("revise", lambda s: self._revise(s))
        g.add_node("finalize", lambda s: self._finalize(s))
        g.add_edge(START, "propose")
        g.add_edge("propose", "validate")
        g.add_conditional_edges("validate", decide,
                                {"revise": "revise", "finalize": "finalize"})
        g.add_edge("revise", "validate")        # the loop back (the cycle)
        g.add_edge("finalize", END)
        return g.compile()

    def run(self, raw_query: str) -> ProductInfo:
        final = self._graph.invoke({"raw_query": raw_query, "iterations": 0, "history": []})
        return final["draft"]

    def _propose(self, state: AgentState) -> dict:
        """Node: the LLM drafts a first ProductInfo (the old single-call agent)."""
        draft = ProductUnderstandingAgent(self.config).run(state["raw_query"])
        return {"draft": draft, "history": [f"proposed: {draft.subreddits}"]}

    def _revise(self, state: AgentState) -> dict:
        """Node: reflection — feed the validation failures back to the LLM so it
        replaces dead subreddits and sharpens generic terms. Ticks iterations."""
        v = state["validation"]
        dead = [s for s, n in v["subreddits"].items() if n == 0]
        prompt = (
            f'Improve product understanding for: "{state["raw_query"]}".\n'
            f'These subreddits had ZERO product threads (likely wrong or fake) — '
            f'replace them with real, active subreddits: {dead}.\n'
            f'These search terms are too generic (they matched off-topic posts) — '
            f'replace them with SHORT, matchable identifiers: the bare model number '
            f'or a common abbreviation (e.g. "WH-1000XM5", "XM5"), NOT descriptive '
            f'phrases or questions: {v["noisy_terms"]}.\n'
            'Return the same JSON schema: canonical_id, canonical_name, category, '
            'search_terms, subreddits (no r/ prefix). Return only JSON.'
        )
        resp = litellm.completion(
            model=self.config.llms.product_understanding,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        draft = parse_product_info(json.loads(resp.choices[0].message.content), state["raw_query"])
        # Keep subreddits already validated as productive — the LLM often returns
        # a fresh list that drops them, losing hard-won progress (Kindle's
        # intermittent 0-comment runs). Productive ones go first.
        productive = [s for s, n in v["subreddits"].items() if isinstance(n, int) and n > 0]
        draft.subreddits = list(dict.fromkeys(productive + draft.subreddits))
        return {"draft": draft, "iterations": state["iterations"] + 1,
                "history": [f"revised: {draft.subreddits}"]}

    def _validate(self, state: AgentState) -> dict:
        """Node: run both tools over the current draft, write findings + a log."""
        draft = state["draft"]
        subs = {s: validate_subreddit(self.client, s, draft.search_terms)
                for s in draft.subreddits}
        # Anchor on the user's raw query — the ground-truth specific identifier.
        # (Using the longest term over-flagged good terms against a rarer, more
        # specific variant, wasting revise loops.)
        anchor = state["raw_query"]
        noisy = [t for t in draft.search_terms
                 if t != anchor
                 and check_term_precision(self.client, t, anchor, draft.subreddits) < NOISY_TERM_THRESHOLD]
        dead = [s for s, n in subs.items() if n == 0]
        log = []
        if dead:
            log.append(f"dead subreddits: {dead}")
        if noisy:
            log.append(f"noisy terms: {noisy}")
        return {"validation": {"subreddits": subs, "noisy_terms": noisy}, "history": log}

    def _finalize(self, state: AgentState) -> dict:
        """Node: emit the validated draft — keep productive (n>0) and unknown
        (None, couldn't-check) subreddits, productive first; drop validated-dead
        (0). Never emit an empty list."""
        draft = state["draft"]
        subs = state["validation"]["subreddits"]
        productive = [s for s in draft.subreddits if isinstance(subs.get(s), int) and subs[s] > 0]
        unknown = [s for s in draft.subreddits if subs.get(s) is None]
        draft.subreddits = (productive + unknown) or draft.subreddits
        return {"draft": draft}
