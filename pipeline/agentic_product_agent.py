import json
import operator
from typing import Annotated, TypedDict

import litellm

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


def _anchor(terms: list[str]) -> str:
    """The most specific term — used as the 'on-topic' yardstick for precision."""
    return max(terms, key=len) if terms else ""


class AgenticProductAgent:
    """Self-correcting product understanding as a LangGraph agent.

    propose -> validate -> (decide) -> revise -> ... -> finalize. The LLM drafts;
    deterministic tools (Pattern B) ground the draft against Arctic-Shift; the
    conditional edge loops until it converges or hits the iteration cap.
    """

    def __init__(self, config, client: ArcticShiftClient | None = None):
        self.config = config
        self.client = client or ArcticShiftClient()

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
            f'make them specific: {v["noisy_terms"]}.\n'
            'Return the same JSON schema: canonical_id, canonical_name, category, '
            'search_terms, subreddits (no r/ prefix). Return only JSON.'
        )
        resp = litellm.completion(
            model=self.config.llms.product_understanding,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        draft = parse_product_info(json.loads(resp.choices[0].message.content), state["raw_query"])
        return {"draft": draft, "iterations": state["iterations"] + 1,
                "history": [f"revised: {draft.subreddits}"]}

    def _validate(self, state: AgentState) -> dict:
        """Node: run both tools over the current draft, write findings + a log."""
        draft = state["draft"]
        subs = {s: validate_subreddit(self.client, s, draft.search_terms)
                for s in draft.subreddits}
        anchor = _anchor(draft.search_terms)
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
