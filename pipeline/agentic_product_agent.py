import operator
from typing import Annotated, TypedDict

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
