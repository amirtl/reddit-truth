# Agentic Product Understanding — Design Spec

**Status:** approved design, pending implementation plan
**Branch:** `feat/agentic-understanding`
**Author:** Amir (with Claude)
**Date:** 2026-06-23

> This spec is also a **learning artifact** for LangGraph / agentic AI. Concepts
> are explained where they're used, with the interview question they map to.

---

## 1. Why — the bug this fixes

Today `ProductUnderstandingAgent.run()` is a **single LLM call**: query in → JSON
out → trust it. With local models this fails three ways (measured): it invents
subreddits that don't exist (r/HeadphoneGears → 0 posts), sometimes returns an
empty subreddit list, and emits generic/contaminated search terms ("Pro 2",
leaking "Sony WH-1000XM5" into a Steam Deck query). Because Arctic-Shift can only
search *within* subreddits, a bad subreddit list silently zeroes recall → "No
opinions found". A single call has **no way to notice or recover.**

An **agent** adds a loop: *propose → check against reality → reflect → fix →
repeat until good enough.* It grounds the LLM's guesses in real Arctic-Shift data.

## 2. What we're building (one paragraph)

A LangGraph agent that replaces the single-call product-understanding stage. It
drafts a `ProductInfo`, **validates** every proposed subreddit and search term
against Arctic-Shift using two tools, and if the draft is weak, feeds concrete
feedback back to the LLM to **revise** — looping (capped) until it has enough
productive subreddits and clean terms. It exposes the same `.run(raw_query) ->
ProductInfo` interface, so the rest of the pipeline is untouched.

## 3. LangGraph primer (mental model)

LangGraph builds an agent as a **state graph**: **nodes** (Python functions that
read/write a shared **state**) connected by **edges** (what runs next). The
feature that beats a plain `while` loop is the **conditional edge** — the graph
branches and *cycles* based on state, and LangGraph manages the state and cycles.

| Concept | Plain-English | Interview phrasing |
|---|---|---|
| **State** | a typed "clipboard" passed to every node; nodes return partial updates that get merged | "a `TypedDict`; each field's merge rule is a **reducer**" |
| **Node** | a function `state -> partial state` | "unit of work" |
| **Edge** | static "after A, run B" | — |
| **Conditional edge** | a function that reads state and *picks* the next node — this is the loop | "`add_conditional_edges`; how you get cycles" |

## 4. Architecture — the graph

```
START → propose ──► validate ──► [decide] ──► finalize → END
                       ▲                │
                       └──── revise ◄────┘     (loop, max N iterations)
```

- **propose** — LLM drafts `ProductInfo` (id, name, category, search_terms, subreddits).
- **validate** — runs both tools against Arctic-Shift; writes findings to state.
- **decide** *(conditional edge, not a node)* — enough productive subreddits AND
  clean terms? → `finalize`. Else if `iterations < MAX_ITERS` → `revise`. Else →
  `finalize` (graceful degradation).
- **revise** — LLM receives *concrete* feedback ("r/HeadphoneGears: 0 threads —
  drop; 'Pro 2' matched 8 off-topic titles — make specific") and proposes fixes;
  `iterations += 1`; edge loops back to `validate`.
- **finalize** — emit the validated `ProductInfo` (unchanged shape).

**Pattern embodied:** ReAct (reason→act) + **Reflection** (critique results, revise).

## 5. State schema & reducers

```python
from typing import Annotated, TypedDict
import operator
from pipeline.types import ProductInfo

class AgentState(TypedDict):
    raw_query: str
    draft: ProductInfo                       # overwrite: only the latest proposal matters
    validation: dict                          # overwrite: latest findings
    iterations: int                           # overwrite
    history: Annotated[list[str], operator.add]   # REDUCER: append, don't overwrite
```

**Reducer = the merge rule for a field.** Default is *overwrite* (new replaces
old). Annotating a field with a reducer (e.g. `operator.add`, or the built-in
`add_messages` for chat) makes updates *accumulate*. `history` accumulates a log
across nodes; `draft` overwrites (it's a dataclass — `operator.add` on it would
raise `TypeError`, the reducer must match the field's type).

> **Interview soundbite:** *"Each state field has a reducer; default overwrite,
> annotate with `operator.add`/`add_messages` to accumulate."*

## 6. The tools (plain Python functions over Arctic-Shift)

```python
def validate_subreddit(subreddit: str, terms: list[str]) -> int | None:
    """Count threads in r/<subreddit> whose TITLE matches the product.
    0 = real-but-empty/fake.  None = couldn't check (API error) — see §9."""

def check_term_precision(term: str, subreddits: list[str]) -> float:
    """Fraction of titles matching `term` that ALSO contain a strong product
    anchor (longest/most-specific term). Low = generic term like 'Pro 2'."""
```

Both reuse `ArcticShiftScraper`'s existing query + title-match logic (we'll
extract a tiny shared client so the tool and the scraper don't duplicate it).
`check_term_precision` needs no LLM: precision ≈ (titles matching the term **and**
a strong anchor) / (titles matching the term).

## 7. Tool invocation — Pattern A vs B (the `bind_tools`/`ToolNode` question)

- **Pattern A — LLM-driven (classic ReAct):** `llm.bind_tools([...])`; the LLM
  *emits* tool calls, a prebuilt **`ToolNode`** executes them, results return to
  the LLM, which decides what's next. Needs a **tool-calling-capable model**.
  Best when tool selection is open-ended.
- **Pattern B — graph-orchestrated (OUR CHOICE):** *we* call the tools inside
  `validate`, always, for every subreddit/term. The LLM keeps agency over
  *reasoning* (propose/revise); tool *invocation* is deterministic.

We always want to validate everything — there's no decision to delegate — so B
gives reliability, lower cost, and easy unit tests.

> **Interview answer to "which and when?":** *"LLM-driven when tool
> selection/sequence is open-ended; deterministic orchestration when you always
> need the same checks. Don't hand the LLM a decision it doesn't need."*

## 8. The model

`propose`/`revise` use **`ollama/gemma2:9b`** (won our 6-product benchmark for
naming real subreddits, 6/6). Because we chose Pattern B, the model needs **no
tool-calling support** — only good JSON, which gemma2 does well. The agent =
gemma2's world knowledge **+** Arctic-Shift grounding to catch its mistakes.

## 9. Error handling & confidence

- **Tool/API error → return `None` ("unknown"), never `0`.** A network hiccup
  must not make the agent drop a *real* subreddit. Retry/backoff first (scraper
  already does); if still failing, treat as unknown and don't penalize it.
- **Malformed LLM JSON** in propose/revise → reuse the existing defensive parser.
- **Max iterations reached** → finalize with the best draft (graceful
  degradation) and record a low **confidence** in `history`/logs.
- **Zero productive subreddits after all tries** → one last broaden attempt, then
  finalize low-confidence rather than crash.
- **Cost/latency guard:** cache validation results within a run (don't re-check a
  subreddit across iterations), cap breadth (top-N subreddits/terms), reuse the
  scraper's request delay/rate-limit.

(Confidence is logged this iteration; surfacing it to the UI is a non-goal — §13.)

## 10. Testing strategy (how you test a non-deterministic agent)

**Isolate the determinism.** Inject the LLM and the tools so they can be mocked.
- **Conditional edge as a pure function:** craft states, assert `decide()` returns
  `finalize`/`revise`/(cap)`finalize`.
- **Each node with mocked LLM + mocked tools:** deterministic.
- **Whole compiled graph with mocks:** assert it loops once on a bad first draft,
  terminates at the cap, finalizes on a good draft — testing **control flow**.
- **A couple of integration tests** hit the real Arctic-Shift tool (contract).

> **Interview soundbite:** *"Graph logic is deterministic → unit-test with mocked
> LLM+tools. LLM output *quality* is non-deterministic → measure it with an eval
> harness (project C), don't unit-test it. And remember: mocks test my
> assumptions, not reality — hence the integration tests."*

## 11. Integration (keep the pipeline untouched)

- Add a `ProductAgent` **Protocol** to `pipeline/ports.py` (mirrors the existing
  `Scraper` port): `run(self, raw_query: str) -> ProductInfo`.
- New module `pipeline/agentic_product_agent.py` holds the graph, nodes, tools.
- `tasks/pipeline_task.py` `build_runner()` swaps `ProductUnderstandingAgent` for
  the new agent — one line. `runner.py` is unchanged (still calls `.run()`).
- The old `ProductUnderstandingAgent` stays as the `propose`/`revise` LLM step
  (reused, not deleted) and as portfolio evidence of the before/after.

## 12. Dependencies

- `langgraph` (and its small `langchain-core` dep) added to `requirements.txt`.
- No vector DB, no LangSmith yet (those are projects C). YAGNI.

## 13. Scope / non-goals (YAGNI)

- **In:** two-tool validation + self-correction loop, deterministic orchestration,
  graceful degradation, full unit tests.
- **Out (later):** LLM-driven tool-calling (Pattern A), product disambiguation,
  surfacing confidence to the UI, LangSmith tracing (project C), parallel tool
  calls unless latency demands it.

## 14. Success criteria

- Across the 6-product benchmark, **no empty subreddit lists**, and **≥2
  productive subreddits per product** (single-call baseline managed 3/6 products
  with *any* result).
- Generic terms (e.g. "Pro 2") get flagged and revised away.
- Graph logic fully covered by deterministic unit tests; ≥1 live integration test.
- `.run()` interface unchanged → rest of pipeline and all existing tests still pass.

## 15. Interview concept cheat-sheet

StateGraph · State as TypedDict · **reducers** (overwrite vs `operator.add`/
`add_messages`) · nodes · edges · **conditional edges** (cycles) · ReAct ·
**Reflection** · `bind_tools` + **ToolNode** (Pattern A) vs deterministic
orchestration (Pattern B) · tool-calling models · grounding/self-correction ·
**graceful degradation** + iteration cap + no-progress detection · **0 vs unknown**
on tool error · dependency injection for testability · "mocks test assumptions,
not reality."
