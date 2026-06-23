# Agentic Product Understanding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-call product-understanding stage with a self-correcting LangGraph agent that grounds its subreddit/term picks against Arctic-Shift.

**Architecture:** A LangGraph `StateGraph` with nodes `propose → validate → (decide) → revise → … → finalize`. `validate` runs two deterministic tools (Pattern B) against Arctic-Shift; the conditional edge `decide` loops back to `revise` until there are enough productive subreddits and clean terms, or an iteration cap is hit. Exposes `.run(raw_query) -> ProductInfo`, so the pipeline is untouched.

**Tech Stack:** Python 3.12, LangGraph, LiteLLM (gemma2:9b via Ollama), requests, pytest + pytest-mock.

## Global Constraints

- Agent LLM stage uses `ollama/gemma2:9b` (from `config.llms.product_understanding`); no tool-calling model required (Pattern B).
- Tools must return `None` on API error (never `0`) so a hiccup never drops a real subreddit.
- `AgenticProductAgent.run(self, raw_query: str) -> ProductInfo` — interface identical to today's agent; the rest of the pipeline and all existing tests must keep passing.
- TDD: test first, watch it fail, minimal impl, watch it pass, commit. Frequent commits.
- Reuse Arctic-Shift query + alias-matching logic; do not duplicate it.

---

### Task 1: Arctic-Shift query client (shared low-level search)

**Files:**
- Create: `pipeline/arctic_shift_client.py`
- Test: `tests/test_arctic_shift_client.py`

**Interfaces:**
- Produces:
  - `class ArcticShiftClient(base_url=..., user_agent="reddit-truth/0.1", request_delay=0.5, max_retries=3)`
  - `post_titles(self, subreddit: str, query: str, limit: int = 10) -> list[str] | None` — titles of posts; `[]` = genuinely none; `None` = API error after retries.
  - `alias_matcher(terms: list[str]) -> re.Pattern | None` — staticmethod, word-bounded OR of escaped terms (so "XM5" ≠ "XM500").

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arctic_shift_client.py
from pipeline.arctic_shift_client import ArcticShiftClient


def test_post_titles_returns_titles(mocker):
    resp = mocker.Mock()
    resp.json.return_value = {"data": [{"title": "Sony WH-1000XM5 review"}, {"title": "x"}]}
    mocker.patch("pipeline.arctic_shift_client.requests.get", return_value=resp)
    client = ArcticShiftClient(request_delay=0)
    assert client.post_titles("headphones", "XM5") == ["Sony WH-1000XM5 review", "x"]


def test_post_titles_returns_none_on_persistent_error(mocker):
    mocker.patch("pipeline.arctic_shift_client.requests.get", side_effect=Exception("boom"))
    client = ArcticShiftClient(request_delay=0, max_retries=2)
    assert client.post_titles("headphones", "XM5") is None


def test_alias_matcher_is_word_bounded():
    m = ArcticShiftClient.alias_matcher(["XM5"])
    assert m.search("my XM5 is great")
    assert not m.search("the XM500 model")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_arctic_shift_client.py -v`
Expected: FAIL — `ModuleNotFoundError: pipeline.arctic_shift_client`.

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/arctic_shift_client.py
import re
import time

import requests


class ArcticShiftClient:
    """Low-level Arctic-Shift search. Returns post titles, or None on error so
    callers can tell 'genuinely empty' (drop the subreddit) from 'couldn't
    check' (don't punish it)."""

    def __init__(self, base_url: str = "https://arctic-shift.photon-reddit.com/api",
                 user_agent: str = "reddit-truth/0.1",
                 request_delay: float = 0.5, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.headers = {"User-Agent": user_agent}
        self.request_delay = request_delay
        self.max_retries = max_retries

    def post_titles(self, subreddit: str, query: str, limit: int = 10) -> list[str] | None:
        for attempt in range(self.max_retries):
            try:
                r = requests.get(self.base_url + "/posts/search", params={
                    "subreddit": subreddit, "query": query, "sort": "desc", "limit": limit,
                }, headers=self.headers, timeout=30)
                payload = r.json()
                if isinstance(payload, dict) and payload.get("error"):
                    time.sleep(self.request_delay * (attempt + 1))
                    continue
                data = payload.get("data") if isinstance(payload, dict) else payload
                time.sleep(self.request_delay)
                return [p.get("title", "") for p in (data or [])]
            except (requests.RequestException, ValueError):
                time.sleep(self.request_delay * (attempt + 1))
        return None

    @staticmethod
    def alias_matcher(terms: list[str]) -> re.Pattern | None:
        parts = [re.escape(t) for t in terms if t]
        if not parts:
            return None
        return re.compile(rf"\b(?:{'|'.join(parts)})\b", re.IGNORECASE)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_arctic_shift_client.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/arctic_shift_client.py tests/test_arctic_shift_client.py
git commit -m "feat: add ArcticShiftClient — low-level search, None on error"
```

---

### Task 2: `validate_subreddit` tool

**Files:**
- Create: `pipeline/agent_tools.py`
- Test: `tests/test_agent_tools.py`

**Interfaces:**
- Consumes: `ArcticShiftClient.post_titles`, `ArcticShiftClient.alias_matcher`.
- Produces: `validate_subreddit(client, subreddit: str, terms: list[str], max_terms: int = 3) -> int | None` — count of unique title-matched posts; `None` if every query errored.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_tools.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: pipeline.agent_tools`.

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/agent_tools.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_tools.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/agent_tools.py tests/test_agent_tools.py
git commit -m "feat: add validate_subreddit tool (None on error, never 0)"
```

---

### Task 3: `check_term_precision` tool

**Files:**
- Modify: `pipeline/agent_tools.py`
- Test: `tests/test_agent_tools.py`

**Interfaces:**
- Produces: `check_term_precision(client, term: str, anchor: str, subreddits: list[str], max_subs: int = 3) -> float` — fraction of titles matching `term` that ALSO contain `anchor` (the most-specific term). `1.0` when there's no evidence (don't flag blindly). Low → generic term.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_tools.py  (append)
from pipeline.agent_tools import check_term_precision


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_tools.py -v`
Expected: FAIL — `ImportError: cannot import name 'check_term_precision'`.

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/agent_tools.py  (append)
import re


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_tools.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/agent_tools.py tests/test_agent_tools.py
git commit -m "feat: add check_term_precision tool (flags generic terms)"
```

---

### Task 4: Shared `parse_product_info` helper (DRY for revise)

**Files:**
- Modify: `pipeline/product_agent.py`
- Test: `tests/test_product_agent.py`

**Interfaces:**
- Produces: `parse_product_info(data: dict, raw_query: str) -> ProductInfo` — the existing defensive parsing, extracted to module level so `revise` can reuse it. `ProductUnderstandingAgent.run` now calls it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_product_agent.py  (append)
from pipeline.product_agent import parse_product_info


def test_parse_product_info_keeps_query_and_drops_empty():
    info = parse_product_info({"canonical_name": "X", "search_terms": [], "subreddits": ["a", ""]}, "My Query")
    assert "My Query" in info.search_terms          # raw query always present
    assert info.subreddits == ["a"]                  # empties dropped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_product_agent.py::test_parse_product_info_keeps_query_and_drops_empty -v`
Expected: FAIL — `ImportError: cannot import name 'parse_product_info'`.

- [ ] **Step 3: Write minimal implementation**

Extract the parsing currently inside `ProductUnderstandingAgent.run` into a module-level function and call it. In `pipeline/product_agent.py`, add:

```python
def parse_product_info(data: dict, raw_query: str) -> ProductInfo:
    canonical_name = data.get("canonical_name") or raw_query
    search_terms = [t for t in (data.get("search_terms") or []) if isinstance(t, str) and t.strip()]
    if raw_query not in search_terms:
        search_terms.insert(0, raw_query)
    subreddits = [s for s in (data.get("subreddits") or []) if isinstance(s, str) and s.strip()]
    return ProductInfo(
        canonical_id=data.get("canonical_id") or _slug(canonical_name),
        canonical_name=canonical_name,
        category=data.get("category") or "",
        search_terms=search_terms,
        subreddits=subreddits,
    )
```

Then replace the body after `data = json.loads(...)` in `run` with:

```python
        return parse_product_info(data, raw_query)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_product_agent.py -v`
Expected: PASS (all existing product-agent tests + the new one).

- [ ] **Step 5: Commit**

```bash
git add pipeline/product_agent.py tests/test_product_agent.py
git commit -m "refactor: extract parse_product_info for reuse by the agent"
```

---

### Task 5: `decide` conditional-edge logic (pure function)

**Files:**
- Create: `pipeline/agentic_product_agent.py`
- Test: `tests/test_agentic_product_agent.py`

**Interfaces:**
- Produces:
  - constants `MAX_ITERS = 2`, `MIN_PRODUCTIVE = 2`, `NOISY_TERM_THRESHOLD = 0.3`
  - `AgentState` TypedDict: `raw_query: str`, `draft: ProductInfo`, `validation: dict`, `iterations: int`, `history: Annotated[list[str], operator.add]`
  - `decide(state: AgentState) -> str` → `"finalize"` or `"revise"`. `validation` shape: `{"subreddits": {name: int|None}, "noisy_terms": [str]}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agentic_product_agent.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agentic_product_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: pipeline.agentic_product_agent`.

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/agentic_product_agent.py
import operator
from typing import Annotated, TypedDict

from pipeline.types import ProductInfo

MAX_ITERS = 2
MIN_PRODUCTIVE = 2
NOISY_TERM_THRESHOLD = 0.3


class AgentState(TypedDict):
    raw_query: str
    draft: ProductInfo
    validation: dict
    iterations: int
    history: Annotated[list[str], operator.add]


def decide(state: AgentState) -> str:
    subs = state["validation"]["subreddits"]
    productive = [s for s, n in subs.items() if isinstance(n, int) and n > 0]
    clean = not state["validation"]["noisy_terms"]
    if len(productive) >= MIN_PRODUCTIVE and clean:
        return "finalize"
    if state["iterations"] >= MAX_ITERS:
        return "finalize"
    return "revise"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agentic_product_agent.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/agentic_product_agent.py tests/test_agentic_product_agent.py
git commit -m "feat: add agent state + decide conditional-edge logic"
```

---

### Task 6: `validate` and `finalize` nodes

**Files:**
- Modify: `pipeline/agentic_product_agent.py`
- Test: `tests/test_agentic_product_agent.py`

**Interfaces:**
- Produces (methods on a new `AgenticProductAgent`, constructed with an injected `client` and `config`):
  - `_validate(self, state) -> dict` → `{"validation": {...}, "history": [...]}`
  - `_finalize(self, state) -> dict` → `{"draft": <draft with validated subreddits, productive first>}`
- Consumes: `validate_subreddit`, `check_term_precision`, `NOISY_TERM_THRESHOLD`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agentic_product_agent.py  (append)
from pipeline.agentic_product_agent import AgenticProductAgent
from pipeline.types import ProductInfo


class FakeClient:
    def __init__(self, sub_counts, titles_by_term=None):
        self.sub_counts = sub_counts            # {subreddit: count} via post_titles
        self.titles_by_term = titles_by_term or {}
    def post_titles(self, subreddit, query, limit=10):
        # validate_subreddit path: synthesize N matching titles for the term
        if query in self.titles_by_term:
            return self.titles_by_term[query]
        return [f"{query} thread"] * self.sub_counts.get(subreddit, 0)


def _agent(client):
    return AgenticProductAgent(config=None, client=client)


def test_validate_flags_dead_subreddits_and_noisy_terms(mocker):
    draft = ProductInfo("x", "AirPods Pro 2", "earbuds",
                        ["AirPods Pro 2", "Pro 2"], ["apple", "fake"])
    client = FakeClient(
        sub_counts={"apple": 5, "fake": 0},
        titles_by_term={"Pro 2": ["AirPods Pro 2 x", "Surface Pro 2", "iPad Pro 2"]},
    )
    agent = _agent(client)
    out = agent._validate({"draft": draft, "validation": {}, "iterations": 0,
                           "raw_query": "AirPods Pro 2", "history": []})
    assert out["validation"]["subreddits"]["fake"] == 0
    assert out["validation"]["subreddits"]["apple"] > 0
    assert "Pro 2" in out["validation"]["noisy_terms"]


def test_finalize_keeps_productive_subreddits_first():
    draft = ProductInfo("x", "X", "c", ["X"], ["fake", "apple"])
    state = {"draft": draft,
             "validation": {"subreddits": {"fake": 0, "apple": 8}, "noisy_terms": []},
             "iterations": 1, "raw_query": "X", "history": []}
    out = _agent(FakeClient({}))._finalize(state)
    assert out["draft"].subreddits[0] == "apple"   # productive first
    assert "fake" not in out["draft"].subreddits    # dead dropped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agentic_product_agent.py -v`
Expected: FAIL — `AttributeError`/`TypeError`: `AgenticProductAgent` has no `_validate`/`_finalize`.

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/agentic_product_agent.py  (append imports + class)
from pipeline.agent_tools import validate_subreddit, check_term_precision
from pipeline.arctic_shift_client import ArcticShiftClient


def _anchor(terms: list[str]) -> str:
    return max(terms, key=len) if terms else ""


class AgenticProductAgent:
    def __init__(self, config, client: ArcticShiftClient | None = None):
        self.config = config
        self.client = client or ArcticShiftClient()

    def _validate(self, state: AgentState) -> dict:
        draft = state["draft"]
        subs = {s: validate_subreddit(self.client, s, draft.search_terms)
                for s in draft.subreddits}
        anchor = _anchor(draft.search_terms)
        noisy = [t for t in draft.search_terms
                 if t != anchor and
                 check_term_precision(self.client, t, anchor, draft.subreddits) < NOISY_TERM_THRESHOLD]
        dead = [s for s, n in subs.items() if n == 0]
        log = []
        if dead:
            log.append(f"dead subreddits: {dead}")
        if noisy:
            log.append(f"noisy terms: {noisy}")
        return {"validation": {"subreddits": subs, "noisy_terms": noisy}, "history": log}

    def _finalize(self, state: AgentState) -> dict:
        draft = state["draft"]
        subs = state["validation"]["subreddits"]
        # keep productive (n>0) and unknown (None); drop validated-dead (0);
        # order productive first
        productive = [s for s in draft.subreddits if isinstance(subs.get(s), int) and subs[s] > 0]
        unknown = [s for s in draft.subreddits if subs.get(s) is None]
        kept = productive + unknown
        draft.subreddits = kept or draft.subreddits  # never emit empty
        return {"draft": draft}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agentic_product_agent.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/agentic_product_agent.py tests/test_agentic_product_agent.py
git commit -m "feat: add validate + finalize agent nodes"
```

---

### Task 7: `propose` and `revise` nodes (LLM)

**Files:**
- Modify: `pipeline/agentic_product_agent.py`
- Test: `tests/test_agentic_product_agent.py`

**Interfaces:**
- Produces:
  - `_propose(self, state) -> dict` → `{"draft": ProductInfo, "history": ["proposed: ..."]}` (uses `ProductUnderstandingAgent`).
  - `_revise(self, state) -> dict` → `{"draft": ProductInfo, "iterations": state["iterations"]+1, "history": ["revised: ..."]}` (LLM with feedback prompt, parsed via `parse_product_info`).
- Consumes: `ProductUnderstandingAgent`, `parse_product_info`, `litellm`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agentic_product_agent.py  (append)
import json


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
    from pipeline.types import ProductInfo
    agent = AgenticProductAgent(config=_FakeConfig(), client=FakeClient({}))
    state = {"raw_query": "P", "iterations": 0, "history": [],
             "draft": ProductInfo("p", "P", "c", ["P"], ["bad"]),
             "validation": {"subreddits": {"bad": 0}, "noisy_terms": []}}
    out = agent._revise(state)
    assert out["iterations"] == 1
    assert out["draft"].subreddits == ["good"]


class _FakeConfig:
    class llms:
        product_understanding = "ollama/gemma2:9b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agentic_product_agent.py -k "propose or revise" -v`
Expected: FAIL — no `_propose`/`_revise`.

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/agentic_product_agent.py  (append imports + methods)
import litellm
from pipeline.product_agent import ProductUnderstandingAgent, parse_product_info


# inside AgenticProductAgent:
    def _propose(self, state: AgentState) -> dict:
        draft = ProductUnderstandingAgent(self.config).run(state["raw_query"])
        return {"draft": draft, "history": [f"proposed: {draft.subreddits}"]}

    def _revise(self, state: AgentState) -> dict:
        v = state["validation"]
        dead = [s for s, n in v["subreddits"].items() if n == 0]
        prompt = (
            f'Improve product understanding for: "{state["raw_query"]}".\n'
            f'These subreddits had ZERO product threads (they may be wrong or fake) '
            f'— replace them with real, active subreddits: {dead}.\n'
            f'These search terms are too generic (they matched off-topic posts) '
            f'— make them specific: {v["noisy_terms"]}.\n'
            'Return the same JSON schema: canonical_id, canonical_name, category, '
            'search_terms, subreddits (no r/ prefix). Return only JSON.'
        )
        resp = litellm.completion(
            model=self.config.llms.product_understanding,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        import json
        draft = parse_product_info(json.loads(resp.choices[0].message.content), state["raw_query"])
        return {"draft": draft, "iterations": state["iterations"] + 1,
                "history": [f"revised: {draft.subreddits}"]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agentic_product_agent.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/agentic_product_agent.py tests/test_agentic_product_agent.py
git commit -m "feat: add propose + revise LLM nodes"
```

---

### Task 8: Assemble the LangGraph graph + `run()`

**Files:**
- Modify: `pipeline/agentic_product_agent.py`, `requirements.txt`
- Test: `tests/test_agentic_product_agent.py`

**Interfaces:**
- Produces: `AgenticProductAgent.run(self, raw_query: str) -> ProductInfo` — compiles the StateGraph and invokes it.

- [ ] **Step 1: Add dependency and install**

Add to `requirements.txt`:
```
langgraph
```
Run: `uv pip install langgraph` (or `pip install langgraph`).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_agentic_product_agent.py  (append)
def test_run_self_corrects_dead_subreddits(mocker):
    # propose returns a fake subreddit; revise returns a real one; graph should
    # loop once and finalize with the real subreddit.
    from pipeline.types import ProductInfo
    agent = AgenticProductAgent(config=_FakeConfig(), client=None)
    mocker.patch.object(agent, "_propose", side_effect=[{
        "draft": ProductInfo("x", "X", "c", ["X"], ["FakeSub"]), "history": []}])
    mocker.patch.object(agent, "_revise", side_effect=[{
        "draft": ProductInfo("x", "X", "c", ["X"], ["realsub"]),
        "iterations": 1, "history": []}])
    # validate: FakeSub -> 0, realsub -> 5
    def fake_validate(state):
        subs = {s: (0 if s == "FakeSub" else 5) for s in state["draft"].subreddits}
        return {"validation": {"subreddits": subs, "noisy_terms": []}, "history": []}
    mocker.patch.object(agent, "_validate", side_effect=fake_validate)

    result = agent.run("X")
    assert "realsub" in result.subreddits
    assert "FakeSub" not in result.subreddits


def test_run_terminates_at_cap(mocker):
    from pipeline.types import ProductInfo
    agent = AgenticProductAgent(config=_FakeConfig(), client=None)
    bad = {"draft": ProductInfo("x", "X", "c", ["X"], ["FakeSub"]), "history": []}
    mocker.patch.object(agent, "_propose", return_value=bad)
    mocker.patch.object(agent, "_revise", side_effect=lambda s: {
        "draft": ProductInfo("x", "X", "c", ["X"], ["FakeSub"]),
        "iterations": s["iterations"] + 1, "history": []})
    mocker.patch.object(agent, "_validate", return_value={
        "validation": {"subreddits": {"FakeSub": 0}, "noisy_terms": []}, "history": []})
    result = agent.run("X")            # must not hang
    assert result.subreddits           # never empty (graceful degradation)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_agentic_product_agent.py -k run -v`
Expected: FAIL — `AgenticProductAgent` has no `run`.

- [ ] **Step 4: Write minimal implementation**

```python
# pipeline/agentic_product_agent.py  (append)
from langgraph.graph import StateGraph, START, END


# inside AgenticProductAgent.__init__, after setting self.client:
        self._graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("propose", self._propose)
        g.add_node("validate", self._validate)
        g.add_node("revise", self._revise)
        g.add_node("finalize", self._finalize)
        g.add_edge(START, "propose")
        g.add_edge("propose", "validate")
        g.add_conditional_edges("validate", decide,
                                {"revise": "revise", "finalize": "finalize"})
        g.add_edge("revise", "validate")
        g.add_edge("finalize", END)
        return g.compile()

    def run(self, raw_query: str) -> ProductInfo:
        final = self._graph.invoke({"raw_query": raw_query, "iterations": 0, "history": []})
        return final["draft"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_agentic_product_agent.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add pipeline/agentic_product_agent.py requirements.txt tests/test_agentic_product_agent.py
git commit -m "feat: assemble LangGraph graph + AgenticProductAgent.run"
```

---

### Task 9: Wire the agent into the pipeline (port + build_runner)

**Files:**
- Modify: `pipeline/ports.py`, `tasks/pipeline_task.py`
- Test: `tests/test_pipeline_task.py` (or wherever `build_runner` is covered)

**Interfaces:**
- Produces: `ProductAgent` Protocol (`run(self, raw_query: str) -> ProductInfo`) in `pipeline/ports.py`; `build_runner` constructs `AgenticProductAgent(config)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_task.py  (append; create if absent)
from pipeline.agentic_product_agent import AgenticProductAgent
from pipeline.config import load_config
from tasks.pipeline_task import build_runner


def test_build_runner_uses_agentic_product_agent():
    runner = build_runner(load_config())
    assert isinstance(runner.product_agent, AgenticProductAgent)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_task.py::test_build_runner_uses_agentic_product_agent -v`
Expected: FAIL — `product_agent` is a `ProductUnderstandingAgent`.

- [ ] **Step 3: Write minimal implementation**

In `pipeline/ports.py` add:
```python
class ProductAgent(Protocol):
    """Resolve a raw query into a validated ProductInfo."""
    def run(self, raw_query: str) -> ProductInfo: ...
```
In `tasks/pipeline_task.py`: replace the import and construction:
```python
from pipeline.agentic_product_agent import AgenticProductAgent
# ...
        product_agent=AgenticProductAgent(config),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline_task.py -v`
Expected: PASS.

- [ ] **Step 5: Run the FULL suite (no regressions)**

Run: `pytest tests/ -q`
Expected: all pass (existing + new).

- [ ] **Step 6: Commit**

```bash
git add pipeline/ports.py tasks/pipeline_task.py tests/test_pipeline_task.py
git commit -m "feat: wire AgenticProductAgent into the pipeline via ProductAgent port"
```

---

### Task 10: Live integration test (real Arctic-Shift, marked slow)

**Files:**
- Test: `tests/test_agent_tools_integration.py`

**Interfaces:**
- Consumes: real `ArcticShiftClient`, `validate_subreddit`.

- [ ] **Step 1: Write the integration test**

```python
# tests/test_agent_tools_integration.py
import pytest
from pipeline.arctic_shift_client import ArcticShiftClient
from pipeline.agent_tools import validate_subreddit


@pytest.mark.integration
def test_real_subreddit_has_threads_fake_one_does_not():
    client = ArcticShiftClient()
    real = validate_subreddit(client, "headphones", ["Sony WH-1000XM5", "WH-1000XM5"])
    fake = validate_subreddit(client, "HeadphoneGears", ["Sony WH-1000XM5"])
    assert real and real > 0          # real subreddit yields threads
    assert fake == 0                   # fabricated subreddit yields none
```

- [ ] **Step 2: Register the marker**

In `pytest.ini` add under `[pytest]`:
```
markers =
    integration: hits live external APIs (slow, network-dependent)
```

- [ ] **Step 3: Run it explicitly**

Run: `pytest -m integration -v`
Expected: PASS (network required). Default `pytest tests/ -q` should still pass; document running integration separately.

- [ ] **Step 4: Commit**

```bash
git add tests/test_agent_tools_integration.py pytest.ini
git commit -m "test: live Arctic-Shift integration test for subreddit validation"
```

---

### Task 11: End-to-end live verification + docs

**Files:**
- Modify: `README.md` (note the agentic stage), optionally `config.yml` comment.

- [ ] **Step 1: Bring up the stack and run a real query**

```bash
# Redis, Ollama (gemma2:9b pulled), Postgres up; then:
make worker            # Celery worker (solo pool)
make run               # Django
# submit a previously-failing product:
curl -s -X POST http://localhost:8000/api/jobs/ -H 'Content-Type: application/json' \
  -d '{"query":"Logitech MX Master 3S"}'
```
Poll `GET /api/jobs/<id>/` until `done`; confirm non-empty subreddits and on-topic aspects (the agent should drop fabricated subreddits and keep productive ones).

- [ ] **Step 2: Update README**

Add a short "Agentic product understanding" subsection describing the propose→validate→revise loop and that it grounds subreddit/term choices against Arctic-Shift.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: describe agentic product-understanding stage"
```

---

## Self-Review

**Spec coverage:**
- §3–4 graph (nodes/edges/conditional) → Tasks 5,6,7,8 ✓
- §5 state + reducers → Task 5 (`AgentState`, `history` reducer) ✓
- §6 tools → Tasks 1,2,3 ✓
- §7 Pattern B (deterministic tools in `validate`) → Task 6 ✓
- §8 gemma2 model → Task 7 (`config.llms.product_understanding`) ✓
- §9 error handling (`None` not `0`, graceful degradation, never-empty) → Tasks 1,2,5,6,8 ✓
- §10 testing (pure decide, mocked nodes, whole-graph, integration) → Tasks 5,8,10 ✓
- §11 integration (ProductAgent port, build_runner, old agent reused) → Tasks 4,7,9 ✓
- §12 deps (langgraph) → Task 8 ✓
- §14 success criteria → Task 11 live check + unit assertions ✓

**Placeholder scan:** none — every code/test step has concrete content.

**Type consistency:** `ProductInfo` shape consistent; `validate_subreddit -> int | None`, `check_term_precision -> float`, `decide -> str`, `AgenticProductAgent.run -> ProductInfo`, `validation` dict shape `{"subreddits": {...}, "noisy_terms": [...]}` used consistently across Tasks 5–8.
