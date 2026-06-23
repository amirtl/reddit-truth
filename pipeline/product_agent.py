import json
import re
import litellm
from .types import ProductInfo
from .config import AppConfig


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "product"


# Words that mark a "term" as a descriptive phrase/question rather than an
# identifier — these title-match almost nothing and starve recall.
_PHRASE_WORDS = {
    "review", "reviews", "vs", "versus", "best", "worth", "problem", "problems",
    "issue", "issues", "game", "games", "alternative", "alternatives", "compared",
    "comparison", "deal", "deals", "cheap", "buy", "for", "good", "bad",
}


def refine_search_terms(terms: list[str]) -> list[str]:
    """Make terms recall-friendly for title matching.

    1. Extract bare model tokens (letter+digit, e.g. WH-1000XM5, V15, M3) and put
       them FIRST — a bare model number matches far more titles than the full
       "Brand Model" phrase, and the scraper only queries the first few terms.
    2. Drop descriptive phrases (>4 words or containing a phrase word) that match
       nothing. Dedupe, preserving order.
    """
    model_tokens: list[str] = []
    kept: list[str] = []
    for t in terms:
        words = t.split()
        lowered = {w.lower().strip("()") for w in words}
        if not (len(words) > 4 or (lowered & _PHRASE_WORDS)):
            kept.append(t)
        for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", t):
            if any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
                model_tokens.append(tok)
    seen: set[str] = set()
    out: list[str] = []
    for t in model_tokens + kept:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def parse_product_info(data: dict, raw_query: str) -> ProductInfo:
    """Defensive parse of the LLM's JSON into a ProductInfo. Shared by the
    single-call agent and the agentic revise node, so both apply the same
    guards: keep the raw query as a search term, drop empty entries, never crash."""
    canonical_name = data.get("canonical_name") or raw_query
    search_terms = [t for t in (data.get("search_terms") or []) if isinstance(t, str) and t.strip()]
    search_terms = refine_search_terms(search_terms)
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


class ProductUnderstandingAgent:
    def __init__(self, config: AppConfig):
        self.model = config.llms.product_understanding

    def run(self, raw_query: str) -> ProductInfo:
        prompt = f"""Given this product query: "{raw_query}"

Return a JSON object with exactly these keys:
- canonical_id: kebab-case product identifier (e.g. "sony-wh-1000xm5")
- canonical_name: full official product name
- category: product category (e.g. "headphones", "laptop", "camera")
- search_terms: 3-5 SPECIFIC identifiers that uniquely name THIS exact product —
  the full model name, the bare model number, and common abbreviations or
  nicknames. For "Sony WH-1000XM5" that is ["Sony WH-1000XM5", "WH-1000XM5",
  "XM5", "Sony XM5"]. Do NOT include generic category terms like "wireless
  headphones" or "noise-cancelling headphones" — they match unrelated products
  and ruin precision. Every term must distinguish this product from its siblings.
- subreddits: list of 3-5 relevant subreddits (without r/ prefix)

Return only valid JSON, no explanation, no markdown."""

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return parse_product_info(data, raw_query)
