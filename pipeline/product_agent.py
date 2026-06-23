import json
import re
import litellm
from .types import ProductInfo
from .config import AppConfig


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "product"


def parse_product_info(data: dict, raw_query: str) -> ProductInfo:
    """Defensive parse of the LLM's JSON into a ProductInfo. Shared by the
    single-call agent and the agentic revise node, so both apply the same
    guards: keep the raw query as a search term, drop empty entries, never crash."""
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
