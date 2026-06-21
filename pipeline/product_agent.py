import json
import litellm
from .types import ProductInfo
from .config import AppConfig


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
        return ProductInfo(
            canonical_id=data["canonical_id"],
            canonical_name=data["canonical_name"],
            category=data["category"],
            search_terms=data["search_terms"],
            subreddits=data["subreddits"],
        )
