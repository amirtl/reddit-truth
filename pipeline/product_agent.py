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
- search_terms: list of 3-5 search terms to find Reddit discussions
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
