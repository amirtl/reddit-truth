import json
import litellm
from .types import RawComment, AspectClaim, ProductInfo
from .config import AppConfig

BATCH_SIZE = 15


class AspectExtractor:
    def __init__(self, config: AppConfig):
        self.model = config.llms.aspect_extraction

    def run(self, comments: list[RawComment], product: ProductInfo | None = None) -> list[AspectClaim]:
        claims: list[AspectClaim] = []
        for i in range(0, len(comments), BATCH_SIZE):
            batch = comments[i : i + BATCH_SIZE]
            claims.extend(self._extract_batch(batch, product))
        return claims

    def _target_block(self, product: ProductInfo | None) -> str:
        """Anchor extraction to the target product. Reddit threads constantly
        COMPARE products (e.g. a QLED TV thread full of OLED talk); without this,
        competitors' opinions leak in as if they were the target's — which makes
        the results untrustworthy. So name the target and forbid other products."""
        if product is None:
            return ""
        aliases = ", ".join(product.search_terms[:5])
        return (
            f'The TARGET product is "{product.canonical_name}" (also called: {aliases}).\n'
            f'Extract aspects ONLY about the TARGET product. These threads often COMPARE '
            f'it to other products (competitors, alternatives, or different technologies). '
            f'If an opinion is about a DIFFERENT product — not the target — DO NOT extract '
            f'it. If a whole comment is about a different product, skip it entirely. Only '
            f'the target product\'s own aspects.\n\n'
        )

    def _extract_batch(self, comments: list[RawComment], product: ProductInfo | None = None) -> list[AspectClaim]:
        valid_ids = {c.id for c in comments}
        numbered = "\n".join(f"[{c.id}] {c.text}" for c in comments)
        prompt = f"""Extract product aspect claims from these Reddit comments.
{self._target_block(product)}For each opinion found, return: comment_id, aspect, sentiment (positive/negative/mixed), quote (short relevant excerpt, max 10 words).
Use CANONICAL aspect labels so the same concept always gets the same name: short lowercase noun phrases, consistent across comments. Prefer broad common aspects (e.g. "battery life", "sound quality", "noise cancellation", "comfort", "build quality", "price", "connectivity"). Expand abbreviations to their full form (e.g. "ANC" -> "noise cancellation"). Do not invent over-specific sub-aspects — use the parent aspect (e.g. "build quality", not "left hinge design").
A single comment can produce multiple claims if it mentions multiple aspects.

Comments:
{numbered}

Return JSON: {{"claims": [{{...}}, ...]}}"""

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        claims = (self._parse_claim(item, valid_ids) for item in data.get("claims", []))
        return [c for c in claims if c is not None]

    def _parse_claim(self, item, valid_ids: set[str]) -> AspectClaim | None:
        """Defensively parse one claim. Skip anything malformed, anything with a
        comment_id the model invented (not in the batch), and normalize an
        unknown sentiment to "mixed" so bad LLM output never crashes or pollutes
        the results."""
        if not isinstance(item, dict):
            return None
        raw_id = item.get("comment_id")
        comment_id = str(raw_id) if raw_id is not None else None  # models may return an int id
        aspect = item.get("aspect")
        if comment_id not in valid_ids or not aspect or not str(aspect).strip():
            return None
        sentiment = str(item.get("sentiment", "")).strip().lower()  # models may capitalize
        if sentiment not in ("positive", "negative", "mixed"):
            sentiment = "mixed"
        return AspectClaim(
            comment_id=comment_id,
            aspect=str(aspect).strip(),
            sentiment=sentiment,
            quote=str(item.get("quote", "")),
        )
