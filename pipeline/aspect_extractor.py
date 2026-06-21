import json
import litellm
from .types import RawComment, AspectClaim
from .config import AppConfig

BATCH_SIZE = 15


class AspectExtractor:
    def __init__(self, config: AppConfig):
        self.model = config.llms.aspect_extraction

    def run(self, comments: list[RawComment]) -> list[AspectClaim]:
        claims: list[AspectClaim] = []
        for i in range(0, len(comments), BATCH_SIZE):
            batch = comments[i : i + BATCH_SIZE]
            claims.extend(self._extract_batch(batch))
        return claims

    def _extract_batch(self, comments: list[RawComment]) -> list[AspectClaim]:
        numbered = "\n".join(f"[{c.id}] {c.text}" for c in comments)
        prompt = f"""Extract product aspect claims from these Reddit comments.
For each opinion found, return: comment_id, aspect, sentiment (positive/negative/mixed), quote (short relevant excerpt, max 10 words).
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
        return [
            AspectClaim(
                comment_id=item["comment_id"],
                aspect=item["aspect"],
                sentiment=item["sentiment"],
                quote=item["quote"],
            )
            for item in data.get("claims", [])
        ]
