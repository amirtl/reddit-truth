import json
import litellm
from .types import Cluster, QuantifiedAspect, AspectSummary
from .config import AppConfig


class Summarizer:
    def __init__(self, config: AppConfig):
        self.model = config.llms.summarization

    def run(self, aspects: list[QuantifiedAspect], clusters: list[Cluster]) -> list[AspectSummary]:
        cluster_map = {c.label: c for c in clusters}
        return [self._summarize(aspect, cluster_map.get(aspect.label)) for aspect in aspects]

    def _summarize(self, aspect: QuantifiedAspect, cluster: Cluster | None) -> AspectSummary:
        prompt = self._build_prompt(aspect, cluster)
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return AspectSummary(
            label=aspect.label,
            mention_pct=aspect.mention_pct,
            positive_pct=aspect.positive_pct,
            negative_pct=aspect.negative_pct,
            recent_trend=aspect.recent_trend,
            headline=data["headline"],
            detail=data["detail"],
            trend_note=data.get("trend_note", ""),
        )

    def _build_prompt(self, aspect: QuantifiedAspect, cluster: Cluster | None) -> str:
        quotes = [c.quote for c in (cluster.claims[:10] if cluster else [])]
        quotes_text = "\n".join(f"- {q}" for q in quotes) if quotes else "No quotes available."

        return f"""Summarize Reddit opinions about the product aspect: "{aspect.label}"

Data:
- {aspect.mention_pct}% of comments mention this aspect
- {aspect.positive_pct}% positive, {aspect.negative_pct}% negative
- Recent trend: {aspect.recent_trend}

Sample user quotes:
{quotes_text}

Return JSON with:
- headline: one punchy sentence summarizing the overall opinion (max 15 words)
- detail: 2-3 sentences describing what users actually say, citing specific patterns
- trend_note: one sentence about recent trend (empty string if trend is stable)"""
