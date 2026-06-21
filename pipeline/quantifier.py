from datetime import datetime, timedelta, timezone
from .types import RawComment, Cluster, QuantifiedAspect

RECENT_DAYS = 90
TREND_THRESHOLD = 0.10
# An aspect mentioned by fewer than this many distinct comments isn't a signal —
# one person's remark is noise, not a trend. Drops the singleton-aspect tail.
MIN_MENTIONS = 2


class Quantifier:
    def run(self, clusters: list[Cluster], comments: list[RawComment]) -> list[QuantifiedAspect]:
        total = len(comments)
        comment_map = {c.id: c for c in comments}
        recent_cutoff = datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)

        aspects = [
            self._quantify(cluster, total, comment_map, recent_cutoff)
            for cluster in clusters
            if self._unique_mentions(cluster) >= MIN_MENTIONS
        ]
        return sorted(aspects, key=lambda a: a.mention_pct, reverse=True)

    def _unique_mentions(self, cluster: Cluster) -> int:
        return len({claim.comment_id for claim in cluster.claims})

    def _quantify(
        self,
        cluster: Cluster,
        total: int,
        comment_map: dict[str, RawComment],
        recent_cutoff: datetime,
    ) -> QuantifiedAspect:
        mention_pct = self._mention_pct(cluster, total)
        positive_pct, negative_pct = self._sentiment_pcts(cluster)
        recent_trend = self._trend(cluster, comment_map, recent_cutoff, positive_pct)

        return QuantifiedAspect(
            label=cluster.label,
            mention_pct=round(mention_pct, 1),
            positive_pct=round(positive_pct, 1),
            negative_pct=round(negative_pct, 1),
            recent_trend=recent_trend,
        )

    def _mention_pct(self, cluster: Cluster, total: int) -> float:
        if total == 0:
            return 0.0
        unique_comment_ids = {claim.comment_id for claim in cluster.claims}
        return len(unique_comment_ids) / total * 100

    def _sentiment_pcts(self, cluster: Cluster) -> tuple[float, float]:
        total = len(cluster.claims)
        if total == 0:
            return 0.0, 0.0
        positive_pct = cluster.positive_count / total * 100
        negative_pct = cluster.negative_count / total * 100
        return positive_pct, negative_pct

    def _trend(
        self,
        cluster: Cluster,
        comment_map: dict[str, RawComment],
        recent_cutoff: datetime,
        all_time_positive_pct: float,
    ) -> str:
        recent_claims = [
            c for c in cluster.claims
            if c.comment_id in comment_map
            and comment_map[c.comment_id].created_at.astimezone(timezone.utc) >= recent_cutoff
        ]
        if not recent_claims:
            return "stable"

        recent_pos = sum(1 for c in recent_claims if c.sentiment == "positive")
        recent_pos_pct = recent_pos / len(recent_claims)
        delta = recent_pos_pct - (all_time_positive_pct / 100)

        if delta > TREND_THRESHOLD:
            return "improving"
        if delta < -TREND_THRESHOLD:
            return "declining"
        return "stable"
