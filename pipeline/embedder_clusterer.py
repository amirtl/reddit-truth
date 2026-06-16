import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MeanShift, estimate_bandwidth
from .types import AspectClaim, Cluster
from .config import AppConfig


class EmbedderClusterer:
    def __init__(self, config: AppConfig):
        self.model = SentenceTransformer(config.embeddings.model)

    def run(self, claims: list[AspectClaim]) -> list[Cluster]:
        if not claims:
            return []

        texts = [f"{c.aspect}: {c.quote}" for c in claims]
        embeddings = self.model.encode(texts)

        bandwidth = estimate_bandwidth(
            embeddings, quantile=0.3, n_samples=min(len(embeddings), 500)
        )
        if bandwidth == 0:
            bandwidth = 0.5

        labels = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(embeddings)

        cluster_map: dict[int, list[AspectClaim]] = {}
        for claim, label in zip(claims, labels):
            cluster_map.setdefault(int(label), []).append(claim)

        clusters = []
        for cluster_claims in cluster_map.values():
            aspects = [c.aspect for c in cluster_claims]
            label = max(set(aspects), key=aspects.count)
            positive = sum(1 for c in cluster_claims if c.sentiment == "positive")
            negative = sum(1 for c in cluster_claims if c.sentiment == "negative")
            clusters.append(Cluster(
                label=label,
                claims=cluster_claims,
                positive_count=positive,
                negative_count=negative,
            ))

        return sorted(clusters, key=lambda c: len(c.claims), reverse=True)
