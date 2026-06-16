import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from .types import AspectClaim, Cluster
from .config import AppConfig


class EmbedderClusterer:
    def __init__(self, config: AppConfig):
        self.model = SentenceTransformer(config.embeddings.model)

    def run(self, claims: list[AspectClaim]) -> list[Cluster]:
        if not claims:
            return []
        if len(claims) == 1:
            c = claims[0]
            pos = 1 if c.sentiment == "positive" else 0
            neg = 1 if c.sentiment == "negative" else 0
            return [Cluster(label=c.aspect, claims=claims, positive_count=pos, negative_count=neg)]

        texts = [f"{c.aspect}: {c.quote}" for c in claims]
        embeddings = self.model.encode(texts)

        min_cluster_size = max(2, int(len(claims) * 0.01))
        labels = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric="cosine",
            cluster_selection_method="eom",
            allow_single_cluster=True,
        ).fit_predict(embeddings)

        # reassign noise points (label == -1) to their nearest cluster
        if -1 in labels and (labels != -1).any():
            noise_mask = labels == -1
            core_mask = ~noise_mask
            nn = NearestNeighbors(n_neighbors=1, metric="cosine")
            nn.fit(embeddings[core_mask])
            _, indices = nn.kneighbors(embeddings[noise_mask])
            labels[noise_mask] = labels[core_mask][indices.flatten()]

        # if everything is noise (all -1), put everything in one cluster
        if (labels == -1).all():
            labels = np.zeros(len(claims), dtype=int)

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
