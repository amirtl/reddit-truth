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
            return [self._build_cluster(claims)]

        embeddings = self._embed(claims)
        labels = self._cluster(embeddings)
        labels = self._reassign_noise(embeddings, labels, len(claims))
        return self._build_clusters(claims, labels)

    def _embed(self, claims: list[AspectClaim]) -> np.ndarray:
        texts = [f"{c.aspect}: {c.quote}" for c in claims]
        return self.model.encode(texts)

    def _cluster(self, embeddings: np.ndarray) -> np.ndarray:
        min_cluster_size = max(2, int(len(embeddings) * 0.01))
        return HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric="cosine",
            cluster_selection_method="eom",
            allow_single_cluster=True,
        ).fit_predict(embeddings)

    def _reassign_noise(self, embeddings: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
        if (labels == -1).all():
            return np.zeros(n, dtype=int)
        if -1 not in labels:
            return labels
        noise_mask = labels == -1
        core_mask = ~noise_mask
        nn = NearestNeighbors(n_neighbors=1, metric="cosine")
        nn.fit(embeddings[core_mask])
        _, indices = nn.kneighbors(embeddings[noise_mask])
        labels[noise_mask] = labels[core_mask][indices.flatten()]
        return labels

    def _build_clusters(self, claims: list[AspectClaim], labels: np.ndarray) -> list[Cluster]:
        cluster_map: dict[int, list[AspectClaim]] = {}
        for claim, label in zip(claims, labels):
            cluster_map.setdefault(int(label), []).append(claim)
        clusters = [self._build_cluster(cluster_claims) for cluster_claims in cluster_map.values()]
        return sorted(clusters, key=lambda c: len(c.claims), reverse=True)

    def _build_cluster(self, claims: list[AspectClaim]) -> Cluster:
        aspects = [c.aspect for c in claims]
        return Cluster(
            label=max(set(aspects), key=aspects.count),
            claims=claims,
            positive_count=sum(1 for c in claims if c.sentiment == "positive"),
            negative_count=sum(1 for c in claims if c.sentiment == "negative"),
        )
