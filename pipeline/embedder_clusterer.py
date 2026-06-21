import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from .types import AspectClaim, Cluster
from .config import AppConfig

# Cosine distance below which two aspect labels are treated as the same concept.
# Tuned on real labels: merges variants like "hinge design"/"durability / hinge"
# while keeping distinct aspects (sound vs build quality, battery vs ANC) apart.
MERGE_DISTANCE = 0.4


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
        return self._build_clusters(claims, labels)

    def _embed(self, claims: list[AspectClaim]) -> np.ndarray:
        # Cluster on the aspect label, not the quote. This step's job is to merge
        # synonymous aspects ("battery"/"battery life"); the quote is opinion
        # content whose shared vocabulary ("worth the price") otherwise drags
        # unrelated aspects into one cluster and collapses the breakdown.
        texts = [c.aspect for c in claims]
        return self.model.encode(texts)

    def _cluster(self, embeddings: np.ndarray) -> np.ndarray:
        # Agglomerative with a cosine distance threshold: directly expresses
        # "merge aspect labels closer than MERGE_DISTANCE". Every point lands in
        # a cluster (no noise), so a unique aspect keeps its own cluster instead
        # of being force-attached to an unrelated one.
        return AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=MERGE_DISTANCE,
        ).fit_predict(embeddings)

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
