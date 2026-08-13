from __future__ import annotations

import asyncio
from typing import Callable, Optional

import numpy as np


def _kmeans(
    data: np.ndarray,
    n_clusters: int,
    seed: int = 42,
    max_iter: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    if n_clusters >= n:
        return data.copy(), np.arange(n)

    idx = rng.choice(n, n_clusters, replace=False)
    centers = data[idx].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        sims = centers @ data.T
        new_labels = np.argmax(sims, axis=0).astype(np.int32)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        new_centers = centers.copy()
        for c in range(n_clusters):
            members = data[labels == c]
            if len(members):
                new_centers[c] = members.mean(axis=0)
        centers = new_centers
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centers = centers / norms
    return centers, labels


class InMemoryVectorDB:
    """In-memory IVF-style vector store over normalised float32 vectors.

    Build-time k-means creates coarse clusters; search probes the top ``probe``
    clusters by centroid similarity and runs an exact dot-product inside them,
    then applies an optional metadata filter. This is a real approximate
    nearest-neighbour pattern, not a brute-force scan.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.metadata: list[dict] = []
        self.centroids: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self._lock = asyncio.Lock()

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        if vectors.shape[0] == 0:
            return
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        self.vectors = np.concatenate([self.vectors, vectors.astype(np.float32)], axis=0)
        self.metadata.extend(metadata)

    def build_index(self, n_clusters: Optional[int] = None, seed: int = 42) -> dict:
        n = self.vectors.shape[0]
        if n_clusters is None:
            n_clusters = int(max(32, min(1024, round(n**0.5))))
        if self.centroids is None:
            self.centroids, self.labels = _kmeans(self.vectors, n_clusters, seed=seed)
        return {
            "num_vectors": n,
            "num_clusters": n_clusters if self.centroids is not None else 0,
            "dim": self.dim,
        }

    def _search_sync(
        self,
        query: np.ndarray,
        top_k: int,
        probe: int,
        filter_fn: Optional[Callable[[dict], bool]],
    ) -> list[tuple[float, int]]:
        query = query.reshape(1, -1)
        if self.centroids is None:
            scores = (self.vectors @ query.T).ravel()
            order = np.argsort(-scores)[: top_k * 4]
            hits: list[tuple[float, int]] = []
            for i in order:
                if filter_fn is not None and not filter_fn(self.metadata[i]):
                    continue
                hits.append((float(scores[i]), int(i)))
                if len(hits) >= top_k:
                    break
            return hits

        probe = min(probe, self.centroids.shape[0])
        centroid_scores = (self.centroids @ query.T).ravel()
        probe_clusters = np.argsort(-centroid_scores)[:probe]
        candidates = np.concatenate([np.flatnonzero(self.labels == c) for c in probe_clusters])
        if candidates.size == 0:
            return []
        scores = (self.vectors[candidates] @ query.T).ravel()
        order = np.argsort(-scores)
        hits: list[tuple[float, int]] = []
        for rank in order:
            i = int(candidates[rank])
            if filter_fn is not None and not filter_fn(self.metadata[i]):
                continue
            hits.append((float(scores[rank]), i))
            if len(hits) >= top_k:
                break
        return hits

    async def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        probe: int = 8,
        filter_fn: Optional[Callable[[dict], bool]] = None,
        return_indices: bool = False,
    ) -> list:
        if self.vectors.shape[0] == 0:
            return []
        async with self._lock:
            hits = await asyncio.to_thread(self._search_sync, query, top_k, probe, filter_fn)
        if return_indices:
            return [(score, self.metadata[i], i) for score, i in hits]
        return [(score, self.metadata[i]) for score, i in hits]
