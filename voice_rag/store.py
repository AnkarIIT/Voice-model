from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .bm25 import BM25Index
from .models import RetrievedChunk
from .vectordb import InMemoryVectorDB

_RRF_K = 60


def _rrf_rank(dense_order: list[int], bm25_order: list[int], top_k: int) -> list[int]:
    """Reciprocal-rank fusion of dense and BM25 candidate rankings."""
    dense_rank = {doc: rank + 1 for rank, doc in enumerate(dense_order)}
    bm25_rank = {doc: rank + 1 for rank, doc in enumerate(bm25_order)}
    union = list(dict.fromkeys(dense_order + bm25_order))
    fused = sorted(
        union,
        key=lambda doc: 1.0 / (_RRF_K + dense_rank.get(doc, _RRF_K * 10))
        + 1.0 / (_RRF_K + bm25_rank.get(doc, _RRF_K * 10)),
        reverse=True,
    )
    return fused[:top_k]


class VectorStoreCollection:
    """Two-tier store: fine-grained children for retrieval, wide parents for context.

    Both layers are searched concurrently and results are merged by ``source_id``
    (a parent), keeping the highest score. This is the parent-child retrieval
    pattern: precise match on children, generous context for generation.

    Retrieval is hybrid: an IVF-dense ranking is fused with a BM25 lexical
    ranking via reciprocal-rank fusion, which is markedly more robust on
    noisy multi-lingual factoid corpora than either signal alone.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.children = InMemoryVectorDB(dim)
        self.parents = InMemoryVectorDB(dim)
        self.bm25: Optional[BM25Index] = None

    def add_child(self, vector: np.ndarray, meta: dict) -> None:
        self.children.add(vector.reshape(1, -1), [meta])

    def add_parent(self, vector: np.ndarray, meta: dict) -> None:
        self.parents.add(vector.reshape(1, -1), [meta])

    def build_index(self, n_clusters_child: Optional[int] = None, n_clusters_parent: Optional[int] = None, seed: int = 42) -> dict:
        return {
            "children": self.children.build_index(n_clusters_child, seed=seed),
            "parents": self.parents.build_index(n_clusters_parent, seed=seed),
        }

    async def search(
        self,
        query: np.ndarray,
        query_text: Optional[str] = None,
        child_top_k: int = 6,
        parent_top_k: int = 3,
        probe: int = 8,
        filter_fn: Optional[Callable[[dict], bool]] = None,
        hybrid: bool = True,
        dense_pool: int = 100,
    ) -> list[RetrievedChunk]:
        if hybrid and self.bm25 is not None and query_text:
            child_hits, parent_hits = await asyncio.gather(
                self._hybrid_children(query, query_text, child_top_k, probe, filter_fn, dense_pool),
                self.parents.search(query, parent_top_k, probe, filter_fn),
            )
            merged: dict[str, tuple[float, dict]] = {}
            for score, meta in child_hits + parent_hits:
                key = meta.get("source_id") or meta.get("chunk_id")
                if key is None:
                    continue
                if key not in merged or score > merged[key][0]:
                    merged[key] = (score, meta)
            ranked = list(merged.values())
        else:
            child_hits, parent_hits = await asyncio.gather(
                self.children.search(query, child_top_k, probe, filter_fn),
                self.parents.search(query, parent_top_k, probe, filter_fn),
            )
            merged = {}
            for score, meta in child_hits + parent_hits:
                key = meta.get("source_id") or meta.get("chunk_id")
                if key is None:
                    continue
                if key not in merged or score > merged[key][0]:
                    merged[key] = (score, meta)
            ranked = sorted(merged.values(), key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=meta.get("chunk_id", ""),
                text=meta.get("source_text", "") or meta.get("text", ""),
                language=meta.get("language", ""),
                source_kind=meta.get("source_kind", ""),
                selected=bool(meta.get("selected", False)),
                score=round(score, 4),
            )
            for score, meta in ranked
        ]

    async def _hybrid_children(
        self,
        query: np.ndarray,
        query_text: str,
        child_top_k: int,
        probe: int,
        filter_fn: Optional[Callable[[dict], bool]],
        dense_pool: int,
    ) -> list[tuple[float, dict]]:
        dense = await self.children.search(query, dense_pool, probe, None, return_indices=True)
        dense_order = [idx for _, _, idx in dense]
        dense_scores = {idx: score for score, _, idx in dense}

        bm25_top = self.bm25.top(query_text, dense_pool)
        bm25_order = [idx for _, idx in bm25_top]
        bm25_scores = dict(bm25_top)

        final = _rrf_rank(dense_order, bm25_order, child_top_k)
        qv = query.reshape(-1)
        hits: list[tuple[float, dict]] = []
        for idx in final:
            meta = self.children.metadata[idx]
            if filter_fn is not None and not filter_fn(meta):
                continue
            if idx in dense_scores:
                score = dense_scores[idx]
            else:
                score = float(self.children.vectors[idx] @ qv)
            hits.append((score, meta))
        return hits

    @property
    def counts(self) -> dict:
        return {
            "children": int(self.children.vectors.shape[0]),
            "parents": int(self.parents.vectors.shape[0]),
        }

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "vectors.npz",
            child_vectors=self.children.vectors,
            parent_vectors=self.parents.vectors,
            child_centroids=self.children.centroids if self.children.centroids is not None else np.zeros((0, self.dim), np.float32),
            parent_centroids=self.parents.centroids if self.parents.centroids is not None else np.zeros((0, self.dim), np.float32),
            child_labels=self.children.labels if self.children.labels is not None else np.zeros(0, np.int32),
            parent_labels=self.parents.labels if self.parents.labels is not None else np.zeros(0, np.int32),
            **({"bm25_rows": self.bm25.rows, "bm25_data": self.bm25.data, "bm25_col_ptr": self.bm25.col_ptr} if self.bm25 else {}),
        )
        meta: dict = {
            "child_meta": self.children.metadata,
            "parent_meta": self.parents.metadata,
            "dim": self.dim,
        }
        if self.bm25 is not None:
            meta.update(
                {
                    "bm25_vocab": self.bm25.vocab,
                    "bm25_idf": self.bm25.idf.tolist(),
                    "bm25_doc_len": self.bm25.doc_len.tolist(),
                    "bm25_n_docs": self.bm25.n_docs,
                    "bm25_avgdl": self.bm25.avgdl,
                }
            )
        (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "VectorStoreCollection":
        arrays = np.load(directory / "vectors.npz")
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        store = cls(int(meta["dim"]))
        store.children.vectors = arrays["child_vectors"]
        store.children.metadata = meta["child_meta"]
        store.parents.vectors = arrays["parent_vectors"]
        store.parents.metadata = meta["parent_meta"]
        if arrays["child_centroids"].shape[0]:
            store.children.centroids = arrays["child_centroids"]
            store.children.labels = arrays["child_labels"]
        if arrays["parent_centroids"].shape[0]:
            store.parents.centroids = arrays["parent_centroids"]
            store.parents.labels = arrays["parent_labels"]
        if "bm25_rows" in arrays:
            store.bm25 = BM25Index(
                vocab=meta["bm25_vocab"],
                idf=np.asarray(meta["bm25_idf"], dtype=np.float32),
                rows=arrays["bm25_rows"],
                data=arrays["bm25_data"],
                col_ptr=arrays["bm25_col_ptr"],
                doc_len=np.asarray(meta["bm25_doc_len"], dtype=np.int32),
                n_docs=int(meta["bm25_n_docs"]),
                avgdl=float(meta["bm25_avgdl"]),
            )
        return store
