from __future__ import annotations

import asyncio
import concurrent.futures
import multiprocessing as mp
from typing import Callable, Optional

import numpy as np


class EmbeddingEngine:
    """Local ONNX embedding backend (FastEmbed) with async thread-pool execution.

    Model files are downloaded once and cached. Vectors are L2-normalised so a
    single matmul yields cosine scores.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", dim: int = 384) -> None:
        self.model_name = model_name
        self.dim = dim
        self._model = None
        self._embed_lock = asyncio.Semaphore(2)
        self._query_cache: dict[str, np.ndarray] = {}

    def _ensure_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _embed_sync(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        vectors = np.asarray(list(model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        async with self._embed_lock:
            return await asyncio.to_thread(self._embed_sync, texts)

    async def embed_query(self, text: str) -> np.ndarray:
        cached = self._query_cache.get(text)
        if cached is not None:
            return cached
        vec = await self.embed([text])
        if len(self._query_cache) < 4096:
            self._query_cache[text] = vec[0]
        return vec[0]

    @property
    def loaded(self) -> bool:
        return self._model is not None


def _worker_init(model_name: str, threads: int) -> None:
    global _WORKER_MODEL
    from fastembed import TextEmbedding

    _WORKER_MODEL = TextEmbedding(model_name=model_name, threads=threads)


def _worker_embed(texts: list[str]) -> np.ndarray:
    global _WORKER_MODEL
    vectors = np.asarray(list(_WORKER_MODEL.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_parallel(
    texts: list[str],
    model_name: str,
    dim: int,
    workers: int = 4,
    chunk_size: int = 256,
    local_embed: Optional[Callable[[list[str]], np.ndarray]] = None,
) -> np.ndarray:
    """Embed many texts across a process pool (index-time path).

    Each worker owns a fastembed session pinned to a single thread, so the pool
    scales linearly with cores instead of thrashing. ``local_embed`` is used as
    the single-process fallback when ``workers <= 1``.
    """
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    if workers <= 1:
        if local_embed is not None:
            return np.asarray(local_embed(texts), dtype=np.float32)
        return _worker_embed(texts)
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(model_name, 1),
    ) as pool:
        results = list(pool.map(_worker_embed, chunks))
    return np.concatenate(results, axis=0)
