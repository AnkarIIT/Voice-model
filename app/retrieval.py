import logging
import threading
import time

from .config import RERANK_DEVICE, RERANK_MODEL
from .embed import resolve_device

logger = logging.getLogger(__name__)

_CE = None
_CE_LOCK = threading.Lock()


def _get_cross_encoder():
    global _CE
    if _CE is None:
        with _CE_LOCK:
            if _CE is None:
                from sentence_transformers import CrossEncoder

                device = resolve_device(RERANK_DEVICE)
                logger.info("loading reranker %s on %s", RERANK_MODEL, device)
                _CE = CrossEncoder(RERANK_MODEL, device=device)
    return _CE


def rerank(query_text: str, results: list, top_k: int) -> list:
    try:
        ce = _get_cross_encoder()
        pairs = [[query_text, r["text"]] for r in results]
        scores = ce.predict(pairs)
        for r, s in zip(results, scores):
            r["ce_score"] = float(s)
        return sorted(results, key=lambda x: x["ce_score"], reverse=True)[:top_k]
    except Exception as e:
        logger.warning("rerank failed (%s); using FAISS order", e)
        return results[:top_k]


def retrieve(query_text: str, index, k: int = 5, use_rerank: bool = False) -> dict:
    t0 = time.perf_counter()
    results, search_ms = index.search(query_text, k * 2 if use_rerank else k)
    if use_rerank and len(results) > 1:
        results = rerank(query_text, results, k)
    else:
        results = results[:k]
    total = (time.perf_counter() - t0) * 1000
    return {
        "query": query_text,
        "results": results,
        "search_ms": round(search_ms, 2),
        "total_ms": round(total, 2),
        "k": k,
        "reranked": bool(use_rerank and len(results) > 1),
    }
