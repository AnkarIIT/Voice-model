from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .bm25 import BM25Index
from .chunking import ChunkNode, adaptive_chunk
from .config import Settings, get_settings
from .embeddings import EmbeddingEngine, embed_parallel
from .store import VectorStoreCollection

TRANSLATED_LANG = "hi"
ENGLISH_LANG = "en"


def load_rows(parquet_path: Path, max_rows: int, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    n = table.num_rows
    if max_rows >= n:
        return table.to_pylist()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_rows, replace=False))
    return table.take(idx).to_pylist()


def _passages_of(row: dict) -> list[dict]:
    raw = row.get("passages")
    if not raw:
        return []
    if isinstance(raw, dict):
        english = raw.get("English_passages") or []
        trans = raw.get("Translated_passages") or []
        selected = raw.get("is_selected") or []
        out = []
        for i, (en, tr) in enumerate(zip(english, trans)):
            out.append(
                {
                    "en": en or "",
                    "hi": tr or "",
                    "selected": bool(selected[i]) if i < len(selected) else False,
                }
            )
        return out
    out = []
    for item in raw:
        out.append(
            {
                "en": item.get("English_passages") or "",
                "hi": item.get("Translated_passages") or "",
                "selected": bool(item.get("is_selected", False)),
            }
        )
    return out


def build_corpus(rows: list[dict]) -> list[dict]:
    docs: list[dict] = []
    for row in rows:
        qid = int(row.get("query_id", -1))
        for i, p in enumerate(_passages_of(row)):
            if p["hi"].strip():
                docs.append(
                    {
                        "text": p["hi"],
                        "language": TRANSLATED_LANG,
                        "source_kind": "translated_passage",
                        "source_query_id": qid,
                        "passage_idx": i,
                        "selected": p["selected"],
                    }
                )
            if p["en"].strip():
                docs.append(
                    {
                        "text": p["en"],
                        "language": ENGLISH_LANG,
                        "source_kind": "english_passage",
                        "source_query_id": qid,
                        "passage_idx": i,
                        "selected": p["selected"],
                    }
                )
        if row.get("Answer"):
            docs.append(
                {
                    "text": row["Answer"],
                    "language": TRANSLATED_LANG,
                    "source_kind": "answer",
                    "source_query_id": qid,
                    "passage_idx": None,
                    "selected": True,
                }
            )
        if row.get("Eng_Answer"):
            docs.append(
                {
                    "text": row["Eng_Answer"],
                    "language": ENGLISH_LANG,
                    "source_kind": "answer",
                    "source_query_id": qid,
                    "passage_idx": None,
                    "selected": True,
                }
            )
    return docs


def sync_embed_fn(engine: EmbeddingEngine) -> Callable[[list[str]], np.ndarray]:
    def _embed(texts: list[str]) -> np.ndarray:
        return engine._embed_sync(texts)

    return _embed


async def build_index(settings: Settings) -> dict:
    started = time.perf_counter()
    engine = EmbeddingEngine(settings.embedding_model, settings.embedding_dim)
    engine._ensure_model()
    embed_fn = sync_embed_fn(engine)

    print(f"[indexer] loading sample from {settings.index_parquet}", flush=True)
    rows = load_rows(settings.index_parquet, settings.index_max_rows, settings.index_seed)
    docs = build_corpus(rows)
    print(f"[indexer] sampled {len(rows)} rows -> {len(docs)} documents", flush=True)

    store = VectorStoreCollection(settings.embedding_dim)

    def meta_of(node: ChunkNode, parent_text: Optional[str] = None, is_parent: bool = False) -> dict:
        return {
            "chunk_id": node.chunk_id,
            "source_id": node.parent_id or node.chunk_id,
            "source_text": parent_text if parent_text is not None else node.text,
            "text": node.text,
            "language": node.language,
            "source_kind": node.source_kind,
            "source_query_id": node.source_query_id,
            "passage_idx": node.passage_idx,
            "selected": node.selected,
            "word_count": node.word_count,
            "strategy": node.strategy,
            "is_parent": is_parent,
        }

    parent_nodes: list[ChunkNode] = []
    child_nodes: list[ChunkNode] = []
    chunk_started = time.perf_counter()
    for doc in docs:
        parents, children = adaptive_chunk(
            doc["text"],
            embed_fn,
            language=doc["language"],
            source_kind=doc["source_kind"],
            source_query_id=doc["source_query_id"],
            passage_idx=doc["passage_idx"],
            selected=doc["selected"],
        )
        parent_nodes.extend(parents)
        child_nodes.extend(children)
    print(
        f"[indexer] chunked {len(docs)} docs in {time.perf_counter() - chunk_started:.1f}s "
        f"(parents={len(parent_nodes)}, children={len(child_nodes)})",
        flush=True,
    )

    parent_text_by_id = {p.chunk_id: p.text for p in parent_nodes}

    embed_started = time.perf_counter()
    embeddable_parents = [p for p in parent_nodes if p.strategy != "atomic"]
    if embeddable_parents:
        parent_vecs = embed_parallel(
            [p.text for p in embeddable_parents],
            settings.embedding_model,
            settings.embedding_dim,
            workers=settings.index_embed_workers,
            local_embed=embed_fn,
        )
        store.parents.add(
            parent_vecs,
            [meta_of(node, parent_text=node.text, is_parent=True) for node in embeddable_parents],
        )
    if child_nodes:
        child_vecs = embed_parallel(
            [c.text for c in child_nodes],
            settings.embedding_model,
            settings.embedding_dim,
            workers=settings.index_embed_workers,
            local_embed=embed_fn,
        )
        store.children.add(
            child_vecs,
            [meta_of(node, parent_text=parent_text_by_id.get(node.parent_id)) for node in child_nodes],
        )
    print(
        f"[indexer] embedded in {time.perf_counter() - embed_started:.1f}s -> {store.counts}",
        flush=True,
    )

    print("[indexer] building BM25 term statistics...", flush=True)
    bm25_started = time.perf_counter()
    store.bm25 = BM25Index.build([c.text for c in child_nodes])
    print(f"[indexer] bm25 built in {time.perf_counter() - bm25_started:.1f}s (vocab={len(store.bm25.vocab)})", flush=True)

    print("[indexer] building IVF index...", flush=True)
    index_stats = store.build_index(seed=settings.index_seed)

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    store.save(settings.index_dir)
    elapsed = time.perf_counter() - started
    stats = {
        "index_name": settings.index_name,
        "sampled_rows": len(rows),
        "documents": len(docs),
        **store.counts,
        **index_stats,
        "embedding_model": settings.embedding_model,
        "build_time_s": round(elapsed, 2),
    }
    (settings.index_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[indexer] done in {elapsed:.1f}s -> {settings.index_dir}")
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MSMARCO-XI index")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    settings = get_settings()
    if args.max_rows is not None:
        settings.index_max_rows = args.max_rows
    if args.seed is not None:
        settings.index_seed = args.seed
    asyncio.run(build_index(settings))


if __name__ == "__main__":
    main()
