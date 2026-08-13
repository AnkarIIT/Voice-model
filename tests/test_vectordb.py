import asyncio

import numpy as np
import pytest

from voice_rag.store import VectorStoreCollection
from voice_rag.vectordb import InMemoryVectorDB


def make_vectors(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test_add_and_bruteforce_search():
    db = InMemoryVectorDB(4)
    vectors = make_vectors(10, 4, seed=1)
    meta = [{"id": i} for i in range(10)]
    db.add(vectors, meta)
    query = vectors[3]
    hits = asyncio.run(db.search(query, top_k=3))
    assert hits[0][0] > 0.99
    assert hits[0][1]["id"] == 3


def test_ivf_matches_bruteforce():
    dim = 8
    vectors = make_vectors(400, dim, seed=2)
    meta = [{"id": i} for i in range(400)]
    db = InMemoryVectorDB(dim)
    db.add(vectors, meta)
    brute = asyncio.run(db.search(vectors[7], top_k=3))
    db.build_index(n_clusters=32, seed=0)
    ivf = asyncio.run(db.search(vectors[7], top_k=3, probe=8))
    assert brute[0][1]["id"] == ivf[0][1]["id"]


def test_metadata_filter():
    db = InMemoryVectorDB(4)
    vectors = make_vectors(20, 4, seed=3)
    meta = [{"id": i, "lang": "hi" if i % 2 == 0 else "en"} for i in range(20)]
    db.add(vectors, meta)
    hits = asyncio.run(db.search(vectors[1], top_k=5, filter_fn=lambda m: m["lang"] == "en"))
    assert all(h[1]["lang"] == "en" for h in hits)


def test_two_tier_store_merge_dedupes_parents(tmp_path):
    dim = 8
    store = VectorStoreCollection(dim)
    v = make_vectors(2, dim, seed=4)
    store.add_child(v[0], {"chunk_id": "c1", "source_id": "p1", "source_text": "parent one text", "language": "en", "source_kind": "passage", "selected": False})
    store.add_parent(v[0], {"chunk_id": "p1", "source_id": "p1", "source_text": "parent one text", "language": "en", "source_kind": "passage", "selected": False, "is_parent": True})
    store.add_parent(v[1], {"chunk_id": "p2", "source_id": "p2", "source_text": "parent two text", "language": "en", "source_kind": "passage", "selected": False, "is_parent": True})
    store.build_index(seed=0)
    hits = asyncio.run(store.search(v[0], child_top_k=3, parent_top_k=3))
    texts = [h.text for h in hits]
    assert "parent one text" in texts
    assert len(texts) == 2


def test_save_load_roundtrip(tmp_path):
    dim = 8
    store = VectorStoreCollection(dim)
    v = make_vectors(3, dim, seed=5)
    store.add_child(v[0], {"chunk_id": "c1", "source_id": "p1", "source_text": "text", "language": "en", "source_kind": "passage", "selected": True})
    store.add_parent(v[1], {"chunk_id": "p2", "source_id": "p2", "source_text": "text2", "language": "hi", "source_kind": "answer", "selected": True, "is_parent": True})
    store.build_index(seed=0)
    store.save(tmp_path)
    loaded = VectorStoreCollection.load(tmp_path)
    assert loaded.counts["children"] == 1
    assert loaded.counts["parents"] == 1
    hits = asyncio.run(loaded.search(v[0], child_top_k=1, parent_top_k=1))
    assert hits[0].text == "text"
    assert hits[0].selected is True
