from __future__ import annotations

import numpy as np

from voice_rag.chunking import FixedSizeChunker, ParentChildChunker, SemanticChunker, split_sentences


def fake_embed(texts):
    vecs = []
    for t in texts:
        rng = np.random.default_rng(sum(ord(c) for c in t) % (2**32))
        v = rng.standard_normal(16)
        v /= np.linalg.norm(v)
        vecs.append(v)
    return np.stack(vecs)


def test_split_sentences_handles_hi_and_en():
    text = "भारत की राजधानी दिल्ली है। Delhi is the capital. Really!"
    sents = split_sentences(text)
    assert len(sents) == 3


def test_fixed_size_chunker_overlap_and_continuity():
    text = " ".join(f"word{i}" for i in range(50))
    chunker = FixedSizeChunker(window_words=20, overlap=0.5)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    assert all(c.word_count <= 20 for c in chunks)
    first, second = chunks[0], chunks[1]
    overlap = set(first.text.split()) & set(second.text.split())
    assert len(overlap) > 0


def test_semantic_chunker_respects_max_words():
    text = " ".join(f"sentence{i} with some words" for i in range(12))
    chunker = SemanticChunker(embed_fn=fake_embed, target_words=24, min_words=8, max_words=32)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    assert all(c.word_count <= 32 for c in chunks)
    assert all(c.strategy in ("semantic", "semantic_oversize") for c in chunks)


def test_adaptive_chunk_atomic_for_short_passage():
    text = "The quick brown fox jumps over the lazy dog."
    parents, children = ParentChildChunker(embed_fn=fake_embed).chunk(text)
    assert len(parents) == 1 and len(children) == 1
    assert parents[0].parent_id == children[0].parent_id
    assert children[0].text == text


def test_parent_child_mapping_integrity():
    para = (
        "The first concept is about gravity which pulls objects together. "
        "Newton observed an apple falling and formulated the inverse square law. "
        "The second concept is electromagnetism which governs electric charges. "
        "Faraday and Maxwell formalized these ideas into field equations. "
        "The third concept is thermodynamics dealing with heat and entropy. "
        "Boltzmann linked entropy to the microscopic states of a system. "
        "Each of these pillars reshaped physics and engineering alike."
    )
    long_text = " ".join([para] * 5)
    assert len(long_text.split()) > 200
    parents, children = ParentChildChunker(embed_fn=fake_embed).chunk(long_text)
    assert len(parents) >= 1
    assert len(children) >= 2
    parent_ids = {p.chunk_id for p in parents}
    for c in children:
        assert c.parent_id in parent_ids


def test_semantic_overlap_sentences_preserves_context():
    text = " ".join(f"sentence{i} here" for i in range(10))
    chunker = SemanticChunker(
        embed_fn=fake_embed, target_words=8, min_words=4, max_words=16, overlap_sentences=1
    )
    chunks = chunker.chunk(text)
    if len(chunks) >= 2:
        assert set(chunks[1].text.split()).intersection(set(chunks[0].text.split()))
