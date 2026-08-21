import pytest

from app.chunking import fixed_size_overlap, hybrid_chunk_row, semantic_split


def test_short_text_single_chunk():
    assert fixed_size_overlap("one two three") == ["one two three"]
    assert fixed_size_overlap("") == []


def test_overlap_windows_terminate_and_cover():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = fixed_size_overlap(text, chunk_tokens=100, overlap=0.2)
    assert all(len(c.split()) >= 20 for c in chunks)
    assert len(chunks) > 1
    words = set(text.split())
    covered = set()
    for c in chunks:
        covered |= set(c.split())
    assert covered == words


def test_invalid_params():
    with pytest.raises(ValueError):
        fixed_size_overlap("x" * 1000, chunk_tokens=10)


def test_semantic_split_no_model_groups_by_count():
    text = "। ".join(f"Sentence number {i} has some words here" for i in range(10))
    chunks = semantic_split(text, model=None, max_sent_per_chunk=3)
    assert 3 <= len(chunks) <= 4


def _row():
    passage = "The capital of India is New Delhi. " * 30
    return {
        "query_id": "q1",
        "query_type": "CAUSE",
        "query": "what is the capital of India?",
        "source_lang": "en",
        "target_lang": "hi",
        "passages": {
            "English_passages": [passage],
            "Translated_passages": [passage],
            "is_selected": [1],
        },
    }


def test_hybrid_chunks_have_clean_text_and_unique_keys():
    row = _row()
    chunks = hybrid_chunk_row(row)
    assert chunks
    strategies = {c["strategy"] for c in chunks}
    assert strategies == {"fixed256_overlap20", "fixed512_overlap15", "semantic", "metadata_raw"}
    keys = [
        (c["doc_id"], c["strategy"], c["text"]) for c in chunks
    ]
    assert len(keys) == len(set(keys))
    for c in chunks:
        assert not c["text"].startswith("[")
        assert c["meta_label"].startswith("CAUSE|hi|doc:q1")


def test_hybrid_dedup_removes_duplicates():
    row = _row()
    short_passage = {"**": None}
    row["passages"]["Translated_passages"] = ["short passage under twenty words total here"]
    chunks = hybrid_chunk_row(row)
    texts = [(c["strategy"], c["text"]) for c in chunks]
    assert len(texts) == len(set(texts))
