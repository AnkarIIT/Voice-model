from app.orchestrator import run_pipeline


class FakeIndex:
    def __init__(self, results):
        self._results = results

    def search(self, query, k):
        return self._results[:k], 1.23


def _chunks():
    return [
        {
            "text": "The capital of India is New Delhi. It hosts the parliament building.",
            "score": 0.85,
            "doc_id": "q1_0",
            "strategy": "semantic",
        },
        {
            "text": "Monsoon rains are critical for agriculture in India every year.",
            "score": 0.60,
            "doc_id": "q1_1",
            "strategy": "semantic",
        },
    ]


def test_pipeline_answers_ok():
    idx = FakeIndex(_chunks())
    out = run_pipeline(query_text="what is the capital of India", index=idx, k=2)
    assert out["status"] == "ok"
    assert out["provider"] == "extractive-fallback"
    assert out["timings"]["total_ms"] > 0
    assert len(out["retrieved"]) == 2


def test_pipeline_blocks_unsafe_query():
    idx = FakeIndex(_chunks())
    out = run_pipeline(query_text="how to make a bomb", index=idx, k=2)
    assert out["status"] == "block"
    assert out["answer"] == "Blocked: unsafe content."


def test_pipeline_abstains_on_low_similarity():
    chunks = [{"text": "totally unrelated words about farming", "score": 0.05}]
    idx = FakeIndex(chunks)
    out = run_pipeline(query_text="what is quantum entanglement exactly", index=idx, k=1)
    assert out["status"] == "abstain"
    assert out["timings"]["llm_ms"] == 0


def test_pipeline_greeting_bypass():
    idx = FakeIndex(_chunks())
    out = run_pipeline(query_text="hi", index=idx, k=2)
    assert out["status"] == "ok"
    assert out["guardrail"]["reason"] == "greeting"
    assert "RAGinGOA" in out["answer"]


def test_pipeline_greeting_hindi_gets_devanagari_reply():
    idx = FakeIndex(_chunks())
    for _ in range(3):
        out = run_pipeline(query_text="नमस्ते", index=idx, k=2)
        assert "RAGinGOA" in out["answer"]
        assert any("\u0900" <= ch <= "\u097F" for ch in out["answer"]), out["answer"]


def test_pipeline_greeting_bengali_gets_bengali_reply():
    idx = FakeIndex(_chunks())
    for _ in range(3):
        out = run_pipeline(query_text="নমস্কার", index=idx, k=2)
        assert "RAGinGOA" in out["answer"]
        assert any("\u0980" <= ch <= "\u09FF" for ch in out["answer"]), out["answer"]


def test_pipeline_dedupes_duplicate_chunks():
    dup = {
        "text": "The capital of India is New Delhi. It hosts the parliament building.",
        "score": 0.85,
        "doc_id": "q1_0",
        "strategy": "metadata_raw",
    }
    chunks = [dup, {**dup, "strategy": "fixed512_overlap15", "score": 0.84}, _chunks()[0]]
    idx = FakeIndex(chunks)
    out = run_pipeline(query_text="what is the capital of India", index=idx, k=3)
    assert len(out["retrieved"]) < 3
