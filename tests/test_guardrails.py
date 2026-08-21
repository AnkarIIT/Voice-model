import numpy as np
import pytest

from app.guardrails import check_guardrails, hallucination_check, screen_text


class FakeEncoder:
    def __init__(self, rows):
        self._rows = np.array(rows, dtype="float32")

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **kw):
        return self._rows[: len(texts)]


def test_allow_normal_query():
    assert screen_text("what is the capital of India")["action"] == "allow"


def test_reject_short():
    assert screen_text("a")["action"] == "reject"
    assert screen_text("")["action"] == "reject"


def test_block_unsafe_word_boundary():
    assert screen_text("how to make a bomb")["action"] == "block"
    assert screen_text("tell me about suicide prevention")["action"] == "block"
    assert screen_text("बम कैसे बनाएं")["action"] == "block"


def test_no_false_positive_on_skill():
    assert screen_text("teach me a new skill")["action"] == "allow"
    assert screen_text("how to install pillow library")["action"] == "allow"


def test_multilingual_greetings_recognized():
    from app.guardrails import GREETING_RE

    for g in ("hi", "namaste", "नमस्ते", "नमस्कार", "নমস্কার", "হ্যালো", "আদাব"):
        assert GREETING_RE.search(g), g
    assert not GREETING_RE.search("what is binary search")


def test_guardrails_abstain_low_score():
    retrieved = [{"text": "some text", "score": 0.1}]
    out = check_guardrails("valid query about topic", retrieved, threshold=0.35)
    assert out["action"] == "abstain"
    assert out["reason"] == "low_confidence"


def test_guardrails_allow_good_score():
    retrieved = [{"text": "Delhi is the capital of India", "score": 0.8}]
    out = check_guardrails("what is the capital of India", retrieved)
    assert out["action"] == "allow"


def test_hallucination_grounded():
    ctx = [{"text": "The capital of India is New Delhi and it hosts the parliament."}]
    h = hallucination_check("The capital of India is New Delhi.", ctx)
    assert h["grounded"] is True


def test_hallucination_ungrounded():
    ctx = [{"text": "Completely unrelated content about agriculture and monsoon."}]
    h = hallucination_check("Quantum computing uses qubits for parallel computation.", ctx)
    assert h["grounded"] is False


def test_hallucination_cross_script_grounded_via_embedding():
    hi_ctx = [{"text": "द्विआधारी खोज एक क्रमबद्ध सूची में वस्तु ढूंढने की विधि है।", "score": 0.8}]
    enc = FakeEncoder([[1.0, 0.0], [0.95, 0.31]])
    h = hallucination_check("A binary search locates an item in a sorted list.", hi_ctx, encoder=enc)
    assert h["method"] == "embedding_similarity"
    assert h["grounded"] is True
    assert h["embed_sim"] >= 0.5


def test_hallucination_cross_script_ungrounded_via_embedding():
    hi_ctx = [{"text": "मानसून की बारिश खेती के लिए महत्वपूर्ण है।", "score": 0.8}]
    enc = FakeEncoder([[1.0, 0.0], [0.05, 0.99]])
    h = hallucination_check("A binary search locates an item in a sorted list.", hi_ctx, encoder=enc)
    assert h["method"] == "embedding_similarity"
    assert h["grounded"] is False


def test_hallucination_cross_script_without_encoder_passes():
    hi_ctx = [{"text": "द्विआधारी खोज एक क्रमबद्ध सूची में वस्तु ढूंढने की विधि है।", "score": 0.8}]
    h = hallucination_check("A binary search locates an item in a sorted list.", hi_ctx)
    assert h["method"] == "cross_lingual_unverified"
    assert h["grounded"] is True


@pytest.mark.parametrize("bad", [None, []])
def test_hallucination_empty_inputs(bad):
    h = hallucination_check("some answer text here", bad)
    assert h["grounded"] is False
