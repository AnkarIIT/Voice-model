from app.guardrails import check_guardrails, hallucination_check, screen_text


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
