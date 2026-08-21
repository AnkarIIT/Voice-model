from app.llm import SYS_PROMPT, _extractive_fallback, build_prompt


def test_prompt_keeps_instruction_intact():
    chunks = [{"text": "word " * 2000, "score": 0.9}]
    p = build_prompt("some question", chunks)
    assert p.endswith(f"Instruction: {SYS_PROMPT}\nAnswer:")
    assert len(p) < 5000


def test_prompt_truncates_context_not_instruction():
    chunks = [{"text": "x" * 10000, "score": 0.5} for _ in range(5)]
    p = build_prompt("q", chunks)
    assert f"Instruction: {SYS_PROMPT}" in p
    assert len(p) <= 3600 + len(SYS_PROMPT) + 250


def test_extractive_fallback_picks_relevant_sentence():
    chunks = [
        {
            "text": (
                "The Eiffel Tower is in Paris. "
                "Agriculture depends on monsoon rains. "
                "Paris is the capital of France."
            ),
            "score": 0.9,
        }
    ]
    out = _extractive_fallback("where is the eiffel tower located", chunks)
    assert "Eiffel" in out


def test_extractive_fallback_empty():
    assert _extractive_fallback("q", []) == ""
