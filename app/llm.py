import logging
import os
import time

logger = logging.getLogger(__name__)

SYS_PROMPT = (
    "Answer concisely and ONLY using the provided context. "
    "If context lacks answer, say 'No reliable answer found in context.'"
)
PROMPT_BUDGET_CHARS = 3600
REQUEST_TIMEOUT_S = 30

_GENAI_CLIENT = None
_PIPE = None


def _ctx_block(chunks: list, budget_chars: int) -> str:
    parts = []
    used = 0
    for i, c in enumerate(chunks[:5]):
        head = f"[{i + 1}] "
        tail = f" (score:{c.get('score', 0):.2f})"
        allowance = max(200, budget_chars - used - len(head) - len(tail))
        body = str(c.get("text", ""))[:allowance].strip()
        if not body:
            break
        part = f"{head}{body}{tail}"
        parts.append(part)
        used += len(part) + 2
        if used >= budget_chars:
            break
    return "\n\n".join(parts)


def build_prompt(query: str, chunks: list) -> str:
    ctx = _ctx_block(chunks, PROMPT_BUDGET_CHARS)
    return (
        f"Question: {query}\n"
        f"Context:\n{ctx}\n\n"
        f"Instruction: {SYS_PROMPT}\nAnswer:"
    )


def generate_answer(query: str, chunks: list, max_tokens: int = 256):
    t0 = time.perf_counter()
    prompt = build_prompt(query, chunks)

    gem_key = os.getenv("GEMINI_API_KEY", "").strip()
    open_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "gemini-3.6-flash")

    if gem_key:
        text, label = _try_gemini(prompt, gem_key, model, max_tokens)
        if text:
            return text, (time.perf_counter() - t0) * 1000, label
    if open_key:
        openai_model = model if "gpt" in model else "gpt-4o-mini"
        text = _try_openai(prompt, open_key, openai_model, max_tokens)
        if text:
            return text, (time.perf_counter() - t0) * 1000, f"openai:{openai_model}"
    if os.getenv("USE_LOCAL_LLM", "0").strip().lower() in {"1", "true", "yes"}:
        local_model = model if "flan" in model else os.getenv("LOCAL_LLM_MODEL", "google/flan-t5-base")
        text = _try_local(prompt, local_model, max_tokens)
        if text:
            return text, (time.perf_counter() - t0) * 1000, f"local:{local_model}"

    logger.warning("no LLM available; using extractive fallback")
    return _extractive_fallback(query, chunks), (time.perf_counter() - t0) * 1000, "extractive-fallback"


def _try_gemini(prompt, key, model, max_tokens):
    global _GENAI_CLIENT
    try:
        from google import genai

        if _GENAI_CLIENT is None:
            _GENAI_CLIENT = genai.Client(api_key=key)
        r = _GENAI_CLIENT.models.generate_content(
            model=model,
            contents=prompt,
            config={"max_output_tokens": max_tokens, "temperature": 0.2},
        )
        text = (r.text or "").strip()
        if text:
            return text, f"gemini:{model}"
        logger.warning("gemini returned empty response")
    except Exception as e:
        logger.warning("gemini failed: %s", e)
    return None, None


def _try_openai(prompt, key, model, max_tokens):
    try:
        from openai import OpenAI

        c = OpenAI(api_key=key, timeout=REQUEST_TIMEOUT_S)
        r = c.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
            timeout=REQUEST_TIMEOUT_S,
        )
        text = (r.choices[0].message.content or "").strip()
        if text:
            return text
        logger.warning("openai returned empty response")
    except Exception as e:
        logger.warning("openai failed: %s", e)
    return None


def _try_local(prompt, model, max_tokens):
    global _PIPE
    try:
        if _PIPE is None:
            from transformers import pipeline
            from .embed import resolve_device
            from .config import WHISPER_DEVICE

            device_id = 0 if resolve_device(WHISPER_DEVICE) == "cuda" else -1
            _PIPE = pipeline("text2text-generation", model=model, device=device_id)
        out = _PIPE(prompt, max_new_tokens=max_tokens, do_sample=False)
        text = out[0]["generated_text"].strip()
        if text:
            return text
        logger.warning("local LLM returned empty response")
    except Exception as e:
        logger.warning("local LLM failed: %s", e)
    return None


def _extractive_fallback(query: str, chunks: list) -> str:
    top = str(chunks[0].get("text", "")) if chunks else ""
    q_terms = set((query or "").lower().split())
    sents = [s.strip() for s in top.split(". ") if s.strip()]
    best = (
        max(sents, key=lambda s: len(q_terms & set(s.lower().split())), default=top[:400])
        if sents
        else top[:400]
    )
    return (best or top)[:600]
