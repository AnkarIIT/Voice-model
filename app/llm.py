import logging
import os
import time

logger = logging.getLogger(__name__)

SYS_PROMPT = (
    "Answer concisely and ONLY using the provided context. "
    "The context may be in Hindi or Bengali while the question is in English, "
    "or the other way around: translate facts faithfully and reply in the "
    "language of the question. "
    "If the context lacks the answer, say 'No reliable answer found in context.'"
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
        sep = 2 if parts else 0
        allowance = budget_chars - used - len(head) - len(tail) - sep
        if allowance < 120:
            break
        body = str(c.get("text", ""))[:allowance].strip()
        if not body:
            continue
        parts.append(f"{head}{body}{tail}")
        used += len(head) + len(body) + len(tail) + sep
    return "\n\n".join(parts)


def _history_prompt(history: list) -> str:
    if not history:
        return ""
    lines = ["Conversation so far:"]
    for turn in history[-6:]:
        role = turn.get("role", "user")
        text = turn.get("text", "")
        if role == "user":
            lines.append(f"User: {text}")
        else:
            lines.append(f"Assistant: {text}")
    lines.append("\nNow answer the user's latest question using the context below.\n")
    return "\n".join(lines)


def build_prompt(query: str, chunks: list, conversation_history: list | None = None) -> str:
    ctx = _ctx_block(chunks, PROMPT_BUDGET_CHARS)
    hist = _history_prompt(conversation_history or [])
    return (
        f"{hist}Question: {query}\n"
        f"Context:\n{ctx}\n\n"
        f"Instruction: {SYS_PROMPT}\nAnswer:"
    )


def generate_answer(query: str, chunks: list, max_tokens: int = 256, conversation_history: list | None = None):
    t0 = time.perf_counter()
    prompt = build_prompt(query, chunks, conversation_history=conversation_history)

    gem_key = os.getenv("GEMINI_API_KEY", "").strip()
    open_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "gemini-3.6-flash")

    if gem_key:
        text = _try_gemini(query, chunks, gem_key, model, max_tokens, conversation_history=conversation_history)
        if text:
            return text, (time.perf_counter() - t0) * 1000, f"gemini:{model}"
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


def generate_answer_stream(query: str, chunks: list, max_tokens: int = 256, conversation_history: list | None = None):
    gem_key = os.getenv("GEMINI_API_KEY", "").strip()
    open_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "gemini-3.6-flash")

    if gem_key:
        yield from _stream_gemini(query, chunks, gem_key, model, max_tokens, conversation_history=conversation_history)
        return
    if open_key:
        openai_model = model if "gpt" in model else "gpt-4o-mini"
        yield from _stream_openai(build_prompt(query, chunks, conversation_history=conversation_history), open_key, openai_model, max_tokens)
        return
    if os.getenv("USE_LOCAL_LLM", "0").strip().lower() in {"1", "true", "yes"}:
        local_model = model if "flan" in model else os.getenv("LOCAL_LLM_MODEL", "google/flan-t5-base")
        text = _try_local(build_prompt(query, chunks, conversation_history=conversation_history), local_model, max_tokens)
        if text:
            yield text
            return
    yield _extractive_fallback(query, chunks)


def _try_gemini(query, chunks, key, model, max_tokens, conversation_history=None):
    global _GENAI_CLIENT
    try:
        from google import genai
        from google.genai import types

        if _GENAI_CLIENT is None:
            _GENAI_CLIENT = genai.Client(api_key=key)
        r = _GENAI_CLIENT.models.generate_content(
            model=model,
            contents=_gemini_contents(query, chunks, conversation_history=conversation_history),
            config=types.GenerateContentConfig(
                system_instruction=SYS_PROMPT,
                max_output_tokens=max_tokens,
                temperature=0.2,
            ),
        )
        text = (r.text or "").strip()
        if text:
            return text
        logger.warning("gemini returned empty response")
    except Exception as e:
        logger.warning("gemini failed: %s", e)
    return None


def _stream_gemini(query, chunks, key, model, max_tokens, conversation_history=None):
    global _GENAI_CLIENT
    try:
        from google import genai
        from google.genai import types

        if _GENAI_CLIENT is None:
            _GENAI_CLIENT = genai.Client(api_key=key)
        stream = _GENAI_CLIENT.models.generate_content_stream(
            model=model,
            contents=_gemini_contents(query, chunks, conversation_history=conversation_history),
            config=types.GenerateContentConfig(
                system_instruction=SYS_PROMPT,
                max_output_tokens=max_tokens,
                temperature=0.2,
            ),
        )
        for chunk in stream:
            text = (getattr(chunk, "text", None) or "").strip()
            if text:
                yield text
    except Exception as e:
        logger.warning("gemini stream failed: %s", e)
        yield _extractive_fallback(query, chunks)


def _gemini_contents(query: str, chunks: list, conversation_history: list | None = None) -> str:
    ctx = _ctx_block(chunks, PROMPT_BUDGET_CHARS)
    hist = _history_prompt(conversation_history or [])
    return f"{hist}Question: {query}\n\nContext:\n{ctx}"


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


def _stream_openai(prompt, key, model, max_tokens):
    try:
        from openai import OpenAI

        c = OpenAI(api_key=key, timeout=REQUEST_TIMEOUT_S)
        stream = c.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
            timeout=REQUEST_TIMEOUT_S,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
    except Exception as e:
        logger.warning("openai stream failed: %s", e)
        yield _extractive_fallback_from_prompt(prompt)


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


def _extractive_fallback_from_prompt(prompt: str) -> str:
    # crude fallback when streaming openai fails
    ctx_marker = "Context:\n"
    if ctx_marker in prompt:
        after = prompt.split(ctx_marker, 1)[1]
        first = after.split("\n\n", 1)[0]
        return first.split(" (score:", 1)[0].strip()[:600]
    return prompt[:600]
