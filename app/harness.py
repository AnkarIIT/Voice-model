import asyncio
import logging
import random
import time
from pathlib import Path

from .guardrails import screen_text
from .stt import STTConfigError, STTResult, get_provider

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_S = 0.4


def transcribe_with_harness(
    audio_path: Path, language_code: str = "hi-IN", provider: str = "sarvam"
) -> dict:
    try:
        prov = get_provider(provider)
    except (STTConfigError, ValueError) as e:
        logger.error("STT provider unavailable: %s", e)
        return _error_result(provider, language_code, str(e), attempts=0)

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()
            r: STTResult = prov.transcribe(audio_path, language_code)
            total_ms = (time.perf_counter() - t0) * 1000
            out = {
                "status": "ok",
                "attempt": attempt,
                "provider_requested": provider,
                "provider_used": r.provider,
                "text": r.text.strip(),
                "language": r.language,
                "confidence": r.confidence,
                "latency_ms": round(r.latency_ms, 2),
                "total_ms": round(total_ms, 2),
                "error": None,
                "guardrail": screen_text(r.text),
            }
            if not out["text"]:
                out["status"] = "empty"
                out["guardrail"] = {"action": "reject", "reason": "empty_transcript"}
            return out
        except STTConfigError as e:
            logger.error("non-retryable STT error: %s", e)
            return _error_result(provider, language_code, str(e), attempts=attempt)
        except Exception as e:
            last_err = str(e)
            logger.warning("STT attempt %d/%d failed: %s", attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt + random.uniform(0, 0.2))
    return _error_result(provider, language_code, last_err, attempts=MAX_RETRIES)


def _error_result(provider: str, language_code: str, err, attempts: int) -> dict:
    return {
        "status": "error",
        "attempt": attempts,
        "provider_requested": provider,
        "provider_used": provider,
        "text": "",
        "language": language_code,
        "confidence": None,
        "latency_ms": 0,
        "total_ms": 0,
        "error": str(err),
        "guardrail": {"action": "error", "reason": "stt_failed"},
    }


async def atranscribe_with_harness(*args, **kwargs):
    return await asyncio.to_thread(transcribe_with_harness, *args, **kwargs)
