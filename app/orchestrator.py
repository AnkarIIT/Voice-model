import logging
import time
from pathlib import Path

from .embed import FaissIndex
from .guardrails import check_guardrails, hallucination_check
from .harness import transcribe_with_harness
from .llm import generate_answer
from .retrieval import retrieve

logger = logging.getLogger(__name__)


def run_pipeline(
    query_text: str = None,
    audio_path: Path = None,
    language_code: str = "hi-IN",
    index: FaissIndex = None,
    k: int = 5,
    use_rerank: bool = False,
    stt_provider: str = "sarvam",
) -> dict:
    timings = {}
    t0 = time.perf_counter()
    stt = None

    if audio_path is not None:
        s0 = time.perf_counter()
        stt = transcribe_with_harness(audio_path, language_code, stt_provider)
        timings["stt_ms"] = round(stt.get("total_ms", 0), 2)
        if stt["status"] != "ok" or not stt["text"]:
            total = (time.perf_counter() - t0) * 1000
            timings["total_ms"] = round(total, 2)
            return {
                "status": "stt_failed",
                "stt": stt,
                "timings": timings,
                "answer": None,
                "guardrail": {"action": "reject", "reason": "stt_failed"},
            }
        query_text = stt["text"]
    else:
        timings["stt_ms"] = 0
        stt = {"text": query_text, "provider_used": "text-input"}

    ret = retrieve(query_text, index, k, use_rerank=use_rerank)
    timings["retrieval_ms"] = ret["total_ms"]
    timings["search_ms"] = ret["search_ms"]
    chunks = ret["results"]

    g = check_guardrails(query_text, chunks)
    if g["action"] != "allow":
        total = (time.perf_counter() - t0) * 1000
        timings["llm_ms"] = 0
        timings["total_ms"] = round(total, 2)
        return {
            "status": g["action"],
            "query": query_text,
            "stt": stt,
            "retrieved": chunks,
            "retrieval": ret,
            "answer": g["answer"],
            "guardrail": g,
            "hallucination": {"grounded": False},
            "timings": timings,
            "provider": "guardrail",
        }

    ans, llm_ms, prov = generate_answer(query_text, chunks)
    timings["llm_ms"] = round(llm_ms, 2)
    h = hallucination_check(ans, chunks)
    if h["grounded"]:
        guardrail = g
    else:
        logger.info("answer flagged ungrounded (hit_rate=%s)", h.get("hit_rate"))
        guardrail = {"action": "flag", "reason": "ungrounded", "answer": ans}

    total = (time.perf_counter() - t0) * 1000
    timings["total_ms"] = round(total, 2)
    timings["retrieval_llm_ms"] = round(ret["total_ms"] + llm_ms, 2)

    return {
        "status": "ok",
        "query": query_text,
        "stt": stt,
        "retrieved": chunks,
        "retrieval": ret,
        "answer": ans,
        "guardrail": guardrail,
        "hallucination": h,
        "provider": prov,
        "timings": timings,
    }
