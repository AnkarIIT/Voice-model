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
    conversation_history: list | None = None,
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

    top_score = chunks[0].get("score", 0.0) if chunks else 0.0
    if top_score < 0.45:
        total = (time.perf_counter() - t0) * 1000
        timings["llm_ms"] = 0
        timings["total_ms"] = round(total, 2)
        timings["retrieval_llm_ms"] = round(ret["total_ms"], 2)
        return {
            "status": "ok",
            "query": query_text,
            "stt": stt,
            "retrieved": chunks,
            "retrieval": ret,
            "answer": "I couldn't find specific information about this in my knowledge base. Try rephrasing or asking about topics like company policy, procedures, or services covered in the docs.",
            "guardrail": {"action": "allow", "reason": "low_relevance"},
            "hallucination": {"grounded": True, "hit_rate": 0.0, "method": "relevance_gate"},
            "timings": timings,
            "provider": "none",
        }

    g = check_guardrails(query_text, chunks, has_history=bool(conversation_history))
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

    if g["action"] == "allow" and g.get("reason") == "greeting":
        import random
        q = (query_text or "").strip()
        is_hindi = any("\u0900" <= ch <= "\u097F" for ch in q)
        is_bengali = any("\u0980" <= ch <= "\u09FF" for ch in q)
        if is_bengali:
            ans = random.choice([
                "নমস্কার! আমি RAGinGOA। হিন্দি, English বা Bengali — যেকোনো ভাষায় প্রশ্ন করুন।",
                "হ্যালো! আমি RAGinGOA। আজ আপনাকে কীভাবে সাহায্য করতে পারি?",
                "আদাব! আমি RAGinGOA — knowledge base থেকে যেকোনো প্রশ্নের উত্তর দিতে প্রস্তুত।",
            ])
        elif is_hindi:
            ans = random.choice([
                "नमस्ते! मैं RAGinGOA हूँ। हिंदी, English या Bengali में कुछ भी पूछें।",
                "नमस्कार! मैं RAGinGOA हूँ। आज मैं आपकी क्या मदद कर सकता हूँ?",
                "प्रणाम! मैं RAGinGOA हूँ — knowledge base से किसी भी सवाल का जवाब देने के लिए तैयार हूँ।",
            ])
        else:
            ans = random.choice([
                "Namaste! I'm RAGinGOA. Ask me anything in Hindi, English, or Bengali.",
                "Hello! I'm RAGinGOA. How can I help you today?",
                "Hi there! I'm RAGinGOA — ready to answer your questions from the knowledge base.",
            ])
        total = (time.perf_counter() - t0) * 1000
        timings["llm_ms"] = 0
        timings["total_ms"] = round(total, 2)
        timings["retrieval_llm_ms"] = round(ret["total_ms"], 2)
        return {
            "status": "ok",
            "query": query_text,
            "stt": stt,
            "retrieved": chunks,
            "retrieval": ret,
            "answer": ans,
            "guardrail": {"action": "allow", "reason": "greeting"},
            "hallucination": {"grounded": True, "hit_rate": 1.0, "method": "greeting_bypass"},
            "timings": timings,
            "provider": "greeting",
        }

    ans, llm_ms, prov = generate_answer(query_text, chunks, conversation_history=conversation_history)
    timings["llm_ms"] = round(llm_ms, 2)

    # If the LLM itself says it has no reliable answer, convert that into a
    # friendly user-facing message instead of surfacing the raw refusal.
    _converted_refusal = False
    _REFUSAL_PHRASES = {
        "no reliable answer found in context.",
        "no reliable answer found in retrieved context.",
    }
    if isinstance(ans, str) and ans.strip().lower() in _REFUSAL_PHRASES:
        _converted_refusal = True
        q = (query_text or "").strip()
        if any("\u0900" <= ch <= "\u097F" for ch in q):
            ans = "मुझे इस सवाल का जवाब knowledge base में नहीं मिला। कृपया दूसरे शब्दों में पूछें।"
        elif any("\u0980" <= ch <= "\u09FF" for ch in q):
            ans = "আমি knowledge base-এ এই প্রশ্নের উত্তর পাইনি। অন্য শব্দে পুনরায় জিজ্ঞাসা করুন।"
        else:
            ans = "I couldn't find specific information about this in my knowledge base. Try rephrasing or asking about topics like company policy, procedures, or services covered in the docs."

    h = hallucination_check(ans, chunks, encoder=getattr(index, "model", None))
    from .guardrails import answer_draws_on_history

    if _converted_refusal:
        guardrail = {"action": "allow", "reason": "no_answer", "answer": None}
        h = {**h, "grounded": True, "method": "refusal_converted"}
    elif h["grounded"] or answer_draws_on_history(ans, conversation_history):
        guardrail = g
        if not h.get("grounded"):
            h = {**h, "grounded": True, "method": "history_follow_up"}
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
