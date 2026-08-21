import logging
import re

from .config import ABSTAIN_THRESHOLD, GROUNDING_EMBED_SIM, GROUNDING_HIT_RATE

logger = logging.getLogger(__name__)

ABSTAIN_ANSWER = "No reliable answer found in retrieved context."

UNSAFE_RE = re.compile(
    r"(?<![\w])("
    r"bomb(?:s|ing)?|kill(?:s|ed|ing)?|suicid\w*|self[\s-]?harm|terror(?:ism|ists?)?"
    r"|porn\w*|child\s*(?:porn|abuse)|how\s+to\s+(?:make|build)\s+(?:a\s+)?(?:bomb|weapon)"
    r"|बम|हत्या|आत्महत्या|आतंकवाद|अश्लील"
    r")(?![\w])",
    re.IGNORECASE,
)

GREETING_RE = re.compile(
    r"^(?:hi|hello|hey|namaste|namaskar|good\s*(?:morning|afternoon|evening)|howdy|greetings|hola|hiya|yo|नमस्ते|नमस्कार|प्रणाम|सुप्रभात|शुभकामनाएं|सत्य)[\s!,.]*$",
    re.IGNORECASE,
)

GREETING_ANSWERS = [
    "Namaste! I'm RAGinGOA. Ask me anything in Hindi, English, or Bengali.",
    "Hello! I'm RAGinGOA. How can I help you today?",
    "Hi there! I'm RAGinGOA — ready to answer your questions from the knowledge base.",
]

HINDI_GREETING_ANSWERS = [
    "नमस्ते! मैं RAGinGOA हूँ। हिंदी, English या Bengali में कुछ भी पूछें।",
    "नमस्कार! मैं RAGinGOA हूँ। आज मैं आपकी क्या मदद कर सकता हूँ?",
    "प्रणाम! मैं RAGinGOA हूँ — knowledge base से किसी भी सवाल का जवाब देने के लिए तैयार हूँ।",
]


def screen_text(text: str) -> dict:
    t = (text or "").strip()
    if len(t) < 2:
        return {"action": "reject", "reason": "too_short_or_empty"}
    if UNSAFE_RE.search(t):
        return {"action": "block", "reason": "unsafe_keyword"}
    return {"action": "allow", "reason": "ok"}


def check_guardrails(query: str, retrieved: list, threshold: float = None) -> dict:
    if threshold is None:
        threshold = ABSTAIN_THRESHOLD
    q = (query or "").strip()
    if len(q) < 3 and not GREETING_RE.search(q):
        return {"action": "reject", "reason": "too_short", "answer": "Query too short."}
    screened = screen_text(q)
    if screened["action"] == "block":
        logger.info("blocked unsafe query")
        return {**screened, "answer": "Blocked: unsafe content."}
    if GREETING_RE.search(q):
        return {"action": "allow", "reason": "greeting", "answer": None}
    if not retrieved:
        return {"action": "abstain", "reason": "no_context", "answer": ABSTAIN_ANSWER}
    max_score = max(r.get("score", 0.0) for r in retrieved)
    if max_score < threshold:
        return {
            "action": "abstain",
            "reason": "low_confidence",
            "answer": ABSTAIN_ANSWER + " (low similarity)",
        }
    return {"action": "allow", "reason": "ok", "answer": None}


def _dominant_script(text: str) -> str:
    deva = latin = other = 0
    for ch in text or "":
        if "\u0900" <= ch <= "\u097F":
            deva += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
        elif ch.isalpha():
            other += 1
    best = max(deva, latin, other)
    if best == 0:
        return "none"
    if best == deva:
        return "devanagari"
    if best == latin:
        return "latin"
    return "other"


def _embed_similarity(answer: str, retrieved: list, encoder):
    if encoder is None:
        return None
    try:
        import numpy as np

        texts = [str(r.get("text", ""))[:1200] for r in retrieved[:8]]
        embs = encoder.encode(
            [answer] + texts, normalize_embeddings=True, convert_to_numpy=True
        )
        sims = embs[1:] @ embs[0]
        return float(np.max(sims))
    except Exception as e:
        logger.warning("embedding grounding check failed: %s", e)
        return None


def hallucination_check(answer: str, retrieved: list, encoder=None) -> dict:
    result = {"method": "lexical_overlap"}
    if not answer or not retrieved:
        result.update({"grounded": False, "hit_rate": 0.0})
        return result
    ctx = " ".join(str(r.get("text", "")) for r in retrieved).lower()
    ans_terms = [w for w in re.findall(r"\w+", answer.lower()) if len(w) > 3][:50]
    if not ans_terms:
        result.update({"grounded": True, "hit_rate": 1.0})
        return result
    hit = sum(1 for w in ans_terms if w in ctx) / len(ans_terms)
    if hit >= GROUNDING_HIT_RATE:
        result.update({"grounded": True, "hit_rate": round(hit, 2)})
        return result

    ans_script = _dominant_script(answer)
    ctx_script = _dominant_script(ctx)
    if ans_script != "none" and ctx_script != "none" and ans_script != ctx_script:
        sim = _embed_similarity(answer, retrieved, encoder)
        if sim is not None:
            logger.info("cross-script grounding via embedding sim=%.3f", sim)
            result.update({
                "method": "embedding_similarity",
                "hit_rate": round(hit, 2),
                "embed_sim": round(sim, 3),
                "grounded": sim >= GROUNDING_EMBED_SIM,
            })
            return result
        result.update({
            "method": "cross_lingual_unverified",
            "hit_rate": round(hit, 2),
            "grounded": True,
        })
        return result

    result.update({"grounded": False, "hit_rate": round(hit, 2)})
    return result
