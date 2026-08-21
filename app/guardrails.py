import logging
import re

from .config import ABSTAIN_THRESHOLD, GROUNDING_HIT_RATE

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
    if len((query or "").strip()) < 3:
        return {"action": "reject", "reason": "too_short", "answer": "Query too short."}
    screened = screen_text(query)
    if screened["action"] == "block":
        logger.info("blocked unsafe query")
        return {**screened, "answer": "Blocked: unsafe content."}
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


def hallucination_check(answer: str, retrieved: list) -> dict:
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
    result.update({
        "grounded": hit >= GROUNDING_HIT_RATE,
        "hit_rate": round(hit, 2),
    })
    return result
