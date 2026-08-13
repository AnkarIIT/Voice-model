from __future__ import annotations

import re
from typing import Optional

import numpy as np

from .models import GuardrailKind, GuardrailVerdict

_WORD_RE = re.compile(r"\w+", re.UNICODE)

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|messages?)", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"(reveal|show|print|output|expose)\s+(your\s+|the\s+)?(system\s+)?(prompt|instructions?|system\s+message)", re.I),
    re.compile(r"(you\s+are|act\s+as|pretend\s+to\s+be)\s+(now\s+)?(a\s+)?(developer|admin|assistant\s+mode|unrestricted|jailbreak)", re.I),
    re.compile(r"\bdan\b|jailbreak|bypass\s+(the\s+)?(rules?|guardrails?|filters?)", re.I),
    re.compile(r"chain[- ]of[- ]thought|let\s+your\s+model\s+think", re.I),
    re.compile(r"forget\s+(everything|all\s+your\s+instructions|your\s+guidelines)", re.I),
re.compile(r"\b(drop|delete|erase|wipe|purge)\s+(all\s+|the\s+)?(passwords?|credentials|databases?|tables?|records|rows|columns)\b", re.I),
]

_UNSAFE_PATTERNS = [
    re.compile(r"\b(kill|murder|slaughter|rape|bomb|explosives?)\b", re.I),
    re.compile(r"how\s+to\s+(build|make|create|construct)\s+(a\s+)?(bomb|explosive|weapon)", re.I),
    re.compile(r"\b(child\s*porn|grooming|sex\s*tape)\b", re.I),
]

_PII_PATTERNS = [
    re.compile(r"\b\d{10}\b"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
]

_OFFTOPIC_PATTERNS = [
    re.compile(r"^(hi|hello|hey|namaste|namaskar|good\s*(morning|afternoon|evening))\b", re.I),
    re.compile(r"^(who\s+are\s+you|what\s+can\s+you\s+do|what\s+are\s+you)\b", re.I),
    re.compile(r"\b(create|compose|write|generate|draw|paint|sing)\s+(a|an|me|the)?\s*(song|poem|story|essay|image|picture|video|code|email|letter)\b", re.I),
    re.compile(r"\b(order|book|buy|purchase|sell|reserve)\b[\w\s]{0,20}\b(pizza|ticket|flight|hotel|table|cab|uber|food|item|product|property)\b", re.I),
    re.compile(r"\b(set|schedule|cancel|remind|play|stop|pause)\s+(a|an|the|my|me)?\s*(reminder|alarm|timer|music|song|call)\b", re.I),
]

_EMPTY_WORDS = {"", "hm", "um", "uh", "a", "the", "hello", "hi"}


class GuardrailEngine:
    """Deterministic first-line guardrails, evaluated in sub-millisecond time.

    Order of evaluation: structural validity, prompt-injection, unsafe content,
    PII harvesting, then domain relevance. Every rejection returns a structured
    verdict (kind + reason) that the harness surfaces verbatim.
    """

    def __init__(
        self,
        off_topic_threshold: float = 0.20,
        max_query_len: int = 600,
    ) -> None:
        self.off_topic_threshold = off_topic_threshold
        self.max_query_len = max_query_len

    def evaluate(self, query: str, best_score: Optional[float] = None) -> GuardrailVerdict:
        q = (query or "").strip()
        if not q or q.lower() in _EMPTY_WORDS:
            return GuardrailVerdict(allowed=False, kind=GuardrailKind.EMPTY, reason="No question detected.")
        if len(q) > self.max_query_len:
            return GuardrailVerdict(allowed=False, kind=GuardrailKind.TOO_LONG, reason="Question exceeds maximum supported length.")

        for pattern in _INJECTION_PATTERNS:
            if pattern.search(q):
                return GuardrailVerdict(
                    allowed=False,
                    kind=GuardrailKind.PROMPT_INJECTION,
                    reason="Query appears to attempt prompt injection or instruction override.",
                )
        for pattern in _UNSAFE_PATTERNS:
            if pattern.search(q):
                return GuardrailVerdict(
                    allowed=False,
                    kind=GuardrailKind.UNSAFE,
                    reason="Query contains unsafe or harmful content and was blocked.",
                )
        for pattern in _PII_PATTERNS:
            if pattern.search(q):
                return GuardrailVerdict(
                    allowed=False,
                    kind=GuardrailKind.PII,
                    reason="Query requests or contains personally identifiable information and was blocked.",
                )
        for pattern in _OFFTOPIC_PATTERNS:
            if pattern.search(q):
                return GuardrailVerdict(
                    allowed=False,
                    kind=GuardrailKind.OFF_TOPIC,
                    reason="This task is outside the scope of the knowledge-base assistant.",
                )
        if best_score is not None and best_score < self.off_topic_threshold:
            return GuardrailVerdict(
                allowed=False,
                kind=GuardrailKind.OFF_TOPIC,
                reason="No relevant information found in the knowledge base for this question.",
                score=float(best_score),
            )
        return GuardrailVerdict(allowed=True, kind=GuardrailKind.NONE)

    @staticmethod
    def lexical_coverage(answer: str, contexts: list[str]) -> float:
        """Fraction of answer content words that appear in the retrieved context."""
        answer_words = set(_WORD_RE.findall(answer.lower()))
        if not answer_words:
            return 0.0
        context_words = set()
        for ctx in contexts:
            context_words.update(_WORD_RE.findall(ctx.lower()))
        return len(answer_words & context_words) / len(answer_words)

    @staticmethod
    def embedding_overlap(answer_emb: np.ndarray, context_embs: list[np.ndarray]) -> float:
        if not context_embs:
            return 0.0
        stack = np.asarray(context_embs, dtype=np.float32)
        return float(np.max(stack @ answer_emb))


class GroundingEngine:
    """Post-generation hallucination guard.

    Combines (a) explicit refusal markers produced by the LLM, (b) lexical
    coverage of the answer by the retrieved context, and (c) semantic overlap
    between the answer embedding and the context embeddings.
    """

    REFUSAL_MARKERS = [
        "insufficient_context",
        "not found in the provided",
        "not available in the provided",
        "not in the context",
        "i cannot",
        "i can't",
        "i don't have",
        "no information provided",
        "कोई जानकारी",
        "नहीं मिल",
        "जानकारी नहीं है",
        "उत्तर नहीं",
    ]

    def __init__(self, lexical_threshold: float = 0.10, embedding_threshold: float = 0.45) -> None:
        self.lexical_threshold = lexical_threshold
        self.embedding_threshold = embedding_threshold

    def check(
        self,
        answer: str,
        contexts: list[str],
        answer_emb: Optional[np.ndarray] = None,
        context_embs: Optional[list[np.ndarray]] = None,
    ) -> tuple[bool, float, str]:
        low = answer.lower()
        if any(marker in low for marker in self.REFUSAL_MARKERS):
            return False, 0.0, "model refused or found insufficient context"
        lexical = GuardrailEngine.lexical_coverage(answer, contexts)
        semantic = 0.0
        if answer_emb is not None and context_embs:
            semantic = GuardrailEngine.embedding_overlap(answer_emb, context_embs)
        combined = 0.5 * lexical + 0.5 * semantic
        threshold = 0.5 * self.lexical_threshold + 0.5 * self.embedding_threshold
        if combined < threshold:
            return False, float(combined), "answer is not grounded in retrieved context"
        return True, float(combined), "grounded"
