from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Optional

import numpy as np

from .config import Settings, get_settings
from .embeddings import EmbeddingEngine
from .guardrails import GroundingEngine, GuardrailEngine
from .models import (
    GuardrailKind,
    GuardrailVerdict,
    LatencyBreakdown,
    RAGResult,
    RetrievedChunk,
    Status,
    SttResult,
    ToolCall,
)
from .providers import (
    CircuitBreaker,
    GroqLLM,
    MockLLM,
    MockSTT,
    ElevenLabsSTT,
    RetryPolicy,
    json_safe_load,
)
from .store import VectorStoreCollection

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)

_FALLBACK_MESSAGES = {
    GuardrailKind.EMPTY: ("I could not detect a question. Please try speaking again.", "मैं प्रश्न समझ नहीं पाया। कृपया दोबारा कोशिश करें।"),
    GuardrailKind.TOO_LONG: ("Your question is too long. Please keep it under 600 characters.", "आपका प्रश्न बहुत लंबा है। कृपया इसे छोटा करें।"),
    GuardrailKind.PROMPT_INJECTION: ("I cannot process that request.", "मैं यह अनुरोध संसाधित नहीं कर सकता।"),
    GuardrailKind.UNSAFE: ("I am not able to help with that.", "मैं इसमें मदद नहीं कर सकता।"),
    GuardrailKind.PII: ("I am not able to process personal or sensitive data.", "मैं व्यक्तिगत या संवेदनशील डेटा संसाधित नहीं कर सकता।"),
    GuardrailKind.OFF_TOPIC: (
        "That is outside what I can help with. Ask me any factual question about the knowledge base.",
        "यह मेरी क्षमता से बाहर है। नॉलेज बेस से जुड़ा कोई भी तथ्यात्मक प्रश्न पूछें।",
    ),
    None: (
        "I could not find a reliable answer in the knowledge base.",
        "मुझे नॉलेज बेस में विश्वसनीय उत्तर नहीं मिला।",
    ),
}

_UNGROUNDED_MESSAGE = (
    "I could not find a reliable answer for this in the knowledge base.",
    "मुझे इसका उत्तर नॉलेज बेस में विश्वसनीय रूप से नहीं मिला।",
)


class RAGHarness:
    """Asynchronous orchestrator for the voice RAG pipeline.

    Runs a fixed node sequence (STT -> guardrail -> embed -> retrieve ->
    generate -> ground) as tool calls with retries, circuit breaking and
    graceful error recovery, tracking per-stage latency throughout.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        engine: Optional[EmbeddingEngine] = None,
        store: Optional[VectorStoreCollection] = None,
        llm: Optional[object] = None,
        stt: Optional[object] = None,
        guardrails: Optional[GuardrailEngine] = None,
        grounding: Optional[GroundingEngine] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.engine = engine or EmbeddingEngine(self.settings.embedding_model, self.settings.embedding_dim)
        self.store = store
        if self.store is None:
            if self.settings.index_dir.exists():
                from .store import VectorStoreCollection as VSC

                self.store = VSC.load(self.settings.index_dir)
            else:
                self.store = VectorStoreCollection(self.settings.embedding_dim)
        self.mock = self.settings.mock_mode
        self.llm = llm or (MockLLM() if self.mock else GroqLLM(self.settings))
        self.stt = stt or (MockSTT() if self.mock else ElevenLabsSTT(self.settings))
        self.retry = RetryPolicy(self.settings.max_retries, self.settings.retry_base_delay_s)
        self._llm_circuit = CircuitBreaker(
            self.settings.circuit_failure_threshold, self.settings.circuit_reset_seconds
        )
        self._stt_circuit = CircuitBreaker(
            self.settings.circuit_failure_threshold, self.settings.circuit_reset_seconds
        )
        self.guardrails = guardrails or GuardrailEngine(self.settings.off_topic_score_threshold)
        self.grounding = grounding or GroundingEngine(
            self.settings.grounding_lexical_threshold, self.settings.grounding_embedding_threshold
        )

    @staticmethod
    def _is_hindi(text: str) -> bool:
        return bool(_DEVANAGARI.search(text))

    @staticmethod
    def _extract_answer(raw: str) -> tuple[str, Optional[float]]:
        parsed = json_safe_load(raw)
        if parsed and isinstance(parsed, dict) and "answer" in parsed:
            return str(parsed["answer"]).strip(), float(parsed.get("confidence") or 0.0)
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
        match = _JSON_OBJECT.search(text)
        if match:
            parsed = json_safe_load(match.group(0))
            if isinstance(parsed, dict) and "answer" in parsed:
                return str(parsed["answer"]).strip(), float(parsed.get("confidence") or 0.0)
        return text, None

    def _fallback(self, kind: Optional[GuardrailKind], hindi: bool) -> str:
        en, hi = _FALLBACK_MESSAGES.get(kind, _FALLBACK_MESSAGES[None])
        return hi if hindi else en

    async def run(
        self,
        query: Optional[str] = None,
        audio: Optional[bytes] = None,
        filename: str = "audio.mp3",
    ) -> RAGResult:
        latency = LatencyBreakdown()
        tool_calls: list[ToolCall] = []
        started = time.perf_counter()
        transcript: Optional[str] = None
        stt_result: Optional[SttResult] = None

        if audio is not None:
            try:
                stt_result = await self._stt_circuit.execute(
                    lambda: self.retry.run(lambda: self.stt.transcribe(audio, filename))
                )
                latency.stt_ms = stt_result.latency_ms
                transcript = stt_result.text
                query = transcript
                tool_calls.append(ToolCall(name="speech_to_text", latency_ms=latency.stt_ms))
            except Exception as exc:  # noqa: BLE001
                latency.total_end_to_end_ms = (time.perf_counter() - started) * 1000
                return RAGResult(
                    query=query or "",
                    transcript=None,
                    answer="Speech recognition failed. Please try again.",
                    status=Status.STT_ERROR,
                    grounded=False,
                    guardrail=GuardrailVerdict(allowed=False, kind=GuardrailKind.EMPTY, reason=str(exc)),
                    latency=latency,
                    error=str(exc),
                )

        text = (query or "").strip()

        t0 = time.perf_counter()
        verdict = self.guardrails.evaluate(text)
        latency.guardrail_ms = (time.perf_counter() - t0) * 1000
        if not verdict.allowed:
            latency.total_core_ms = latency.guardrail_ms
            latency.total_end_to_end_ms = (time.perf_counter() - started) * 1000
            return RAGResult(
                query=text,
                transcript=transcript,
                answer=self._fallback(verdict.kind, self._is_hindi(text)),
                status=Status.GUARDRAIL_REJECTED,
                grounded=False,
                guardrail=verdict,
                latency=latency,
                tool_calls=tool_calls,
            )

        if self.store.counts["children"] == 0:
            latency.total_end_to_end_ms = (time.perf_counter() - started) * 1000
            return RAGResult(
                query=text,
                transcript=transcript,
                answer="The knowledge base is empty. Run the indexer first.",
                status=Status.INTERNAL_ERROR,
                grounded=False,
                guardrail=verdict,
                latency=latency,
                error="empty index",
            )

        try:
            t0 = time.perf_counter()
            query_vec = await self.engine.embed_query(text)
            latency.embed_ms = (time.perf_counter() - t0) * 1000
            tool_calls.append(ToolCall(name="embed_query", latency_ms=latency.embed_ms))

            t0 = time.perf_counter()
            results = await self.store.search(
                query_vec,
                query_text=text,
                child_top_k=self.settings.child_search_top_k,
                parent_top_k=self.settings.parent_search_top_k,
            )
            latency.retrieval_ms = (time.perf_counter() - t0) * 1000
            tool_calls.append(ToolCall(name="retrieve_context", latency_ms=latency.retrieval_ms))

            passing = [r for r in results if r.score >= self.settings.retrieval_score_threshold]
            best_score = results[0].score if results else 0.0

            t0 = time.perf_counter()
            verdict = self.guardrails.evaluate(text, best_score=best_score)
            latency.guardrail_ms += (time.perf_counter() - t0) * 1000
            if not verdict.allowed:
                latency.total_core_ms = (
                    latency.guardrail_ms + latency.embed_ms + latency.retrieval_ms
                )
                latency.total_end_to_end_ms = (time.perf_counter() - started) * 1000
                return RAGResult(
                    query=text,
                    transcript=transcript,
                    answer=self._fallback(GuardrailKind.OFF_TOPIC, self._is_hindi(text)),
                    status=Status.OFF_TOPIC,
                    grounded=False,
                    guardrail=verdict,
                    latency=latency,
                    tool_calls=tool_calls,
                )

            contexts = passing[: self.settings.top_k]
            if not contexts:
                latency.total_core_ms = (
                    latency.guardrail_ms + latency.embed_ms + latency.retrieval_ms
                )
                latency.total_end_to_end_ms = (time.perf_counter() - started) * 1000
                return RAGResult(
                    query=text,
                    transcript=transcript,
                    answer=self._fallback(None, self._is_hindi(text)),
                    status=Status.NO_CONTEXT,
                    grounded=False,
                    guardrail=verdict,
                    latency=latency,
                    tool_calls=tool_calls,
                )

            hindi = self._is_hindi(text)
            context_block = "\n".join(
                f"[{i + 1}] ({c.language}) {c.text}" for i, c in enumerate(contexts)
            )
            if self.settings.llm_json_mode:
                format_rule = (
                    'Respond with a JSON object: {"answer": "...", "confidence": 0.0 to 1.0} '
                    'using "INSUFFICIENT_CONTEXT" as the answer if the context does not contain it.'
                )
            else:
                format_rule = (
                    'Respond with exactly "INSUFFICIENT_CONTEXT" if the context does not contain the answer.'
                )
            system_prompt = (
                "You are a grounded question-answering assistant. Answer the user's question "
                f"in {'Hindi' if hindi else 'English'}.\n"
                "Rules:\n"
                "1. Use ONLY the CONTEXT below. Do not use outside knowledge.\n"
                "2. Be concise (1-3 sentences).\n"
                "3. " + format_rule + "\n\n"
                "CONTEXT:\n" + context_block
            )

            t0 = time.perf_counter()
            try:
                raw_answer, llm_latency_ms = await self._llm_circuit.execute(
                    lambda: self.retry.run(lambda: self.llm.complete(system_prompt, text))
                )
            except Exception as exc:  # noqa: BLE001
                latency.generation_ms = (time.perf_counter() - t0) * 1000
                latency.total_core_ms = (
                    latency.guardrail_ms + latency.embed_ms + latency.retrieval_ms + latency.generation_ms
                )
                latency.total_end_to_end_ms = (time.perf_counter() - started) * 1000
                return RAGResult(
                    query=text,
                    transcript=transcript,
                    answer="I hit a temporary problem generating an answer. Please try again.",
                    status=Status.GENERATION_ERROR,
                    grounded=False,
                    guardrail=verdict,
                    latency=latency,
                    tool_calls=tool_calls,
                    error=str(exc),
                )
            latency.generation_ms = llm_latency_ms or (time.perf_counter() - t0) * 1000
            tool_calls.append(ToolCall(name="answer_question", latency_ms=latency.generation_ms))

            answer, llm_confidence = self._extract_answer(raw_answer)
            if not answer:
                answer = raw_answer or ""

            t0 = time.perf_counter()
            answer_vec = await self.engine.embed_query(answer)
            context_vecs = [await self.engine.embed_query(c.text) for c in contexts]
            grounded, score, note = self.grounding.check(answer, [c.text for c in contexts], answer_vec, context_vecs)
            latency.grounding_ms = (time.perf_counter() - t0) * 1000
            tool_calls.append(ToolCall(name="verify_grounding", latency_ms=latency.grounding_ms))

            if not grounded:
                latency.total_core_ms = (
                    latency.guardrail_ms + latency.embed_ms + latency.retrieval_ms + latency.generation_ms + latency.grounding_ms
                )
                latency.total_end_to_end_ms = (time.perf_counter() - started) * 1000
                return RAGResult(
                    query=text,
                    transcript=transcript,
                    answer=_UNGROUNDED_MESSAGE[0] if not hindi else _UNGROUNDED_MESSAGE[1],
                    status=Status.UNGROUNDED,
                    grounded=False,
                    guardrail=verdict,
                    contexts=contexts,
                    latency=latency,
                    tool_calls=tool_calls,
                    grounding_score=round(score, 4),
                )

            latency.total_core_ms = (
                latency.guardrail_ms + latency.embed_ms + latency.retrieval_ms + latency.generation_ms + latency.grounding_ms
            )
            latency.total_end_to_end_ms = (time.perf_counter() - started) * 1000
            return RAGResult(
                query=text,
                transcript=transcript,
                answer=answer,
                status=Status.SUCCESS,
                grounded=True,
                guardrail=verdict,
                contexts=contexts,
                latency=latency,
                tool_calls=tool_calls,
                llm_model=getattr(self.llm, "model", None),
                grounding_score=round(score, 4),
            )
        except Exception as exc:  # noqa: BLE001
            latency.total_core_ms = (
                latency.guardrail_ms + latency.embed_ms + latency.retrieval_ms + latency.generation_ms + latency.grounding_ms
            )
            latency.total_end_to_end_ms = (time.perf_counter() - started) * 1000
            return RAGResult(
                query=text,
                transcript=transcript,
                answer="Something went wrong while processing your question.",
                status=Status.INTERNAL_ERROR,
                grounded=False,
                guardrail=verdict,
                latency=latency,
                tool_calls=tool_calls,
                error=str(exc),
            )
