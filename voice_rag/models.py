from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Status(str, Enum):
    SUCCESS = "SUCCESS"
    GUARDRAIL_REJECTED = "GUARDRAIL_REJECTED"
    OFF_TOPIC = "OFF_TOPIC"
    NO_CONTEXT = "NO_CONTEXT"
    UNGROUNDED = "UNGROUNDED"
    STT_ERROR = "STT_ERROR"
    GENERATION_ERROR = "GENERATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class GuardrailKind(str, Enum):
    NONE = "NONE"
    EMPTY = "EMPTY"
    TOO_LONG = "TOO_LONG"
    UNSAFE = "UNSAFE"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    PII = "PII"
    OFF_TOPIC = "OFF_TOPIC"


class LatencyBreakdown(BaseModel):
    stt_ms: float = 0.0
    guardrail_ms: float = 0.0
    embed_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    grounding_ms: float = 0.0
    total_core_ms: float = 0.0
    total_end_to_end_ms: float = 0.0


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    language: str
    source_kind: str
    selected: bool
    score: float


class GuardrailVerdict(BaseModel):
    allowed: bool
    kind: GuardrailKind = GuardrailKind.NONE
    reason: str = ""
    score: float = 0.0


class ToolCall(BaseModel):
    name: str
    latency_ms: float = 0.0


class RAGResult(BaseModel):
    query: str
    transcript: Optional[str] = None
    answer: str
    status: Status
    grounded: bool
    guardrail: GuardrailVerdict = Field(default_factory=GuardrailVerdict)
    contexts: list[RetrievedChunk] = Field(default_factory=list)
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    llm_model: Optional[str] = None
    grounding_score: Optional[float] = None
    error: Optional[str] = None


class SttResult(BaseModel):
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    duration_ms: Optional[float] = None
    latency_ms: float = 0.0
