from __future__ import annotations

import asyncio
import json
import random
import re
import time
from typing import Any, Awaitable, Callable, Optional

from .config import Settings
from .models import SttResult


class CircuitBreaker:
    """Simple circuit breaker with closed / open / half-open states."""

    def __init__(self, failure_threshold: int = 5, reset_seconds: int = 30) -> None:
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        return (time.monotonic() - self._opened_at) < self.reset_seconds

    async def execute(self, call: Callable[[], Awaitable[Any]]) -> Any:
        async with self._lock:
            if self.is_open:
                raise CircuitOpenError("circuit is open")
            if self._opened_at is not None:
                self._opened_at = None
                self._failures = 0
        try:
            result = await call()
        except Exception:
            async with self._lock:
                self._failures += 1
                if self._failures >= self.failure_threshold:
                    self._opened_at = time.monotonic()
            raise
        async with self._lock:
            self._failures = 0
        return result


class CircuitOpenError(RuntimeError):
    pass


class RetryPolicy:
    """Exponential backoff with jitter over a set of retryable outcomes."""

    def __init__(self, max_retries: int = 3, base_delay_s: float = 0.3) -> None:
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s

    @staticmethod
    def is_retryable(exc: BaseException) -> bool:
        message = str(exc).lower()
        if isinstance(exc, CircuitOpenError):
            return False
        if any(tag in message for tag in ("429", "rate", "timeout", "timed out", "503", "502", "504", "overloaded", "temporarily unavailable", "network error", "connect")):
            return True
        return False

    async def run(self, call: Callable[[], Awaitable[Any]]) -> Any:
        attempt = 0
        while True:
            try:
                return await call()
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt > self.max_retries or not self.is_retryable(exc):
                    raise
                delay = self.base_delay_s * (2 ** (attempt - 1)) + random.uniform(0, 0.05)
                await asyncio.sleep(delay)


class GroqLLM:
    def __init__(self, settings: Settings, circuit_breaker: Optional[CircuitBreaker] = None) -> None:
        self.settings = settings
        self.model = settings.groq_model
        self.retry = RetryPolicy(settings.max_retries, settings.retry_base_delay_s)
        self.circuit = circuit_breaker or CircuitBreaker(
            settings.circuit_failure_threshold, settings.circuit_reset_seconds
        )
        self._client: Any = None
        self.available = bool(settings.groq_api_key)

    def _get_client(self) -> Any:
        if self._client is None:
            from groq import AsyncGroq

            self._client = AsyncGroq(api_key=self.settings.groq_api_key, timeout=self.settings.llm_timeout_s)
        return self._client

    async def complete(self, system: str, user: str) -> tuple[str, float]:
        return await self.complete_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

    async def complete_messages(self, messages: list[dict]) -> tuple[str, float]:
        started = time.perf_counter()
        client = self._get_client()
        use_json = self.settings.llm_json_mode

        async def call() -> str:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.settings.max_answer_tokens + (48 if use_json else 0),
                "temperature": self.settings.llm_temperature,
            }
            if use_json:
                kwargs["response_format"] = {"type": "json_object"}
            resp = await client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()

        async def guarded() -> str:
            return await self.circuit.execute(lambda: self.retry.run(call))

        text = await guarded()
        latency_ms = (time.perf_counter() - started) * 1000
        return text, latency_ms


class ElevenLabsSTT:
    def __init__(self, settings: Settings, circuit_breaker: Optional[CircuitBreaker] = None) -> None:
        self.settings = settings
        self.model = settings.elevenlabs_stt_model
        self.retry = RetryPolicy(settings.max_retries, settings.retry_base_delay_s)
        self.circuit = circuit_breaker or CircuitBreaker(
            settings.circuit_failure_threshold, settings.circuit_reset_seconds
        )
        self._client: Any = None
        self.available = bool(settings.elevenlabs_api_key)

    def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self.settings.stt_timeout_s)
        return self._client

    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.mp3") -> SttResult:
        started = time.perf_counter()
        client = self._get_client()

        async def call() -> dict:
            resp = await client.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": self.settings.elevenlabs_api_key},
                data={"model_id": self.model},
                files={"file": (filename, audio_bytes, "audio/mpeg")},
            )
            resp.raise_for_status()
            return resp.json()

        async def guarded() -> dict:
            return await self.circuit.execute(lambda: self.retry.run(call))

        payload = await guarded()
        text = (payload.get("text") or "").strip()
        if not text:
            raise RuntimeError("STT returned empty transcript")
        return SttResult(
            text=text,
            language=payload.get("language"),
            confidence=payload.get("language_probability"),
            duration_ms=payload.get("duration"),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class MockLLM:
    """Deterministic extractive answerer used in MOCK_MODE and tests.

    Returns the first sentence of the best context (grounded by construction),
    or an explicit refusal marker when no context is supplied.
    """

    available = True
    model = "mock-extractive"

    async def complete(self, system: str, user: str) -> tuple[str, float]:
        return await self.complete_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

    async def complete_messages(self, messages: list[dict]) -> tuple[str, float]:
        await asyncio.sleep(0)
        system = messages[0]["content"]
        if system.strip().startswith("INSUFFICIENT_CONTEXT"):
            return "INSUFFICIENT_CONTEXT", 1.0
        context_marker = system.rfind("CONTEXT:")
        if context_marker == -1:
            return "INSUFFICIENT_CONTEXT", 1.0
        context_block = system[context_marker + len("CONTEXT:"):].strip()
        first_line = re.sub(r"^\[\d+\]\s*\([^)]*\)\s*", "", context_block.split("\n")[0].strip())
        first_sentence = first_line.split(".")[0].strip()
        if not first_sentence:
            return "INSUFFICIENT_CONTEXT", 1.0
        return first_sentence, 1.0


class MockSTT:
    """Fake STT that derives a transcript from the filename, for MOCK_MODE/tests."""

    available = True

    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.mp3") -> SttResult:
        await asyncio.sleep(0)
        return SttResult(text=filename.replace("_", " "), latency_ms=1.0)


def json_safe_load(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
