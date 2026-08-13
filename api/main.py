from __future__ import annotations

import base64
import json
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from voice_rag.benchmark import _percentiles
from voice_rag.config import get_settings
from voice_rag.harness import RAGHarness
from voice_rag.models import RAGResult, Status


class AskRequest(BaseModel):
    query: str


class AskVoiceRequest(BaseModel):
    audio_base64: str
    filename: str = "audio.mp3"


class LiveTracker:
    def __init__(self, maxlen: int = 500) -> None:
        self._core = deque(maxlen=maxlen)
        self._e2e = deque(maxlen=maxlen)
        self._status = deque(maxlen=maxlen)

    def record(self, result: RAGResult) -> None:
        self._core.append(result.latency.total_core_ms)
        self._e2e.append(result.latency.total_end_to_end_ms)
        self._status.append(result.status.value)

    def snapshot(self) -> dict:
        return {
            "live_core_ms": _percentiles(list(self._core)),
            "live_end_to_end_ms": _percentiles(list(self._e2e)),
            "status_counts": {s: list(self._status).count(s) for s in set(self._status)},
        }


settings = get_settings()
harness: RAGHarness = None
tracker = LiveTracker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global harness
    effective = settings
    if not settings.groq_api_key:
        effective = settings.model_copy(update={"mock_mode": True})
    harness = RAGHarness(effective)
    if not harness.store.counts["children"]:
        print("[api] WARNING: index is empty - run `python -m voice_rag.indexer` first")
    yield


app = FastAPI(title="Voice RAG - MSMARCO-XI", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "index": harness.store.counts,
        "llm": getattr(harness.llm, "model", None),
        "stt_available": harness.stt.available,
        "llm_available": harness.llm.available if hasattr(harness.llm, "available") else True,
        "mock_mode": settings.mock_mode or not settings.groq_api_key,
    }


@app.post("/api/ask", response_model=RAGResult)
async def ask(req: AskRequest):
    result = await harness.run(query=req.query)
    tracker.record(result)
    return result


@app.post("/api/ask/voice", response_model=RAGResult)
async def ask_voice(file: UploadFile = File(...)):
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="empty audio file")
    result = await harness.run(audio=audio, filename=file.filename or "audio.mp3")
    tracker.record(result)
    return result


@app.post("/api/ask/voice_base64", response_model=RAGResult)
async def ask_voice_base64(req: AskVoiceRequest):
    try:
        audio = base64.b64decode(req.audio_base64)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid base64: {exc}") from exc
    result = await harness.run(audio=audio, filename=req.filename)
    tracker.record(result)
    return result


@app.get("/api/metrics")
async def metrics():
    benchmark_path = settings.metrics_dir / "latency_benchmark.json"
    benchmark = None
    if benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    return {"benchmark": benchmark, "live": tracker.snapshot()}


@app.websocket("/ws/ask")
async def ws_ask(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = json.loads(await websocket.receive_text())
            if "text" in payload:
                await websocket.send_json({"event": "status", "message": "processing text query"})
                result = await harness.run(query=payload["text"])
            elif "audio" in payload:
                audio = base64.b64decode(payload["audio"])
                await websocket.send_json({"event": "status", "message": "transcribing audio"})
                result = await harness.run(audio=audio, filename=payload.get("filename", "audio.mp3"))
            else:
                await websocket.send_json({"event": "error", "message": "send text or audio"})
                continue
            tracker.record(result)
            await websocket.send_json(
                {
                    "event": "result",
                    "status": result.status.value,
                    "transcript": result.transcript,
                    "answer": result.answer,
                    "grounded": result.grounded,
                    "guardrail": result.guardrail.kind.value if not result.guardrail.allowed else None,
                    "guardrail_reason": result.guardrail.reason or None,
                    "contexts": [c.model_dump() for c in result.contexts],
                    "latency": result.latency.model_dump(),
                    "tool_calls": [t.model_dump() for t in result.tool_calls],
                    "grounding_score": result.grounding_score,
                    "error": result.error,
                }
            )
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        await websocket.send_json({"event": "error", "message": str(exc)})
