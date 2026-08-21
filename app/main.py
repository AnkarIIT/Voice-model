import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from .config import (
    AUDIO_EXTENSIONS,
    BASE_DIR,
    CORS_ORIGINS,
    DEFAULT_K,
    INDEX_DIR,
    MAX_K,
    MAX_UPLOAD_MB,
    STT_PROVIDER,
    TTS_MODEL,
    TTS_SPEAKER_EN,
    TTS_SPEAKER_HI,
    WHISPER_DEVICE,
    setup_logging,
)
from .embed import FaissIndex
from .harness import atranscribe_with_harness
from .guardrails import check_guardrails
from .llm import generate_answer, generate_answer_stream
from .orchestrator import run_pipeline
from .retrieval import retrieve

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice RAG - Full Harness", version="0.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_INDEX = None
_INDEX_LOCK = threading.Lock()
_BADDIE_COUNT = 0
_BADDIE_LOCK = threading.Lock()
_HISTORY: dict[str, list] = {}
_HISTORY_LOCK = threading.Lock()
MAX_HISTORY = 10


def _inc_baddie():
    global _BADDIE_COUNT
    with _BADDIE_LOCK:
        _BADDIE_COUNT += 1


def _get_history(session_id: str) -> list:
    with _HISTORY_LOCK:
        return list(_HISTORY.get(session_id, []))


def _add_history(session_id: str, query: str, answer: str):
    with _HISTORY_LOCK:
        hist = _HISTORY.setdefault(session_id, [])
        hist.append({"role": "user", "text": query})
        hist.append({"role": "assistant", "text": answer})
        if len(hist) > MAX_HISTORY:
            del hist[: len(hist) - MAX_HISTORY]


def get_index() -> FaissIndex:
    global _INDEX
    if _INDEX is None:
        with _INDEX_LOCK:
            if _INDEX is None:
                if not (INDEX_DIR / "faiss.index").exists():
                    logger.error("index not found at %s; run build_index.py", INDEX_DIR)
                    return None
                _INDEX = FaissIndex.load(INDEX_DIR, device=WHISPER_DEVICE)
    return _INDEX


@app.on_event("startup")
def warmup():
    try:
        idx = get_index()
        if idx is not None:
            logger.info("index ready: ntotal=%d", idx.index.ntotal)
    except Exception as e:
        logger.exception("index warmup failed: %s", e)


def _clamp_k(k: int) -> int:
    return max(1, min(int(k), MAX_K))


def health_payload():
    loaded = _INDEX is not None
    return {
        "status": "ok",
        "provider": STT_PROVIDER,
        "index_loaded": loaded,
        "index_available": (INDEX_DIR / "faiss.index").exists(),
        "ntotal": _INDEX.index.ntotal if loaded else 0,
        "baddie_detectors": _BADDIE_COUNT,
    }


async def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix and suffix not in AUDIO_EXTENSIONS:
        raise HTTPException(400, f"unsupported audio type {suffix!r}")
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".wav") as tmp:
            tmp_path = Path(tmp.name)
            size = 0
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_MB}MB limit")
                tmp.write(chunk)
        return tmp_path
    except HTTPException:
        if tmp_path:
            with suppress(OSError):
                tmp_path.unlink()
        raise


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/tester", include_in_schema=False)
def tester():
    return FileResponse(Path(__file__).parent / "static" / "tester.html")


def _detect_tts_lang(text: str) -> str:
    for ch in text:
        if "\u0900" <= ch <= "\u097F":
            return "hi-IN"
        if "\u0980" <= ch <= "\u09FF":
            return "bn-IN"
    return "en-IN"


@app.post("/speak")
def speak(text: str = Form(...)):
    import io

    clean = text.strip()[:1000]
    if not clean:
        raise HTTPException(400, "empty text")
    # Try Sarvam TTS first for better Indic/Hinglish quality
    sarvam_tts_key = os.getenv("SARVAM_API_KEY", "").strip()
    if sarvam_tts_key:
        try:
            from sarvamai import SarvamAI

            lang = _detect_tts_lang(clean)
            speaker = {"hi-IN": TTS_SPEAKER_HI, "bn-IN": TTS_SPEAKER_HI}.get(lang, TTS_SPEAKER_EN)
            client = SarvamAI(api_subscription_key=sarvam_tts_key)
            audio_chunks = client.text_to_speech.convert_stream(
                text=clean,
                language_code=lang,
                speaker=speaker,
                model=TTS_MODEL,
                output_audio_codec="mp3",
                output_audio_bitrate="128k",
            )
            buf = io.BytesIO(b"".join(audio_chunks))
            buf.seek(0)
            return Response(content=buf.read(), media_type="audio/mpeg")
        except Exception as e:
            logger.warning("Sarvam TTS failed (%s); falling back to gTTS", e)

    # Fallback: gTTS
    try:
        from gtts import gTTS

        gtts_lang = _detect_tts_lang(clean).split("-")[0]
        buf = io.BytesIO()
        gTTS(text=clean, lang=gtts_lang).write_to_fp(buf)
    except Exception as e:
        logger.warning("TTS failed: %s", e)
        raise HTTPException(502, "text-to-speech failed upstream")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/mpeg")


@app.get("/health")
def health():
    return health_payload()


@app.get("/baddies")
def baddies():
    return {"baddie_detectors": _BADDIE_COUNT}


@app.get("/history")
def history_endpoint(session_id: str = "default"):
    turns = _get_history(session_id)
    return {"session_id": session_id, "turns": turns, "count": len(turns) // 2}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language_code: str = Form("hi-IN"),
    provider: str = Form(None),
):
    prov = provider or STT_PROVIDER
    tmp_path = await _save_upload(file)
    try:
        t0 = time.perf_counter()
        result = await atranscribe_with_harness(tmp_path, language_code, prov)
        result["endpoint_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["filename"] = file.filename
        if result["status"] == "error":
            logger.error("transcription failed: %s", result.get("error"))
            raise HTTPException(502, "transcription failed upstream")
        return JSONResponse(result)
    finally:
        with suppress(OSError):
            tmp_path.unlink()


def _run_and_track(query_text, audio_path, language_code, index, k, use_rerank, stt_provider, session_id=None, conversation_history=None):
    out = run_pipeline(
        query_text=query_text,
        audio_path=audio_path,
        language_code=language_code,
        index=index,
        k=k,
        use_rerank=use_rerank,
        stt_provider=stt_provider,
        conversation_history=conversation_history,
    )
    if out.get("guardrail", {}).get("action") == "block":
        _inc_baddie()
    if session_id and out.get("answer"):
        _add_history(session_id, query_text or "", out["answer"])
    out["session_id"] = session_id
    return out


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/query")
def query_text(
    query: str = Form(...),
    k: int = Form(DEFAULT_K),
    rerank: bool = Form(False),
    session_id: str = Form(None),
):
    idx = get_index()
    if idx is None:
        raise HTTPException(503, "Index not built. Run build_index.py first.")
    history = _get_history(session_id)
    out = _run_and_track(query_text=query, audio_path=None, language_code="hi-IN", index=idx, k=_clamp_k(k), use_rerank=bool(rerank), stt_provider=STT_PROVIDER, session_id=session_id, conversation_history=history)
    return JSONResponse(out)


@app.post("/query-stream")
async def query_stream(
    request: Request,
    query: str = Form(...),
    k: int = Form(DEFAULT_K),
    rerank: bool = Form(False),
    session_id: str = Form(None),
):
    idx = get_index()
    if idx is None:
        raise HTTPException(503, "Index not built. Run build_index.py first.")

    def _build() -> dict:
        ret = retrieve(query, idx, _clamp_k(k), use_rerank=bool(rerank))
        return ret

    async def event_stream():
        t0 = time.perf_counter()
        try:
            ret = await asyncio.to_thread(_build)
            chunks = ret["results"]
            history = _get_history(session_id)
            g = check_guardrails(query, chunks, has_history=bool(history))
            yield _sse("stage", {
                "stage": "retrieval",
                "retrieved": chunks,
                "search_ms": ret["search_ms"],
                "total_ms": ret["total_ms"],
                "reranked": ret["reranked"],
            })
            if g["action"] != "allow":
                out = {
                    "status": g["action"],
                    "query": query,
                    "stt": {"text": query, "provider_used": "text-input"},
                    "retrieved": [],
                    "retrieval": {"search_ms": ret["search_ms"], "total_ms": ret["total_ms"], "reranked": ret["reranked"]},
                    "answer": g.get("answer", ""),
                    "guardrail": g,
                    "hallucination": {"grounded": False},
                    "provider": "guardrail",
                    "timings": {"retrieval_ms": round(ret["total_ms"], 2), "llm_ms": 0, "total_ms": round((time.perf_counter() - t0) * 1000, 2)},
                    "session_id": session_id,
                }
                yield _sse("guardrail", g)
                yield _sse("done", out)
                return
            full = ""
            llm_t0 = time.perf_counter()
            for chunk in generate_answer_stream(query, chunks, conversation_history=history):
                full += chunk
                yield _sse("token", {"text": chunk})
            from .guardrails import answer_draws_on_history, hallucination_check

            h = hallucination_check(full, chunks, encoder=getattr(idx, "model", None))
            if not h.get("grounded") and answer_draws_on_history(full, history):
                h = {**h, "grounded": True, "method": "history_follow_up"}
            total = (time.perf_counter() - t0) * 1000
            llm_ms = (time.perf_counter() - llm_t0) * 1000
            prov = f"gemini:{os.getenv('LLM_MODEL', 'gemini-3.5-flash-lite')}"
            out = {
                "status": "ok",
                "query": query,
                "stt": {"text": query, "provider_used": "text-input"},
                "retrieved": chunks,
                "retrieval": {"search_ms": ret["search_ms"], "total_ms": ret["total_ms"], "reranked": ret["reranked"]},
                "answer": full,
                "guardrail": g,
                "hallucination": h,
                "provider": prov,
                "timings": {"retrieval_ms": round(ret["total_ms"], 2), "llm_ms": round(llm_ms, 2), "total_ms": round(total, 2)},
                "session_id": session_id,
            }
            _add_history(session_id, query, full)
            yield _sse("done", out)
        except Exception as e:
            logger.exception("stream failed")
            yield _sse("error", {"error": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/voice-query")
async def voice_query(
    file: UploadFile = File(...),
    language_code: str = Form("hi-IN"),
    k: int = Form(DEFAULT_K),
    stt_provider: str = Form("sarvam"),
    session_id: str = Form(None),
):
    idx = get_index()
    if idx is None:
        raise HTTPException(503, "Index not built. Run build_index.py first.")
    tmp_path = await _save_upload(file)
    try:
        history = _get_history(session_id)
        out = await run_pipeline_async(
            audio_path=tmp_path,
            language_code=language_code,
            index=idx,
            k=_clamp_k(k),
            stt_provider=stt_provider,
            session_id=session_id,
            conversation_history=history,
        )
        if out.get("guardrail", {}).get("action") == "block":
            _inc_baddie()
        return JSONResponse(out)
    finally:
        with suppress(OSError):
            tmp_path.unlink()


@app.get("/bench")
def bench():
    p = BASE_DIR / "bench_results.json"
    if not p.exists():
        return {"available": False}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {"available": True, "summary": data.get("summary", {})}
    except Exception:
        return {"available": False}


async def run_pipeline_async(**kwargs):
    import asyncio

    return await asyncio.to_thread(run_pipeline, **kwargs)
