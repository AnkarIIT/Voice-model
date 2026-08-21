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
    CORS_ORIGINS,
    DEFAULT_K,
    INDEX_DIR,
    MAX_K,
    MAX_UPLOAD_MB,
    STT_PROVIDER,
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


def _inc_baddie():
    global _BADDIE_COUNT
    with _BADDIE_LOCK:
        _BADDIE_COUNT += 1


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

            lang = "hi-IN" if any("\u0900" <= ch <= "\u097F" for ch in clean) else "en-IN"
            speaker = "manisha" if lang == "hi-IN" else "anushka"
            client = SarvamAI(api_subscription_key=sarvam_tts_key)
            audio_chunks = client.text_to_speech.convert_stream(
                text=clean,
                language_code=lang,
                speaker=speaker,
                model="bulbul:v3",
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

        lang = "hi" if any("\u0900" <= ch <= "\u097F" for ch in clean) else "en"
        buf = io.BytesIO()
        gTTS(text=clean, lang=lang).write_to_fp(buf)
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


def _run_and_track(query_text, audio_path, language_code, index, k, use_rerank, stt_provider):
    out = run_pipeline(
        query_text=query_text,
        audio_path=audio_path,
        language_code=language_code,
        index=index,
        k=k,
        use_rerank=use_rerank,
        stt_provider=stt_provider,
    )
    if out.get("guardrail", {}).get("action") == "block":
        _inc_baddie()
    return out


@app.post("/query")
def query_text(
    query: str = Form(...),
    k: int = Form(DEFAULT_K),
    rerank: bool = Form(False),
):
    idx = get_index()
    if idx is None:
        raise HTTPException(503, "Index not built. Run build_index.py first.")
    out = _run_and_track(query_text=query, audio_path=None, language_code="hi-IN", index=idx, k=_clamp_k(k), use_rerank=bool(rerank), stt_provider=STT_PROVIDER)
    return JSONResponse(out)


@app.post("/query-stream")
async def query_stream(
    request: Request,
    query: str = Form(...),
    k: int = Form(DEFAULT_K),
    rerank: bool = Form(False),
):
    idx = get_index()
    if idx is None:
        raise HTTPException(503, "Index not built. Run build_index.py first.")

    async def event_stream():
        try:
            ret = retrieve(query, idx, _clamp_k(k), use_rerank=bool(rerank))
            chunks = ret["results"]
            g = check_guardrails(query, chunks)
            if g["action"] != "allow":
                yield f"event: guardrail\ndata: {json.dumps(g)}\n\n"
                return
            full = ""
            for chunk in generate_answer_stream(query, chunks):
                full += chunk
                yield 'event: token\ndata: ' + json.dumps({"text": chunk}) + '\n\n'
            yield 'event: done\ndata: ' + json.dumps({"text": full}) + '\n\n'
        except Exception as e:
            yield 'event: error\ndata: ' + json.dumps({"error": str(e)}) + '\n\n'

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/voice-query")
async def voice_query(
    file: UploadFile = File(...),
    language_code: str = Form("hi-IN"),
    k: int = Form(DEFAULT_K),
    stt_provider: str = Form("sarvam"),
):
    idx = get_index()
    if idx is None:
        raise HTTPException(503, "Index not built. Run build_index.py first.")
    tmp_path = await _save_upload(file)
    try:
        out = await run_pipeline_async(
            audio_path=tmp_path,
            language_code=language_code,
            index=idx,
            k=_clamp_k(k),
            stt_provider=stt_provider,
        )
        if out.get("guardrail", {}).get("action") == "block":
            _inc_baddie()
        return JSONResponse(out)
    finally:
        with suppress(OSError):
            tmp_path.unlink()


async def run_pipeline_async(**kwargs):
    import asyncio

    return await asyncio.to_thread(run_pipeline, **kwargs)
