# Step 2 — STT Harness (Sarvam chosen)

## Why Sarvam
MSMARCO-XI = 14 Indic languages. Sarvam Saarika v2.5 supports hi-IN, bn-IN etc natively. ElevenLabs is English-centric. So default provider=sarvam, fallback=local whisper if no key.

## Harness Features
- structured I/O: {text, language, confidence, latency_ms, total_ms, provider, guardrail}
- retries x3 with backoff + jitter; non-retryable errors (missing key, auth) fail fast
- guardrail: empty/unsafe check (shared with app/guardrails.py)
- latency analytics per request

## Endpoints (app/main.py)
- GET  /health       — status, index availability
- POST /transcribe   — async audio -> text
- POST /query        — text RAG pipeline
- POST /voice-query  — audio RAG pipeline

## Run
```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000
```
Local whisper needs ffmpeg on PATH. Test: `python test_stt.py` -> voice (gTTS) -> transcribe -> print

## Verified
- gTTS "what is a corporation?" -> whisper tiny CPU 139ms, base 600ms (after warmup). Saarika would be ~300ms API.
