# VoxRAG — Voice-First RAG for Indic Languages

**Phase:** Step 2 — STT Harness + Production Frontend  
**Stack:** FastAPI · Sarvam STT · FAISS · Sentence Transformers · Gemini/OpenAI LLM  
**Dataset:** [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) (Hindi + Bengali validation splits)

---

## What it does

VoxRAG lets users **talk to their documents** in Hindi, English, and Bengali. It:

1. **Transcribes** voice input via Sarvam Saarika v2.5 (cloud) or faster-whisper (local fallback)
2. **Retrieves** relevant chunks from a FAISS index built on MSMARCO-XI passages
3. **Answers** using Gemini / OpenAI / local LLM with strict context grounding
4. **Guardrails** unsafe content, low-confidence retrieval, and ungrounded answers

---

## Current Phase Status

| Component | Status | Notes |
|-----------|--------|-------|
| Sarvam STT | ✅ Working | Saarika v2.5, ~2-5s latency on real speech |
| faster-whisper fallback | ⚠️ Functional | `base` model hallucinates on Hinglish/code-mixed speech |
| FAISS index | ✅ Built | 49,912 vectors, 384-d, hybrid chunking |
| RAG retrieval | ✅ Working | Dense search + optional Cross-Encoder rerank |
| LLM generation | ⚠️ Needs API key | Gemini/OpenAI required; falls back to extractive |
| Guardrails | ✅ Working | Keyword block, abstain-on-low-score, lexical grounding |
| Frontend | ✅ Landing page | Split-screen hero, chat demo, voice UI |
| Tests | ✅ 31/31 passing | Unit tests for core logic |

---

## Architecture

```
[Audio Input] ──▶ [STT Harness] ──▶ [Query Text]
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
             [Text Query]                      [Voice Query]
                    │                                   │
                    ▼                                   ▼
          ┌─────────────────┐               ┌─────────────────┐
          │  orchestrator   │               │  main.py        │
          │  run_pipeline() │               │  /voice-query   │
          └────────┬────────┘               └────────┬────────┘
                   │                                │
                   ▼                                ▼
          ┌─────────────────┐               ┌─────────────────┐
          │  retrieval.py   │               │  harness.py     │
          │  FAISS + rerank │               │  retry + guard  │
          └────────┬────────┘               └────────┬────────┘
                   │                                │
                   ▼                                ▼
          ┌─────────────────┐               ┌─────────────────┐
          │  guardrails.py  │               │  stt.py         │
          │  screen + check │               │  Sarvam/Whisper │
          └────────┬────────┘               └─────────────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  llm.py         │
          │  Gemini/OpenAI/ │
          │  Local/fallback │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  JSON Response  │
          └─────────────────┘
```

---

## Tech Stack

| Layer | Technology | Details |
|-------|-----------|---------|
| **API** | FastAPI + Uvicorn | Async routes, CORS, upload handling |
| **STT** | Sarvam Saarika v2.5 | Primary; faster-whisper `base` fallback |
| **Embeddings** | SentenceTransformers | `paraphrase-multilingual-MiniLM-L12-v2` (384-d) |
| **Vector DB** | FAISS | IndexFlatIP, CPU/GPU, legacy `meta.pkl` + `meta.json` |
| **Reranking** | Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` (optional) |
| **LLM** | Gemini / OpenAI / Local | Extractive fallback if none available |
| **Frontend** | Tailwind CSS + Vanilla JS | Landing page + live demo UI |
| **Dataset** | MSMARCO-XI | `hinval_5000` + `benval_2000` sampled parquets |

---

## Run it

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
copy .env.example .env
# Edit .env and add your keys:
#   SARVAM_API_KEY=...
#   GEMINI_API_KEY=... (optional, for generation)
#   OPENAI_API_KEY=... (optional)

# 3. Build index (if not already built)
python build_index.py --rows 2000

# 4. Start server
python -m uvicorn app.main:app --port 8000

# 5. Open browser
# http://localhost:8000
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SARVAM_API_KEY` | *(empty)* | Sarvam STT key |
| `SARVAM_MODEL` | `saarika:v2.5` | Sarvam model ID |
| `WHISPER_MODEL` | `base` | Local fallback model (`tiny`, `base`, `small`, `medium`) |
| `WHISPER_DEVICE` | `cuda` | Device for local Whisper |
| `GEMINI_API_KEY` | *(empty)* | Google Gemini key |
| `OPENAI_API_KEY` | *(empty)* | OpenAI key |
| `LLM_MODEL` | `gemini-1.5-flash` | Preferred LLM model |
| `USE_LOCAL_LLM` | `0` | Enable local HF model |
| `EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | SentenceTransformer |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder |
| `MAX_UPLOAD_MB` | `25` | Audio upload limit |
| `DEFAULT_K` | `5` | Default retrieval count |
| `ABSTAIN_THRESHOLD` | `0.35` | Min similarity for answer |
| `GROUNDING_HIT_RATE` | `0.4` | Min lexical overlap for grounded answer |

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Index status, provider, vector count |
| `POST` | `/transcribe` | Upload audio → text (provider override supported) |
| `POST` | `/query` | Text RAG pipeline |
| `POST` | `/voice-query` | Audio → STT → RAG pipeline |

---

## Tests

```powershell
python -m pytest tests/ -q
```

**31 tests** covering:
- Chunking (fixed-size, semantic, hybrid, deduplication)
- Preprocessing (HTML strip, Unicode normalize)
- Guardrails (unsafe keywords, short queries, low similarity, hallucination check)
- Harness (success, empty, retries, permanent failure)
- Orchestrator (full pipeline: allow, block, abstain, reject)

---

## Project Structure

```
C:/Codes/ai/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app, routes, upload handling
│   ├── config.py          # Environment variables, logging
│   ├── stt.py             # Sarvam, ElevenLabs, faster-whisper providers
│   ├── harness.py         # STT retry/backoff, guardrail on transcript
│   ├── orchestrator.py    # Wires STT → retrieval → guardrails → LLM
│   ├── retrieval.py       # FAISS search, cross-encoder rerank
│   ├── embed.py           # SentenceTransformer + FAISS index CRUD
│   ├── chunking.py        # Fixed-size, semantic, hybrid chunking
│   ├── preprocess.py      # Text cleaning, Unicode normalize
│   ├── guardrails.py      # Safety, abstain, hallucination checks
│   ├── llm.py             # Gemini/OpenAI/local generation + fallback
│   └── static/
│       └── index.html     # Landing page + demo UI
├── data/
│   ├── sample_hinval_5000.parquet
│   └── sample_benval_2000.parquet
├── index_hinval/
│   ├── faiss.index        # 49,912 vectors
│   └── meta.pkl           # Legacy metadata
├── tests/                 # 31 unit tests
├── build_index.py         # Index builder
├── pull_dataset.py        # HF dataset downloader
├── test_stt.py            # Standalone STT test script
├── bench_latency.py       # Latency benchmarking
├── requirements.txt
├── .env
└── .env.example
```

---

## Known Issues

| Issue | Severity | Status |
|-------|----------|--------|
| faster-whisper `base` hallucinates on Hinglish | 🔴 High | Documented; use Sarvam primary |
| `/voice-query` hardcodes `sarvam` provider | 🟠 Medium | Provider override not exposed |
| No VAD / audio transcoding before local STT | 🟠 Medium | Planned for Step 3 |
| No TTS output endpoint | 🟡 Medium | Planned |
| `meta.pkl` legacy format (no `meta.json`) | 🟡 Medium | Works but should migrate |
| `llm.py` silent fallback between providers | 🟡 Medium | Hard to debug in production |

---

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **Step 1** | Dataset, chunking, FAISS index, basic RAG | ✅ Done |
| **Step 2** | STT harness (Sarvam + Whisper), guardrails, frontend | ✅ Current |
| **Step 3** | VAD, audio transcoding, TTS output, improved hallucination handling | ⏳ Next |
| **Step 4** | Multi-turn conversation, streaming, auth, deployment | 📋 Planned |

---

## License

MIT
