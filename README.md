# HH Goa 2026 — Task 2: Voice-Enabled RAG over MSMARCO-XI

A voice-first Retrieval-Augmented Generation (RAG) system built for the **HH Goa 2026** hackathon. It
transcribes Hindi/English speech (ElevenLabs STT), retrieves context from the
[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) corpus using hybrid
dense + lexical search, and produces grounded answers with a Groq LLM — all behind latency analytics,
guardrails, grounding checks, circuit breakers and retries.

## Architecture

```
[Voice] --STT--> [Query] --Guardrail--> --Embed--> --Hybrid Retrieval--> --LLM--> --Grounding--> [Answer]
             ElevenLabs       1ms         7ms           5ms                  ~100ms     60ms
```

### Pipeline nodes (`voice_rag/harness.py`)
1. **STT** — ElevenLabs `scribe_v1` (`voice_rag/providers.py`), measured separately as `stt_ms`.
2. **Guardrail** — deterministic patterns for prompt-injection, unsafe content, PII harvesting and
   off-topic intents, plus a retrieval-score relevance gate (`voice_rag/guardrails.py`).
3. **Embedding** — `paraphrase-multilingual-MiniLM-L12-v2` (384-d, multilingual EN/HI) via FastEmbed.
4. **Retrieval** — hybrid search (`voice_rag/store.py`):
   - *Dense*: in-memory IVF store (`k`-means coarse clusters + probe search) on a **parent-child
     two-tier** index (`voice_rag/vectordb.py`): fine-grained children for precision, wide parents
     for context; results merged by source.
   - *Lexical*: lightweight **Okapi BM25** over the child layer (`voice_rag/bm25.py`), built once at
     index time, queried in ~1ms.
   - *Fusion*: **reciprocal-rank fusion (RRF)** of the dense and BM25 rankings. This is what makes
     factoid queries robust on this noisy corpus — pure dense ranking surfaces generic medical
     boilerplate above true matches.
5. **Generation** — Groq `llama-3.1-8b-instant` in JSON mode, restricted to retrieved context only.
6. **Grounding** — refusal markers + lexical coverage + embedding overlap; ungrounded answers are
   replaced with an explicit refusal (`voice_rag/guardrails.py`).

Reliability: per-provider **circuit breakers** and a harness-level **retry policy**; every node is
recorded as a tool call with latency (`LatencyBreakdown`).

### Latency methodology
The **200ms budget applies to the core pipeline** (guardrail → embed → retrieve → generate → ground).
STT is provider-network-bound (upload to ElevenLabs), so it is tracked separately (`stt_ms`) and
excluded from `total_core_ms`.

Measured on the bundled index (mock LLM, real embeddings/retrieval):

| metric | P50 | P70 | P95 |
|---|---|---|---|
| core pipeline | 74.5ms | 83.3ms | 102.7ms |
| retrieval (hybrid) | 5.2ms | 5.9ms | 8.6ms |
| grounding | 61.0ms | 70.4ms | 87.8ms |

Status across 120 queries: 116 SUCCESS / 4 GUARDRAIL_REJECTED, grounded avg 0.93.

## Index

Built from `data/raw/hinval.parquet` (440MB Hindi validation split, 97,941 rows). The indexer
(`voice_rag/indexer.py`) samples rows, extracts English + Hindi passages and answers, applies
adaptive chunking (fixed-size → semantic → parent-child with oversize fallback), embeds with a
multi-process pool, builds IVF + BM25, and saves to `data/index/msmarco_hi_val_v1/`.

- 1,500 sampled rows → 32,928 documents → 35,908 child + 1,970 parent vectors.
- Most passages are atomic (≤160 words); long passages get parent-child semantic splits.
- **Index build time ~20 min** on this machine: embedding is the bottleneck (~35 vec/s on
  real-length texts); lowering `INDEX_MAX_ROWS` scales build time linearly.

## Run it

```powershell
# 1. install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. download the dataset (once)
#    https://huggingface.co/datasets/ai4bharat/MSMARCO-XI
#    → 'hinval.parquet' into data/raw/  (or set INDEX_PARQUET in .env)

# 3. build the index
#    (the built index also exists locally at data/index/msmarco_hi_val_v1/
#     from this session; data/index/ is gitignored as a build artifact)
.\.venv\Scripts\python.exe -m voice_rag.indexer --max-rows 1500

# 4. benchmark
.\.venv\Scripts\python.exe -m voice_rag.benchmark

# 5. API + UI
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
# → http://localhost:8000  (mic-enabled chat UI)
```

### API
| endpoint | purpose |
|---|---|
| `POST /api/ask` | `{ "query": "..." }` → RAG result |
| `POST /api/ask/voice` | multipart `file` (audio) → STT + RAG |
| `POST /api/ask/voice_base64` | `{ "audio_base64": "...", "filename": "..." }` |
| `GET /api/metrics` | benchmark report + live request stats |
| `GET /api/health` | index size, provider availability, mock mode |
| `WS /ws/ask` | streaming request/response |

### Keys & mock mode
Copy `.env.example` → `.env` and fill `GROQ_API_KEY` / `ELEVENLABS_API_KEY`. Without keys the app
**auto-falls back to deterministic mock providers** (`MOCK_MODE=1`) so the full flow (including STT,
grounding and guardrails) runs keyless — benchmark and `/api/health` report `mock_mode: true`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q   # 37 tests
```

Coverage: chunking (fixed/semantic/parent-child/oversize), IVF + save/load, BM25 + RRF hybrid
retrieval, guardrails (injection/unsafe/PII/off-topic/data-destruction), grounding, harness paths
(success, rejected, off-topic, no-context, ungrounded, STT error/success, retries, generation error,
circuit open, empty query) and mock provider behavior.

## Files

```
voice_rag/
  harness.py      orchestrator (guardrails, circuits, retries, latency)
  indexer.py      corpus → adaptive chunks → IVF + BM25 index
  chunking.py     fixed-size / semantic / parent-child chunkers
  embeddings.py   FastEmbed engine + process-pool batch embedding
  vectordb.py     in-memory IVF (k-means coarse clusters + probe search)
  bm25.py         Okapi BM25 term index (CSC) for hybrid retrieval
  store.py        two-tier parent/child store, RRF fusion, save/load
  guardrails.py   pre/post generation safety + grounding engine
  providers.py    Groq LLM, ElevenLabs STT, circuit breaker, retry policy, mocks
  benchmark.py    P50/P70/P95/P100 latency reporter + challenge queries
  models.py       typed pipeline contracts
api/              FastAPI app + mic-enabled web UI
tests/            37 tests
```

## Submission checklist
- [x] Dataset: ai4bharat/MSMARCO-XI (Hindi + English passages, 97,941-row validation split)
- [x] Voice: ElevenLabs STT (web mic + REST upload)
- [x] Retrieval: hybrid (IVF-dense + BM25, RRF), parent-child chunking
- [x] Generation: Groq Llama 3.1 8B, JSON mode, context-restricted
- [x] Grounding + guardrails + circuit breakers + retries
- [x] Latency analytics (core vs STT) + benchmark report `data/metrics/latency_benchmark.json`
- [x] Tests (37) and live demo UI
```
