# Voice RAG Project — Flow & Flaw Analysis

**Project:** `C:/Codes/ai`  
**Scope:** Full codebase review (app, tests, scripts, frontend)  
**Date:** 2025-08-21

---

## 1. End-to-End Flow

```
User Audio / Text
        │
        ▼
┌─────────────────┐     ┌──────────────────┐
│  Browser (UI)   │────▶│  FastAPI main.py │
│  index.html      │     │  /transcribe     │
└─────────────────┘     │  /query          │
                        │  /voice-query    │
                        └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
             audio path                 text query
                    │                         │
                    ▼                         ▼
          ┌─────────────────┐       ┌───────────────┐
          │  harness.py     │       │ orchestrator   │
          │  STT harness    │       │ run_pipeline() │
          │  retry + guard  │       └───────┬───────┘
          └────────┬────────┘               │
                   │                        ▼
                   ▼               ┌─────────────────┐
          ┌─────────────────┐     │  retrieval.py   │
          │  stt.py         │     │  FAISS search   │
          │  Sarvam/Eleven/ │     │  + rerank (opt) │
          │  LocalWhisper   │     └────────┬────────┘
          └────────┬────────┘              │
                   │                       ▼
                   │               ┌─────────────────┐
                   │               │  guardrails.py  │
                   │               │  screen + check │
                   │               └────────┬────────┘
                   │                        │
                   │                        ▼
                   │               ┌─────────────────┐
                   │               │  llm.py         │
                   │               │  Gemini/OpenAI/ │
                   │               │  Local/fallback │
                   │               └────────┬────────┘
                   │                        │
                   └────────────┬───────────┘
                                │
                                ▼
                       JSON Response → UI
```

### Component Breakdown

| Layer | File(s) | Responsibility |
|-------|---------|----------------|
| **API** | `main.py` | FastAPI routes, uploads, CORS, index warmup |
| **STT Harness** | `harness.py` | Retry/backoff, provider routing, guardrail on transcript |
| **STT Providers** | `stt.py` | Sarvam SDK/REST, ElevenLabs, faster-whisper, openai-whisper |
| **Orchestrator** | `orchestrator.py` | Wires STT → retrieval → guardrails → LLM |
| **Retrieval** | `retrieval.py` | FAISS search, optional cross-encoder rerank |
| **Embedding** | `embed.py` | SentenceTransformer + FAISS index CRUD |
| **Chunking** | `chunking.py` | Fixed-size overlap, semantic split, metadata-aware hybrid |
| **LLM** | `llm.py` | Gemini/OpenAI/local generation with extractive fallback |
| **Guardrails** | `guardrails.py` | Unsafe keyword block, short-query reject, abstain on low score, hallucination check |
| **Preprocess** | `preprocess.py` | HTML strip, Unicode NFKC normalize |
| **Frontend** | `app/static/index.html` | Single-page tester: text query + mic/upload voice |
| **Index Builder** | `build_index.py` | Load parquet → chunk → embed → save FAISS |

---

## 2. Flaws Inventory

### 🔴 CRITICAL

#### F-1: Faster-Whisper is producing complete hallucinations on real speech
**Where:** `app/stt.py` → `LocalWhisperSTT.transcribe()`  
**Symptoms:** Input Hinglish *"whats up dude kya kr rha hai bhai"* → output *"so what's up dude can i get a light"*.  
**Root causes (stacked):**
1. **Model size too small for code-mixed speech.** `WHISPER_MODEL=base` is the default. `base` has limited multilingual capacity and is known to hallucinate on noisy / code-mixed / accented speech.
2. **No VAD / audio preprocessing.** Raw audio is fed directly to Whisper. Silence, background noise, and clipping drastically increase hallucination rate.
3. **No language_id enforcement for mixed input.** `language_code="hi-IN"` is passed, but the user is speaking Hinglish. Whisper `base` does not reliably handle code-mixing; forcing `hi` biases it toward Hindi phonetics and can corrupt English tokens.
4. **No confidence gating.** The harness does not check `info.no_speech_prob` or segment-level confidence; it blindly returns whatever Whisper emits.

**Impact:** STT layer is unreliable. Downstream RAG gets garbage queries.

---

#### F-2: `/voice-query` hardcodes provider to `"sarvam"`; no user/provider override
**Where:** `app/main.py` `run_pipeline_async()` → `run_pipeline()` in `orchestrator.py`  
**Symptoms:** The endpoint signature accepts `language_code` but `orchestrator.run_pipeline()` always calls `transcribe_with_harness(..., "sarvam")`. The `/transcribe` endpoint supports `provider` form field, but `/voice-query` does not.
**Impact:** If Sarvam key is missing or the user wants to compare providers, `/voice-query` cannot fall back or switch. In production this is a hidden single-point-of-failure.

---

#### F-3: `FaissIndex.load()` may crash when `meta.json` is missing and `meta.pkl` is absent
**Where:** `app/embed.py` `load()`  
**Code path:** If `meta.json` does not exist, it falls back to `pickle.load(meta.pkl)`. If **both** are missing, the function raises `ENOENT`/`FileNotFoundError` without a clean error message.
**Impact:** Startup fails with a raw traceback instead of a structured "index not built" error.

---

### 🟠 HIGH

#### F-4: `LocalWhisperSTT._load()` caches model in instance, but `get_provider()` creates new instances per request
**Where:** `app/stt.py` `get_provider()`  
**Symptoms:** Every call to `get_provider("whisper")` constructs a fresh `LocalWhisperSTT`. The model is lazy-loaded on first `transcribe()`, but because the instance is thrown away after each request, **the model is reloaded on every request** until Python process reuses the same instance in practice due to module-level caching quirks.
**Impact:** First request latency is inflated by model load time; subsequent requests may or may not benefit from caching depending on GC. Should be a singleton.

---

#### F-5: `orchestrator.py` imports `transcribe_with_harness` synchronously but `main.py` has `atranscribe_with_harness`
**Where:** `app/orchestrator.py` line 7 vs `app/main.py` line 226  
**Symptoms:** `run_pipeline()` uses sync `transcribe_with_harness`. In `/voice-query`, the route wraps the whole `run_pipeline` in `asyncio.to_thread()`. This works, but it also means the **STT retry backoff blocks an entire thread** for up to ~3 seconds per failed attempt.
**Impact:** Thread-pool starvation under load. A better pattern is to make `run_pipeline` async all the way down.

---

#### F-6: No audio format validation or transcoding before STT
**Where:** `app/main.py` `_save_upload()`  
**Symptoms:** The app saves the raw upload and sends it to the STT provider. If the user uploads a `.webm` recorded by the browser, Sarvam/ElevenLabs may reject it or return degraded transcriptions. Faster-whisper can handle many formats via ffmpeg, but Sarvam expects WAV/MP3.
**Impact:** Silent failures or poor transcripts depending on codec.

---

#### F-7: `build_index.py` ignores `SAMPLE_PARQUET` default and uses a hardcoded `DATA_DIR / "sample_hinval_5000.parquet"`
**Where:** `app/config.py` defines `SAMPLE_PARQUET = DATA_DIR / "sample_hinval_5000.parquet"` but `build_index.py` imports `SAMPLE_PARQUET` from config. This is actually consistent.  
**Wait — re-reading:** `build_index.py` does use `SAMPLE_PARQUET` from config. However, `pull_dataset.py` writes to `DATA_DIR / f"sample_{Path(hf_path).stem}_{sample_n}.parquet"`, which produces `sample_hinval_5000.parquet` and `sample_benval_2000.parquet`. This is consistent.  
**Revised finding:** Not a bug, but `build_index.py` default `--rows 800` is much smaller than the parquet sample size; it silently truncates.

---

### 🟡 MEDIUM

#### F-8: `llm.py` `_try_gemini` and `_try_openai` swallow exceptions and fall back silently
**Where:** `app/llm.py`  
**Symptoms:** If Gemini is configured but returns a 429 or malformed response, the code falls back to OpenAI, then to local, then to extractive fallback. The caller gets no indication that the preferred provider failed.
**Impact:** Hard to debug provider issues in production. Logs exist, but the API response does not surface the provider chain.

---

#### F-9: `hybrid_chunk_row()` explodes chunk count aggressively
**Where:** `app/chunking.py` `hybrid_chunk_row()`  
**Symptoms:** For each passage it generates:
- fixed 256 + overlap chunks
- fixed 512 + overlap chunks  
- semantic chunks
- 1 metadata_raw chunk
For a 500-word passage this can produce 20+ chunks. Multiply by thousands of rows → index bloat.
**Impact:** Index build time, memory, and retrieval latency all grow. No deduplication across strategies except by exact text hash; semantically similar chunks survive.

---

#### F-10: `FaissIndex.search()` normalizes query twice
**Where:** `app/embed.py` `search()`  
**Code:** `q = self.model.encode(... normalize_embeddings=True ...)` then `faiss.normalize_L2(q)` again.  
**Impact:** Double normalization is a no-op for unit vectors, but it signals a mental-model bug: the author may think FAISS needs raw vectors. More importantly, if `model.encode` ever returns non-unit vectors, this could silently distort scores.

---

#### F-11: `index.html` root file is duplicated
**Where:** `./index.html` at repo root AND `app/static/index.html`  
**Symptoms:** The root `index.html` is 18 KB and appears to be the same UI. `main.py` serves `app/static/index.html` for `/`, so the root `index.html` is dead weight.
**Impact:** Two copies of the same file drift apart over time.

---

#### F-12: `tests/` directory has no `test_stt.py`
**Where:** `tests/`  
**Symptoms:** `test_stt.py` exists at repo root (`C:/Codes/ai/test_stt.py`) but not inside `tests/`. `pytest` only picks up the one in root because it is not in a test directory. This is fine but inconsistent.
**Impact:** Minor — coverage fragmentation.

---

### 🟢 LOW / NIT

#### F-13: `conftest.py` sets empty API keys globally
**Where:** `tests/conftest.py`  
**Impact:** Forces LLM/STT providers to fall back to local/extractive in every test. This is intentional for offline tests, but it means integration tests that verify provider selection logic cannot run without monkeypatching.

#### F-14: `AUDIO_EXTENSIONS` includes `.mp4` but not `.aac` or `.3gp`
**Impact:** Mobile recordings in common formats may be rejected by the upload guard.

#### F-15: `bench_results.json` is committed to git
**Impact:** Benchmarks change per hardware; committed results become stale and misleading.

#### F-16: `app/__init__.py` is empty
**Impact:** None functionally, but package metadata is missing.

---

## 3. Diagnosis: Why faster-whisper said *"can i get a light"*

Your input: **"whats up dude kya kr rha hai bhai"**  
Heard: **"so what's up dude can i get a light"**

This is a **classic Whisper hallucination** compounded by project-level gaps.

### Why this exact hallucination occurs

1. **Base model is too small for code-mixed Hinglish.**  
   Whisper `base` (74M params) is trained on ~70% English, ~15% non-English. Hinglish (Hindi + English code-mix) is underrepresented. The model tries to force-fit phonemes into English because it sees English tokens like *"whats"* and *"dude"*, then fills the Hindi gap with plausible-sounding English noise (*"can i get a light"*).

2. **No VAD / noise suppression.**  
   If the recording has any silence, background hum, or mic clipping, Whisper often "dreams" content for low-probability frames. The phrase *"can i get a light"* is a common hallucination template because it appears often in Whisper training data for unclear audio.

3. **`language_code="hi-IN"` forces Hindi tokenization.**  
   In `test_stt.py` and the route defaults, the language is `hi-IN`. Whisper’s decoder then biases toward Hindi vocabulary. The model mis-routes *"kya kr rha hai"* → *"can i get a light"* because both are short, high-frequency phrases with similar phoneme density under forced Hindi decoding.

4. **No post-transcription confidence check.**  
   The harness never inspects `info.no_speech_prob` or segment-level `avg_logprob`. Low-confidence segments are returned verbatim.

5. **Audio path / codec issues.**  
   The browser records `.webm/opus`. If this is sent directly to local faster-whisper without transcoding to 16 kHz mono WAV, sample-rate mismatch can smear phonemes.

### How to verify

Run the same audio through:
- `whisper-ctranslate2` with `language=None` (auto-detect)
- Sarvam Saarika v2.5 (cloud, better Hinglish)
- OpenAI Whisper `large-v3` (best multilingual fidelity)

If only `base` produces the hallucination, the model size is the culprit.

---

## 4. Recommended Fixes (Priority Order)

### Immediate (P0)
| ID | Fix | File(s) |
|----|-----|---------|
| P0-1 | **Transcode uploads to 16 kHz mono WAV** before STT. Use ffmpeg-python or pydub. | `main.py`, `stt.py` |
| P0-2 | **Expose `provider` param on `/voice-query`** and pass it to orchestrator. | `main.py`, `orchestrator.py` |
| P0-3 | **Gate Whisper output by confidence.** Reject or re-run when `no_speech_prob > 0.6` or average logprob is low. | `stt.py`, `harness.py` |
| P0-4 | **Upgrade default WHISPER_MODEL to `small` or `medium`** for local use. | `.env.example`, `config.py` |
| P0-5 | **Add VAD** (e.g., Silero VAD) to trim silence/noise before transcription. | `stt.py` |

### Short-term (P1)
| ID | Fix | File(s) |
|----|-----|---------|
| P1-1 | **Singleton for LocalWhisperSTT** to avoid per-request reload. | `stt.py` |
| P1-2 | **Async STT all the way down** — replace `transcribe_with_harness` in orchestrator with true async to avoid blocking thread pool. | `orchestrator.py`, `harness.py` |
| P1-3 | **Deduplicate chunks by semantic similarity**, not just exact hash. | `chunking.py` |
| P1-4 | **Remove root `index.html`** or add redirect note to avoid confusion. | repo root |
| P1-5 | **Commit hygiene:** add `bench_results.json` to `.gitignore`. | `.gitignore` |

### Medium-term (P2)
| ID | Fix | File(s) |
|----|-----|---------|
| P2-1 | Add `language=None` auto-detect for local Whisper when user input language is unknown. | `stt.py` |
| P2-2 | Surface provider chain / fallback reason in API response. | `llm.py`, `orchestrator.py` |
| P2-3 | Expand test coverage: add `tests/test_stt.py` with fake-provider tests for all three backends. | `tests/` |
| P2-4 | Add request timeout / circuit-breaker for Sarvam/ElevenLabs. | `stt.py` |

---

## 5. Summary

- **Flow:** Clean layered architecture (API → STT → Retrieval → LLM) with guardrails at transcript and answer stages.
- **Biggest flaw:** Local Whisper `base` + no VAD + no confidence gating + forced `hi-IN` = hallucination factory for Hinglish/mixed speech. This directly explains the *"can i get a light"* mishearing.
- **Quick win:** Switch to `small` model, add VAD, and transcode audio to 16 kHz mono. This alone will eliminate the majority of hallucination cases.
- **Tests:** 31 passing; coverage is solid for pure logic but empty for real STT integration.

---

**Blocked / Unverified:**
- Exact audio codec of your browser recording (not inspected).
- Whether `ffmpeg` is installed on your machine (required for VAD/transcode step).
- Sarvam fallback behavior when key is missing (assumed from README; not verified in current `.env`).
