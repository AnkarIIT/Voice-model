# RAGinGOA — HH Goa 2026 Submission Plan

## Task Requirements Checklist

### 1. Speech-to-text ✅
- **Provider:** Sarvam Saarika v2.5 (primary)
- **Fallback:** faster-whisper (local)
- **Languages:** hi-IN, en-IN, bn-IN
- **Status:** Working, tested with real audio

### 2. Chunking ✅
- **Strategies:**
  - Fixed-size overlap (256 tokens, 20% overlap)
  - Fixed-size overlap (512 tokens, 15% overlap)
  - Semantic splitting (embedding-based similarity threshold 0.68)
  - Metadata-aware chunking (query_type, target_lang, doc_id)
- **Deduplication:** SHA1 hash-based exact dedup across strategies
- **Status:** Implemented and tested

### 3. Latency Target ⚠️
- **Retrieval engine:** P50=25ms, P70=27ms, P100=203ms ✅ (meets <200ms)
- **LLM generation:** P50=750ms (gemini-3.5-flash-lite)
- **Total pipeline:** ~775ms
- **STT (Sarvam):** ~2-5s
- **Honest assessment:** The retrieval engine meets the 200ms target. Full pipeline with cloud LLM exceeds it. For <200ms end-to-end, a local GPU-accelerated model would be required.

### 4. Latency Analytics ✅
- **Benchmark script:** `bench_latency.py`
- **Metrics measured:** retrieval_llm_ms, total_ms, voice_ms
- **Test queries:** 50 real queries from MSMARCO-XI
- **Report:** `bench_results.json`

### 5. Harness ✅
- **File:** `app/harness.py`
- **Features:**
  - Structured I/O: `{text, language, confidence, latency_ms, total_ms, provider, guardrail}`
  - Retry: 3 attempts with exponential backoff + jitter
  - Fail-fast for non-retryable errors (missing key, auth)
  - Provider routing: Sarvam → ElevenLabs → Local Whisper
  - Guardrail on transcript: empty/unsafe check

### 6. Guardrails ✅
- **Transcript guardrail:** `screen_text()` - empty/unsafe keyword check
- **Query guardrail:** `check_guardrails()` - short query, no context, low similarity
- **Hallucination check:** `hallucination_check()` - lexical overlap grounding
- **Unsafe patterns:** Hindi + English keyword regex
- **Abstain threshold:** 0.35 similarity
- **Grounding threshold:** 0.4 lexical hit rate

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Sarvam STT | ✅ Working | ~2-5s latency |
| FAISS index | ✅ 49,912 vectors | 384-d, hybrid chunking |
| Retrieval | ✅ P50=25ms | Meets 200ms target |
| LLM (Gemini) | ✅ Working | 3.5-flash-lite, ~750ms |
| Guardrails | ✅ Active | Keyword, similarity, grounding |
| Harness | ✅ 3 retries | Backoff + jitter |
| Frontend | ✅ Landing page | Professional UI |
| Tests | ✅ 31/31 | Passing |
| Benchmark | ✅ Ready | 50 queries measured |

---

## Submission Artifacts Needed

### 1. GitHub Repository
- Current repo: `C:/Codes/ai`
- Ensure `.env` is in `.gitignore` ✅
- Push to GitHub

### 2. Live Working Link
- Deploy to: Render / Railway / Vercel
- Or: ngrok tunnel for demo

### 3. Video 1: Team/Process (90 seconds)
- Show code structure
- Show chunking strategies
- Show benchmark running
- Show tests passing
- Show guardrails in action

### 4. Video 2: Demo (end-to-end)
- Record browser session
- Upload audio or use mic
- Show transcription
- Show retrieval
- Show answer generation
- Show guardrail blocking unsafe query

### 5. Social Media Posts
- Instagram: 1 public post per team member
- X/Twitter: 1 post per team member
- Hashtag: `#RAGInGoa` (mandatory on every post)
- Content: Demo clip + project description

### 6. Submission Form
- URL: https://forms.gle/MNvCjcv23Hn2Eeu58
- Fill: GitHub repo, live link, video links

---

## Next Steps (Priority Order)

### P0 - Before Submission
1. **Verify Gemini latency** with gemini-3.5-flash-lite across 50 queries
2. **Generate bench_results.json** with final numbers
3. **Record Video 1** (process) - 90 seconds
4. **Record Video 2** (demo) - end-to-end working
5. **Push to GitHub** with clean commit history
6. **Deploy live** (Render/Railway/ngrok)
7. **Fill submission form**

### P1 - Code Polish
1. Fix `/voice-query` provider override
2. Add VAD + audio transcoding (nice-to-have)
3. Migrate `meta.pkl` to `meta.json`
4. Add TTS endpoint for voice output

### P2 - Post-Submission
1. Multi-turn conversation
2. Streaming responses
3. Auth / user management
4. Docker deployment

---

## Known Limitations (Be Honest in Submission)

1. **200ms target:** Retrieval engine meets it (P50=25ms). Full pipeline with cloud LLM is ~775ms. This is acceptable for a cloud-based RAG system; local GPU would be needed for <200ms end-to-end.

2. **Whisper fallback:** `base` model hallucinates on Hinglish. Primary path uses Sarvam which works well.

3. **No TTS:** System returns text only. Voice output not implemented yet.

4. **Single-turn only:** No conversation memory/history.

---

## Files to Submit

```
GitHub repo:
├── app/
│   ├── main.py
│   ├── stt.py
│   ├── harness.py
│   ├── orchestrator.py
│   ├── retrieval.py
│   ├── embed.py
│   ├── chunking.py
│   ├── preprocess.py
│   ├── guardrails.py
│   ├── llm.py
│   └── static/index.html
├── tests/
├── bench_latency.py
├── build_index.py
├── requirements.txt
├── README.md
├── .env.example
└── bench_results.json
```

---

## Timeline

| Task | Owner | Deadline |
|------|-------|----------|
| Final benchmark run | | Aug 22, 2026 |
| Video 1 recording | | Aug 22, 2026 |
| Video 2 recording | | Aug 22, 2026 |
| GitHub push | | Aug 22, 2026 |
| Live deployment | | Aug 22, 2026 |
| Form submission | | Aug 22, 2026 11:59 PM |
| Social media posts | All members | Aug 22, 2026 |
