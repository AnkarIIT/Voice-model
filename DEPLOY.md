# Deploying RAGinGOA

Vercel is **not** suitable for this app (faiss + torch + a 76MB index need a
long-running container with real memory). Use Render (or Railway/Fly) with the
included Docker setup instead.

## What's in the deploy kit

| File | Purpose |
|---|---|
| `Dockerfile` | python:3.11-slim + ffmpeg, CPU torch, copies `app/` + `index_hinval/` |
| `requirements.deploy.txt` | Runtime-only deps (no polars/pyarrow/pytest) |
| `.dockerignore` | Keeps .env/tests/data out of the image |
| `render.yaml` | One-click Render blueprint |

## Render (recommended)

1. Push this repo to GitHub. `index_hinval/faiss.index` is 76MB — under
   GitHub's 100MB hard limit, so committing it works.
2. On render.com: **New → Blueprint**, pick the repo — it reads `render.yaml`.
3. Fill in the secret env vars when prompted:
   - `SARVAM_API_KEY`
   - `GEMINI_API_KEY`
4. Deploy. Health check hits `/health`; first boot loads the FAISS index +
   embedder (~60-90s), so give the start period time.

### Local Docker test first

```bash
docker build -t ramingoa .
docker run --rm -p 8000:8000 --env-file .env ramingoa
curl http://localhost:8000/health
```

## Env vars reference

| Var | Default | Notes |
|---|---|---|
| `STT_PROVIDER` | sarvam | sarvam / elevenlabs / whisper |
| `LLM_MODEL` | gemini-3.5-flash-lite | any Gemini model id |
| `WHISPER_MODEL` | small | fallback STT quality |
| `TTS_MODEL` | bulbul:v3 | Sarvam TTS model |
| `GROUNDING_EMBED_SIM` | 0.5 | cross-script grounding threshold |
| `INDEX_DIR` | ./index_hinval | set to `/srv/index_hinval` in Docker |
