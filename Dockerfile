FROM python:3.11-slim

# ffmpeg is required by the whisper fallback (webm/opus -> 16kHz mono WAV)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.deploy.txt ./
RUN pip install --no-cache-dir -r requirements.deploy.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY app ./app
COPY index_hinval ./index_hinval
COPY bench_results.json ./bench_results.json

ENV INDEX_DIR=/srv/index_hinval \
    WHISPER_DEVICE=cpu \
    RERANK_DEVICE=cpu \
    PYTHONUNBUFFERED=1

EXPOSE 8000
ENV PORT=8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
