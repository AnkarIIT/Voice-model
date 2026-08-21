import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(BASE_DIR / "index_hinval")))
SAMPLE_PARQUET = DATA_DIR / os.getenv("SAMPLE_PARQUET_NAME", "sample_hinval_5000.parquet")

STT_PROVIDER = os.getenv("STT_PROVIDER", "sarvam").strip().lower()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "saarika:v2.5")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda").strip().lower()

EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", WHISPER_DEVICE).strip().lower()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
USE_LOCAL_LLM = _bool("USE_LOCAL_LLM")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "google/flan-t5-base")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_K = int(os.getenv("MAX_K", "20"))
DEFAULT_K = int(os.getenv("DEFAULT_K", "5"))
ABSTAIN_THRESHOLD = float(os.getenv("ABSTAIN_THRESHOLD", "0.35"))
GROUNDING_HIT_RATE = float(os.getenv("GROUNDING_HIT_RATE", "0.4"))

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".oga", ".webm", ".mp4",
}


def setup_logging() -> None:
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        )
    else:
        logging.getLogger().setLevel(level)
