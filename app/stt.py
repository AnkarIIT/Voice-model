import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_S = 30


def to_16k_mono(src: Path) -> Path:
    """Transcode audio to 16kHz mono WAV via ffmpeg (best-effort).

    Browser recordings are often webm/opus at 48kHz stereo, which hurts
    whisper accuracy. Returns the original path if ffmpeg is unavailable.
    """
    ff = shutil.which("ffmpeg")
    if not ff:
        return src
    dst = src.with_suffix(".16k.wav")
    try:
        r = subprocess.run(
            [ff, "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
            capture_output=True,
            timeout=60,
        )
        if r.returncode == 0 and dst.exists():
            return dst
        logger.warning("ffmpeg transcode failed (%s); using original file", r.stderr.decode(errors="replace")[:200])
    except Exception as e:
        logger.warning("ffmpeg transcode error (%s); using original file", e)
    return src


class STTError(RuntimeError):
    pass


class STTConfigError(STTError):
    """Non-retryable: missing/invalid credentials or bad request."""


@dataclass
class STTResult:
    text: str
    language: str
    confidence: Optional[float]
    provider: str
    latency_ms: float
    raw: dict


class SarvamSTT:
    URL = "https://api.sarvam.ai/speech-to-text"
    _client = None
    _client_key = None

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise STTConfigError("SARVAM_API_KEY not set")
        self.api_key = api_key
        self.model = model

    def _sdk_client(self):
        cls = type(self)
        if cls._client is None or cls._client_key != self.api_key:
            from sarvamai import SarvamAI

            cls._client = SarvamAI(api_subscription_key=self.api_key)
            cls._client_key = self.api_key
        return cls._client

    def transcribe(self, audio_path: Path, language_code: str = "hi-IN") -> STTResult:
        t0 = time.perf_counter()
        try:
            with open(audio_path, "rb") as f:
                resp = self._sdk_client().speech_to_text.transcribe(
                    file=f, model=self.model, language_code=language_code
                )
        except Exception as sdk_err:
            logger.warning("sarvam SDK failed (%s); trying REST fallback", sdk_err)
            return self._transcribe_rest(audio_path, language_code, t0)
        j = resp.model_dump() if hasattr(resp, "model_dump") else dict(vars(resp))
        text = j.get("transcript") or j.get("text") or ""
        if not text and isinstance(j.get("transcripts"), list) and j["transcripts"]:
            text = j["transcripts"][0].get("transcript", "")
        return STTResult(
            text=text,
            language=language_code,
            confidence=j.get("confidence"),
            provider=f"sarvam:{self.model}",
            latency_ms=(time.perf_counter() - t0) * 1000,
            raw=j,
        )

    def _transcribe_rest(self, audio_path: Path, language_code: str, t0: float) -> STTResult:
        try:
            with open(audio_path, "rb") as f:
                r = requests.post(
                    self.URL,
                    files={"file": (audio_path.name, f, "audio/wav")},
                    data={"model": self.model, "language_code": language_code},
                    headers={"api-subscription-key": self.api_key},
                    timeout=REQUEST_TIMEOUT_S,
                )
        except requests.RequestException as e:
            raise STTError(f"sarvam request failed: {e}") from e
        if r.status_code in (401, 403):
            raise STTConfigError(f"sarvam auth failed (HTTP {r.status_code})")
        try:
            r.raise_for_status()
            j = r.json()
        except (requests.HTTPError, ValueError) as e:
            raise STTError(f"sarvam REST failed: {e}") from e
        return STTResult(
            text=j.get("transcript") or j.get("text") or "",
            language=language_code,
            confidence=j.get("confidence"),
            provider=f"sarvam:{self.model}",
            latency_ms=(time.perf_counter() - t0) * 1000,
            raw=j,
        )


class ElevenLabsSTT:
    URL = "https://api.elevenlabs.io/v1/speech-to-text"

    def __init__(self, api_key: str, model: str = "scribe_v1"):
        if not api_key:
            raise STTConfigError("ELEVENLABS_API_KEY not set")
        self.api_key = api_key
        self.model = model

    def transcribe(self, audio_path: Path, language_code: str = "hi") -> STTResult:
        t0 = time.perf_counter()
        with open(audio_path, "rb") as f:
            try:
                r = requests.post(
                    self.URL,
                    files={"file": f},
                    data={"model_id": self.model},
                    headers={"xi-api-key": self.api_key},
                    timeout=REQUEST_TIMEOUT_S,
                )
            except requests.RequestException as e:
                raise STTError(f"elevenlabs request failed: {e}") from e
        if r.status_code in (401, 403):
            raise STTConfigError(f"elevenlabs auth failed (HTTP {r.status_code})")
        try:
            r.raise_for_status()
            j = r.json()
        except (requests.HTTPError, ValueError) as e:
            raise STTError(f"elevenlabs failed: {e}") from e
        return STTResult(
            text=j.get("text", ""),
            language=j.get("language_code") or language_code,
            confidence=j.get("confidence"),
            provider="elevenlabs",
            latency_ms=(time.perf_counter() - t0) * 1000,
            raw=j,
        )


class LocalWhisperSTT:
    def __init__(self, model_name: str = "base", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self.backend = None

    def _resolve_device(self) -> str:
        if self.device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            logger.warning("CUDA requested but unavailable; whisper falling back to CPU")
            return "cpu"
        return "cpu"

    def _load(self):
        if self._model is not None:
            return
        device = self._resolve_device()
        compute = "float16" if device == "cuda" else "int8"
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_name, device=device, compute_type=compute)
            self.backend = "faster-whisper"
        except Exception as fw_err:
            logger.warning("faster-whisper unavailable (%s); trying openai-whisper", fw_err)
            import whisper

            self._model = whisper.load_model(self.model_name).to(device)
            self.backend = "openai-whisper"
        logger.info("loaded %s (%s) on %s", self.model_name, self.backend, device)

    def _lang(self, language_code: str) -> Optional[str]:
        code = (language_code or "").split("-")[0].strip().lower()
        return code or None

    def transcribe(self, audio_path: Path, language_code: str = "hi") -> STTResult:
        self._load()
        t0 = time.perf_counter()
        processed = to_16k_mono(audio_path)
        try:
            lang = self._lang(language_code)
            if self.backend == "faster-whisper":
                segments, info = self._model.transcribe(
                    str(processed),
                    language=lang,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 300},
                    condition_on_previous_text=False,
                )
                kept, logprobs = [], []
                for s in segments:
                    # drop likely hallucinations: silent audio w/ low model confidence
                    if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                        continue
                    kept.append(s.text)
                    logprobs.append(s.avg_logprob)
                text = " ".join(kept).strip()
                detected = info.language
            else:
                result = self._model.transcribe(str(processed), language=lang)
                segs = [
                    s for s in result.get("segments", [])
                    if not (s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -1.0)
                ]
                text = "".join(s.get("text", "") for s in segs).strip() or result["text"].strip()
                detected = result.get("language", lang)
                logprobs = [s.get("avg_logprob") for s in segs if s.get("avg_logprob") is not None]
        finally:
            if processed != audio_path:
                try:
                    processed.unlink(missing_ok=True)
                except OSError:
                    pass
        confidence = round(sum(logprobs) / len(logprobs), 2) if logprobs else None
        return STTResult(
            text=text,
            language=detected,
            confidence=confidence,
            provider=f"local-{self.backend}",
            latency_ms=(time.perf_counter() - t0) * 1000,
            raw={},
        )

def get_provider(name: str = "sarvam"):
    from .config import (
        ELEVENLABS_API_KEY,
        SARVAM_API_KEY,
        SARVAM_MODEL,
        WHISPER_DEVICE,
        WHISPER_MODEL,
    )
    from .embed import resolve_device

    name = (name or "").lower()
    device = resolve_device(WHISPER_DEVICE)
    if name in ("whisper", "local", "local-whisper"):
        return LocalWhisperSTT(WHISPER_MODEL, device)
    if name == "sarvam":
        if SARVAM_API_KEY:
            return SarvamSTT(SARVAM_API_KEY, SARVAM_MODEL)
        logger.warning("no SARVAM_API_KEY; falling back to local whisper")
    elif name == "elevenlabs":
        if ELEVENLABS_API_KEY:
            return ElevenLabsSTT(ELEVENLABS_API_KEY)
        logger.warning("no ELEVENLABS_API_KEY; falling back to local whisper")
    else:
        raise ValueError(f"unknown STT provider {name!r}; expected sarvam|elevenlabs|whisper")
    return LocalWhisperSTT(WHISPER_MODEL, device)
