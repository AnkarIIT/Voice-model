import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import (
    SARVAM_API_KEY,
    SARVAM_MODEL,
    setup_logging,
)

LANG = os.getenv("SARVAM_LANGUAGE", "hi-IN")


def transcribe_with_sarvam(audio_file_path: str, api_key: str = SARVAM_API_KEY):
    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=api_key)
    with open(audio_file_path, "rb") as f:
        response = client.speech_to_text.transcribe(file=f, model=SARVAM_MODEL, language_code=LANG)
    j = response.model_dump() if hasattr(response, "model_dump") else dict(vars(response))
    text = j.get("transcript") or j.get("text") or ""
    print(f"User Query: {text}")
    return response


def _make_test_audio(wav: Path) -> str:
    q = "what is a corporation"
    try:
        from gtts import gTTS

        gTTS(q, lang="en").save(str(wav))
    except Exception as e:
        print(f"gTTS failed ({e}); using placeholder silence")
        wav.write_bytes(b"")
    return str(wav)


if __name__ == "__main__":
    setup_logging()
    tmp = Path(__file__).parent / ".tmp_standalone_test.wav"
    audio = _make_test_audio(tmp)
    if not SARVAM_API_KEY:
        print("Set SARVAM_API_KEY in .env to test the Sarvam SDK directly.")
        print("Falling back to harness (local whisper):")
        from app.harness import transcribe_with_harness

        res = transcribe_with_harness(Path(audio), language_code="en-IN", provider="sarvam")
        print(res)
    else:
        transcribe_with_sarvam(audio)
    tmp.unlink(missing_ok=True)
