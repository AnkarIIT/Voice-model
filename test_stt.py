import math
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import polars as pl

from app.config import SAMPLE_PARQUET, setup_logging


def synth_wav(text: str, out: Path, duration=1.2, freq=440):
    sr = 16000
    n = int(sr * duration)
    frames = b"".join(
        struct.pack("<h", int(3000 * math.sin(2 * math.pi * freq * i / sr)))
        for i in range(n)
    )
    with wave.open(str(out), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames)


def make_tts_wav(text: str, out: Path, lang="hi") -> bool:
    try:
        from gtts import gTTS

        gTTS(text=text, lang=lang, slow=False).save(str(out))
        print(f"gTTS ok -> {out} {out.stat().st_size} bytes")
        return True
    except Exception as e:
        print(f"gTTS fail {e}, using synth")
        synth_wav(text, out)
        return False


if __name__ == "__main__":
    setup_logging()
    df = pl.read_parquet(SAMPLE_PARQUET)
    row = df.row(0, named=True)
    query_en = row["Eng_Query"]
    query_hi = row["query"]
    print(f"Sample EN: {query_en}")
    print(f"Sample HI: {query_hi.encode('ascii', 'ignore').decode()[:120]}")
    q = query_en[:200] if len(query_en) > 5 else "what is the capital of India"
    tmp = Path(tempfile.gettempdir()) / "test_query.wav"
    make_tts_wav(q, tmp, lang="en")
    print(
        f"Audio: {tmp} exists={tmp.exists()} "
        f"size={tmp.stat().st_size if tmp.exists() else 0}"
    )
    from app.harness import transcribe_with_harness

    print("\n--- transcribing via harness (auto-fallback to local whisper if no SARVAM key) ---")
    res = transcribe_with_harness(tmp, language_code="en-IN", provider="sarvam")
    print(res)
    print(f"\nTEXT: {res['text']}")
    print(
        f"Provider used: {res['provider_used']} "
        f"latency={res['latency_ms']}ms total={res['total_ms']}ms"
    )
    print(f"Guardrail: {res['guardrail']}")
    print("\n--- direct local whisper test ---")
    from app.stt import LocalWhisperSTT
    from app.embed import resolve_device

    w = LocalWhisperSTT("tiny", device="cpu")
    r2 = w.transcribe(tmp, "en")
    print(f"local tiny cpu: '{r2.text}' {r2.latency_ms:.1f}ms")
