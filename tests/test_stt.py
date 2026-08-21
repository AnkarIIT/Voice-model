import struct
import wave
from pathlib import Path

from app.stt import to_16k_mono


def _write_wav(path: Path, rate=48000, channels=2, seconds=0.2):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"\x00\x00" * channels * int(rate * seconds)
        w.writeframes(frames)


def test_to_16k_mono_transcodes(tmp_path):
    src = tmp_path / "in.webm"
    # ffmpeg handles a real wav regardless of the extension name used here
    _write_wav(tmp_path / "in.wav")
    src = tmp_path / "in.wav"
    out = to_16k_mono(src)
    assert out.exists()
    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1


def test_to_16k_mono_falls_back_gracefully(tmp_path, monkeypatch):
    monkeypatch.setattr("app.stt.shutil.which", lambda _: None)
    src = tmp_path / "in.wav"
    _write_wav(src)
    assert to_16k_mono(src) == src
