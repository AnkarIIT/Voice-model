from pathlib import Path

import pytest

import app.harness as harness
from app.harness import transcribe_with_harness
from app.stt import STTConfigError, STTResult


class FakeProvider:
    def __init__(self, text="hello world", fail_times=0):
        self.text = text
        self.fail_times = fail_times
        self.calls = 0

    def transcribe(self, audio_path, language_code):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient network error")
        return STTResult(
            text=self.text,
            language="hi",
            confidence=0.9,
            provider="fake",
            latency_ms=12.0,
            raw={},
        )


class ConfigErrorProvider:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio_path, language_code):
        self.calls += 1
        raise STTConfigError("invalid api key")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(harness.time, "sleep", lambda s: None)


def run(tmp_path, provider):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    return transcribe_with_harness(audio, "hi-IN", "sarvam")


def test_success(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "get_provider", lambda name: FakeProvider("  नमस्ते दोस्त  "))
    out = run(tmp_path, None)
    assert out["status"] == "ok"
    assert out["text"] == "नमस्ते दोस्त"
    assert out["guardrail"]["action"] == "allow"


def test_empty_transcript_marked_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "get_provider", lambda name: FakeProvider(""))
    out = run(tmp_path, None)
    assert out["status"] == "empty"


def test_transient_failure_retries_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "get_provider", lambda name: FakeProvider(fail_times=2))
    out = run(tmp_path, None)
    assert out["status"] == "ok"
    assert out["attempt"] == 3


def test_permanent_failure_no_retries(tmp_path, monkeypatch):
    p = ConfigErrorProvider()
    monkeypatch.setattr(harness, "get_provider", lambda name: p)
    out = run(tmp_path, None)
    assert out["status"] == "error"
    assert p.calls == 1
    assert "invalid api key" in out["error"]


def test_all_attempts_exhausted(tmp_path, monkeypatch):
    class AlwaysFail(FakeProvider):
        def transcribe(self, a, l):
            self.calls += 1
            raise RuntimeError("down")

    p = AlwaysFail()
    monkeypatch.setattr(harness, "get_provider", lambda name: p)
    out = run(tmp_path, None)
    assert out["status"] == "error"
    assert p.calls == harness.MAX_RETRIES
