"""Integration and endpoint tests for FastAPI backend (backend/app/main.py)."""

import io
import os
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app.main import app
from ml.preprocessing.generate_sample_data import generate_audio_signal

client = TestClient(app)


@pytest.fixture
def sample_15s_wav_bytes() -> bytes:
    """Generates 15-second healthy chicken WAV bytes (produces 3 windows)."""
    sr = 22050
    y = generate_audio_signal(duration=15.0, sr=sr, signal_type="healthy")
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def sample_short_wav_bytes() -> bytes:
    """Generates 5-second sub-threshold audio bytes (< 15s requirement)."""
    sr = 22050
    y = generate_audio_signal(duration=5.0, sr=sr, signal_type="healthy")
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def sample_silent_wav_bytes() -> bytes:
    """Generates 15-second silent audio bytes below minimum RMS floor."""
    sr = 22050
    y = np.zeros(int(15.0 * sr), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


def test_health_endpoint():
    """Verify GET /health returns 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_valid_wav(sample_15s_wav_bytes):
    """Verify POST /analyze returns 200 and matches AnalyzeResponse schema."""
    files = {"file": ("recording.wav", sample_15s_wav_bytes, "audio/wav")}
    response = client.post("/analyze", files=files)

    assert response.status_code == 200
    data = response.json()

    assert "risk_score" in data and isinstance(data["risk_score"], (int, float))
    assert data["risk_level"] in ("low", "moderate", "high")
    assert "message" in data and len(data["message"]) > 0
    assert "disclaimer" in data
    assert data["windows_analyzed"] >= 3
    assert data["status"] in ("calibrated", "out_of_range")


def test_analyze_multi_window_count(sample_15s_wav_bytes):
    """Verify windows_analyzed equals 3 for a 15-second clip."""
    files = {"file": ("flock_15s.wav", sample_15s_wav_bytes, "audio/wav")}
    response = client.post("/analyze", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["windows_analyzed"] == 3


def test_analyze_audio_too_short(sample_short_wav_bytes):
    """Verify sub-15-second audio returns clean 400 error message."""
    files = {"file": ("short.wav", sample_short_wav_bytes, "audio/wav")}
    response = client.post("/analyze", files=files)

    assert response.status_code == 400
    assert "too short" in response.json()["detail"].lower()


def test_analyze_silent_audio_rejected(sample_silent_wav_bytes):
    """Verify purely silent audio is rejected by energy pre-filter with 400."""
    files = {"file": ("silent.wav", sample_silent_wav_bytes, "audio/wav")}
    response = client.post("/analyze", files=files)

    assert response.status_code == 400
    assert "silent" in response.json()["detail"].lower() or "no flock vocalizations" in response.json()["detail"].lower()


def test_analyze_empty_file():
    """Verify empty audio file returns 400."""
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/analyze", files=files)

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_analyze_corrupt_bytes():
    """Verify corrupt/garbage payload returns 400, never 500."""
    corrupt_data = b"NOT_A_VALID_AUDIO_HEADER_JUST_GARBAGE_BYTES_1234567890"
    files = {"file": ("corrupt.wav", corrupt_data, "audio/wav")}
    response = client.post("/analyze", files=files)

    assert response.status_code == 400
    assert "could not decode" in response.json()["detail"].lower()


def test_analyze_unsupported_format():
    """Verify unsupported MIME type returns 400."""
    files = {"file": ("data.txt", b"plain text content", "text/plain")}
    response = client.post("/analyze", files=files)

    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_browser_mediarecorder_ogg_decoding():
    """Verify pydub/ffmpeg decodes browser-style audio containers (e.g. OGG/FLAC)."""
    # Create Ogg Vorbis or FLAC bytes using soundfile
    sr = 22050
    y = generate_audio_signal(duration=15.0, sr=sr, signal_type="healthy")
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="OGG")
    ogg_bytes = buf.getvalue()

    files = {"file": ("browser_recording.ogg", ogg_bytes, "audio/ogg")}
    response = client.post("/analyze", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["windows_analyzed"] == 3
    assert "risk_score" in data


def test_analyze_sparse_vocalization_accepted():
    """
    Verify a 15-second clip with 10s quiet barn ambience and a brief 5s vocalization burst
    passes the max-window energy pre-filter (does NOT get false-rejected).
    """
    sr = 22050
    # 10s of quiet background hiss (RMS ~ 0.001) + 5s of real vocalization (RMS ~ 0.25)
    quiet = np.random.normal(0, 0.001, int(10.0 * sr)).astype(np.float32)
    vocal = generate_audio_signal(duration=5.0, sr=sr, signal_type="healthy")
    y = np.concatenate([quiet, vocal])

    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    wav_bytes = buf.getvalue()

    files = {"file": ("sparse_vocal.wav", wav_bytes, "audio/wav")}
    response = client.post("/analyze", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["windows_analyzed"] == 3
    assert data["status"] in ("calibrated", "out_of_range")
