"""Audio format conversion utilities using pydub and ffmpeg."""

import io
import soundfile as sf
from pydub import AudioSegment


class ConversionError(Exception):
    """Raised when an uploaded audio file cannot be decoded."""
    pass


def convert_to_wav_bytes(raw_bytes: bytes) -> bytes:
    """
    Takes arbitrary browser-recorded or uploaded audio (webm/ogg/mp4/wav/mp3/flac)
    and returns clean WAV PCM bytes.
    Tries native soundfile decoding first (WAV/OGG/FLAC), then falls back to pydub/ffmpeg
    for container formats (WebM/MP4/MP3).
    """
    if not raw_bytes:
        raise ConversionError("Received empty audio payload.")

    # 1. Fast native decoding with soundfile (supports WAV, OGG, FLAC)
    try:
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV")
        return buf.getvalue()
    except Exception:
        pass

    # 2. Fallback to pydub (requires ffmpeg for WebM/MP4/MP3)
    try:
        audio = AudioSegment.from_file(io.BytesIO(raw_bytes))
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        return buf.getvalue()
    except Exception as e:
        raise ConversionError(f"Could not decode audio: {e}") from e
