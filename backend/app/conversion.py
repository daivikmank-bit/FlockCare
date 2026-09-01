"""Audio format conversion utilities using soundfile and imageio-ffmpeg."""

import io
import subprocess
import soundfile as sf

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = "ffmpeg"


class ConversionError(Exception):
    """Raised when an uploaded audio file cannot be decoded."""
    pass


def _decode_with_ffmpeg(raw_bytes: bytes) -> bytes:
    """
    Decodes arbitrary audio streams (WebM, Opus, MP4, AAC, MP3, etc.)
    directly to 22050 Hz 16-bit mono WAV using bundled FFmpeg pipe.
    """
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-nostdin",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "wav",
        "-ac", "1",
        "-ar", "22050",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        wav_bytes, err = proc.communicate(input=raw_bytes, timeout=10)
        if proc.returncode != 0 or len(wav_bytes) == 0:
            err_msg = err.decode("utf-8", errors="ignore").strip()
            raise ConversionError(f"FFmpeg decoding failed: {err_msg or 'invalid audio format'}")
        return wav_bytes
    except subprocess.TimeoutExpired:
        proc.kill()
        raise ConversionError("Audio decoding timed out.")
    except Exception as e:
        if isinstance(e, ConversionError):
            raise
        raise ConversionError(f"Could not decode audio: {e}") from e


def convert_to_wav_bytes(raw_bytes: bytes) -> bytes:
    """
    Takes arbitrary browser-recorded or uploaded audio (webm/ogg/mp4/wav/mp3/flac)
    and returns clean WAV PCM bytes.
    Tries native soundfile decoding first (WAV/OGG/FLAC), then falls back to FFmpeg pipe.
    """
    if not raw_bytes:
        raise ConversionError("Received empty audio payload.")

    # 1. Fast native decoding with soundfile (supports standard WAV, OGG, FLAC)
    try:
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV")
        return buf.getvalue()
    except Exception:
        pass

    # 2. Robust container decoding with FFmpeg
    return _decode_with_ffmpeg(raw_bytes)
