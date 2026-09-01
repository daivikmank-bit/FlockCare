from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import MAX_FILE_MB
from .conversion import ConversionError
from .inference import AudioTooShortError, InsufficientSignalError, analyze_audio_bytes
from .schemas import AnalyzeResponse, HealthResponse

app = FastAPI(
    title="FlockCare API",
    version="0.1.0",
    description="Acoustic respiratory health screening for backyard and smallholder poultry flocks.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to specific origin in production deployment
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {
    "audio/webm",
    "audio/ogg",
    "audio/mp4",
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/aac",
    "audio/flac",
    "application/octet-stream",  # often sent by generic binary form uploads
}

ALLOWED_EXTENSIONS = {".wav", ".webm", ".ogg", ".mp4", ".mp3", ".m4a", ".flac", ".aac"}


def _is_allowed_format(content_type: str, filename: str) -> bool:
    """Validates whether the upload is an allowed audio format by MIME type or extension."""
    base_ct = (content_type or "").split(";")[0].strip().lower()
    if base_ct in ALLOWED_CONTENT_TYPES:
        return True

    # Fallback to extension check
    if filename:
        for ext in ALLOWED_EXTENSIONS:
            if filename.lower().endswith(ext):
                return True

    return False


@app.get("/")
def root():
    """Root welcome endpoint with API discovery."""
    return {
        "name": "FlockCare API",
        "version": "0.1.0",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness and readiness health check endpoint."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    """
    Receives an audio file (WebM/Ogg/MP4/WAV/MP3), normalizes and segments it into
    5-second spectrogram windows, runs CNN inference with OOD gating, and returns
    a clinical screening risk classification.
    """
    if not _is_allowed_format(file.content_type, file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: '{file.content_type}'. Supported formats: WebM, Ogg, MP4, WAV, MP3.",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    if len(raw_bytes) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large (max {MAX_FILE_MB}MB).",
        )

    try:
        import asyncio
        return await asyncio.to_thread(analyze_audio_bytes, raw_bytes)
    except ConversionError as e:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")
    except (AudioTooShortError, InsufficientSignalError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Internal analysis error: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed. Please try recording again.")
