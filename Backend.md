# FlockCare — Part 5: Backend (FastAPI)

*Standalone build guide. Assumes Part 4's output exists: `ml/saved_models/flockcare_cnn.h5`, taking `(128, 216, 1)` mel-spectrograms per 5-second window and outputting softmax over `[healthy, elevated_respiratory]`.*

**Goal of this part:** an API that takes a farmer's raw ~30s phone recording (whatever format the browser actually sends) and returns a risk result, using the exact same preprocessing the model was trained on.

---

## 5.0 Two things that will bite you if you skip them

1. **Browser audio isn't a clean `.wav`.** `MediaRecorder` in the browser (Part 6 will use this) typically produces `webm`/opus or `mp4`, not `wav`. Plain `librosa.load` can *sometimes* decode these if the system has the right codecs, but that's exactly the kind of thing that works on your dev machine (which has ffmpeg installed for some other reason) and silently breaks in a bare Docker container. Convert explicitly — 5.3 below handles it.
2. **Training/serving skew.** The chunking and spectrogram functions here must be *byte-for-byte identical* to Part 3's. They're restated in full below so this file stands alone, but in your actual repo, import one shared `ml/preprocessing/audio_utils.py` from both the training scripts and this backend — don't maintain two copies. A silent drift between the two (different `hop_length`, different normalization) is a very hard bug to notice, because the API won't error, it'll just quietly give worse predictions.

---

## 5.1 Structure
```
flockcare/
├── ml/
│   ├── preprocessing/
│   │   └── audio_utils.py       ← from Part 3, imported here — not copy-pasted
│   └── saved_models/
│       └── flockcare_cnn.h5     ← from Part 4
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── inference.py
│   │   ├── conversion.py
│   │   ├── schemas.py
│   │   └── config.py
│   ├── requirements.txt
│   └── Dockerfile
```
Run everything from the repo root so `ml` is importable as a package (`PYTHONPATH` handling is in the Dockerfile in 5.10 and the local-run command in 5.9).

## 5.2 Environment setup
```bash
pip install fastapi "uvicorn[standard]" python-multipart tensorflow librosa soundfile pydub numpy
```
`pydub` also needs **ffmpeg installed on the system** (not a pip package) — `apt-get install ffmpeg` on Linux/Docker, `brew install ffmpeg` on macOS for local testing.

## 5.3 Config
```python
# backend/app/config.py
import os

MODEL_PATH = os.getenv("FLOCKCARE_MODEL_PATH", "ml/saved_models/flockcare_cnn.h5")
MAX_FILE_MB = int(os.getenv("FLOCKCARE_MAX_FILE_MB", "15"))
MIN_AUDIO_SECONDS = int(os.getenv("FLOCKCARE_MIN_AUDIO_SECONDS", "5"))
```

## 5.4 Response schema
```python
# backend/app/schemas.py
from pydantic import BaseModel

class AnalyzeResponse(BaseModel):
    risk_score: float
    risk_level: str          # "low" | "moderate" | "high"
    message: str
    disclaimer: str
    windows_analyzed: int    # lets you sanity-check chunking is behaving (30s clip -> ~6)
```

## 5.5 Audio conversion (the ffmpeg step)
```python
# backend/app/conversion.py
import io
from pydub import AudioSegment

class ConversionError(Exception):
    pass

def convert_to_wav_bytes(raw_bytes: bytes) -> bytes:
    """Takes whatever the browser sent (webm/ogg/mp4/wav) and returns clean WAV bytes."""
    try:
        audio = AudioSegment.from_file(io.BytesIO(raw_bytes))
    except Exception as e:
        raise ConversionError(f"Could not decode audio: {e}") from e
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    return buf.getvalue()
```

## 5.6 Preprocessing (restated from Part 3 — keep this identical to that file)
```python
# ml/preprocessing/audio_utils.py
import librosa
import numpy as np

TARGET_SR = 22050
WINDOW_SEC = 5
N_MELS = 128
HOP_LENGTH = 512
TARGET_FRAMES = int(np.ceil(WINDOW_SEC * TARGET_SR / HOP_LENGTH))  # ~216

def chunk_audio(y, sr=TARGET_SR, window_sec=WINDOW_SEC):
    window = int(window_sec * sr)
    if len(y) <= window:
        return [np.pad(y, (0, max(0, window - len(y))))]
    chunks = [y[i:i + window] for i in range(0, len(y) - window + 1, window)]
    remainder = y[len(chunks) * window:]
    if len(remainder) > window * 0.3:
        chunks.append(np.pad(remainder, (0, window - len(remainder))))
    return chunks

def to_mel_spectrogram(y_chunk, sr=TARGET_SR, n_mels=N_MELS, hop_length=HOP_LENGTH):
    mel = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)

def fix_length(mel, target_frames=TARGET_FRAMES):
    if mel.shape[1] < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])))
    else:
        mel = mel[:, :target_frames]
    return mel
```

## 5.7 Inference — the full request path
```python
# backend/app/inference.py
import io
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf

from ml.preprocessing.audio_utils import chunk_audio, to_mel_spectrogram, fix_length, TARGET_SR
from .conversion import convert_to_wav_bytes, ConversionError
from .config import MODEL_PATH, MIN_AUDIO_SECONDS

LABELS = ["healthy", "elevated_respiratory"]

class AudioTooShortError(Exception):
    pass

_model = tf.keras.models.load_model(MODEL_PATH)  # loaded once, at import time -- not per-request

def _load_audio_bytes(raw_bytes: bytes) -> np.ndarray:
    wav_bytes = convert_to_wav_bytes(raw_bytes)         # ConversionError propagates up to main.py
    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)                              # downmix stereo to mono
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    y, _ = librosa.effects.trim(y, top_db=20)
    y = librosa.util.normalize(y)
    return y

def to_risk_label(prob_elevated: float) -> dict:
    score = round(float(prob_elevated) * 100, 1)
    if score >= 70:
        level, message = "high", "Elevated respiratory sounds detected -- isolate the flock and consult a veterinarian."
    elif score >= 40:
        level, message = "moderate", "Some signs of respiratory stress. Monitor closely over the next 24-48 hours."
    else:
        level, message = "low", "Flock sounds healthy. No signs of respiratory distress detected."
    return {"risk_score": score, "risk_level": level, "message": message}

def analyze_audio_bytes(raw_bytes: bytes) -> dict:
    y = _load_audio_bytes(raw_bytes)

    if len(y) < MIN_AUDIO_SECONDS * TARGET_SR:
        raise AudioTooShortError(f"Recording too short -- need at least {MIN_AUDIO_SECONDS}s of clear audio.")

    windows = chunk_audio(y)
    specs = np.stack([fix_length(to_mel_spectrogram(w)) for w in windows])[..., np.newaxis]

    probs = _model.predict(specs, verbose=0)
    elevated_probs = probs[:, LABELS.index("elevated_respiratory")]
    max_prob = float(elevated_probs.max())              # matches Part 3/4's max-aggregation choice

    result = to_risk_label(max_prob)
    result["disclaimer"] = "This is a screening tool, not a diagnosis."
    result["windows_analyzed"] = len(windows)
    return result
```

## 5.8 The API
```python
# backend/app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .inference import analyze_audio_bytes, AudioTooShortError
from .conversion import ConversionError
from .schemas import AnalyzeResponse
from .config import MAX_FILE_MB

app = FastAPI(title="FlockCare API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten to your actual frontend origin before real deployment
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"audio/webm", "audio/ogg", "audio/mp4", "audio/wav", "audio/x-wav", "audio/mpeg"}

def _base_content_type(content_type):
    # browsers send e.g. "audio/webm;codecs=opus" -- strip the codec parameter before matching
    return (content_type or "").split(";")[0].strip().lower()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    if _base_content_type(file.content_type) not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, f"Unsupported audio format: {file.content_type}")

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(400, "Empty file.")
    if len(raw_bytes) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {MAX_FILE_MB}MB).")

    try:
        return analyze_audio_bytes(raw_bytes)
    except ConversionError as e:
        raise HTTPException(400, f"Could not read audio file: {e}")
    except AudioTooShortError as e:
        raise HTTPException(400, str(e))
    except Exception:
        # Log the real exception server-side (not shown here) -- never leak internals to the client
        raise HTTPException(500, "Analysis failed. Please try recording again.")
```

## 5.9 Running it locally
```bash
# from the repo root, so `ml` resolves as a package
PYTHONPATH=. uvicorn backend.app.main:app --reload
```
```bash
curl -X POST -F "file=@sample.wav" http://localhost:8000/analyze
```

## 5.10 Dockerfile
```dockerfile
FROM python:3.11-slim

# ffmpeg is required by pydub to decode webm/ogg/mp4 audio from browser recordings --
# easy to forget since local dev machines often already have it for unrelated reasons
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/ ./ml/
COPY backend/ ./backend/
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`backend/requirements.txt`:
```
fastapi
uvicorn[standard]
python-multipart
tensorflow
librosa
soundfile
pydub
numpy
```

---

## 5.11 Verification checklist before moving to Part 6
- [ ] `/health` returns `200 {"status": "ok"}`
- [ ] A real browser-recorded file (webm, not a clean wav you made yourself) round-trips through `/analyze` successfully -- test with actual `MediaRecorder` output, not just a `.wav` from your training set, since that's the gap 5.0 flags
- [ ] Corrupt/garbage bytes uploaded as `file` return a clean `400`, never a `500`
- [ ] A sub-5-second clip returns the `AudioTooShortError` message, not a crash
- [ ] `windows_analyzed` matches expectations for a known clip length (a real ~30s recording should show `6`, give or take one for the trailing-window rule in `chunk_audio`)
- [ ] Model loads once at startup -- confirm via startup logs, and confirm the *first* request isn't dramatically slower than the rest (would indicate lazy loading crept back in)
- [ ] Built and ran the **Docker image itself**, not just the local dev server -- this is where a missing ffmpeg install would actually surface
- [ ] Ran one known-healthy and one known-sick sample end-to-end through the deployed API and got the expected `risk_level` -- a fast smoke test before wiring up the frontend

---

## Handoff to Part 6
- **Base URL:** wherever this deploys (e.g. `http://localhost:8000` in dev)
- **`POST /analyze`:** `multipart/form-data`, field name **`file`**, any of `webm` / `ogg` / `mp4` / `wav` / `mpeg` audio, >=5 seconds of audio, <=15MB
- **Success response:**
  ```json
  {
    "risk_score": 87.0,
    "risk_level": "high",
    "message": "Elevated respiratory sounds detected -- isolate the flock and consult a veterinarian.",
    "disclaimer": "This is a screening tool, not a diagnosis.",
    "windows_analyzed": 6
  }
  ```
- **Error response (4xx):** `{ "detail": "<message>" }`
- **`GET /health`:** `{ "status": "ok" }` -- useful for a connectivity check on the frontend's first screen

Say the word when you're ready for Part 6 (React frontend).
