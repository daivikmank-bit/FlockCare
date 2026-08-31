# FlockCare — Implementation Plan

*A stethoscope for the backyard flock — from prototype to field-ready screening tool.*

**Team:** Daivik Mankame, Divyansh Acharya
**Theme:** Healthcare & Wellbeing
**Core idea:** Smartphone-only, AI-powered respiratory disease screening for backyard poultry flocks.

This document turns the FlockCare concept into a build plan, split into 12 parts that roughly follow build order — data → model → backend → frontend → integration → deployment → validation → roadmap. Each part is self-contained enough to hand to whichever teammate owns it.

---

## Part 1 — Foundations: Scope & Success Criteria

### 1.1 Problem recap
Poultry diseases like Newcastle disease, infectious bronchitis, and avian influenza show up acoustically before they're visually obvious — coughing, sneezing, wheezing. The detection science already exists (acoustic AI classification), but existing tools like NESTLER are built for commercial farms with dedicated microphones and sensor infrastructure. The gap FlockCare targets isn't new science — it's accessibility for smallholders with a few dozen birds and no budget for farm hardware.

### 1.2 What "done" looks like (MVP definition)
A farmer can:
1. Open a web/mobile page on any smartphone.
2. Record ~30 seconds of coop audio.
3. Get a plain-language result within a few seconds: healthy vs. elevated respiratory risk, with next-step guidance.

No installed sensors, no per-farm hardware, no account required for the MVP.

### 1.3 In scope vs. out of scope for v1
| In scope (v1) | Out of scope (v1 — roadmap item) |
|---|---|
| Binary/ternary classification (healthy vs. elevated respiratory) | Multi-disease differentiation (Newcastle vs. bronchitis vs. avian flu) |
| Record → analyze → result flow | Continuous background monitoring |
| One or two regional languages for output text | Full i18n across many languages |
| Cloud inference via FastAPI | On-device (offline) inference |
| Public-dataset-trained model | Vet-verified, field-validated model |

### 1.4 Suggested role split
Two-person team, so split by layer, with a shared preprocessing/model handoff:
- **ML + data owner:** dataset prep, spectrogram pipeline, CNN training, benchmarking.
- **Product owner:** FastAPI service, React frontend, deployment, demo polish.

Both should agree on the API contract (Part 7) in the first hour — it's the seam between your two workstreams, and locking it early avoids integration pain later.

---

## Part 2 — System Architecture

### 2.1 High-level flow
This mirrors the 7-step pipeline from your deck:

```
[Farmer's phone mic]
       │  record ~30s audio
       ▼
[1. Record Audio Data] ──▶ [2. Preprocess Audio] ──▶ [3. Convert to Spectrogram]
                                                              │
                                                              ▼
                                              [4. CNN Feature Extraction]
                                                              │
                                                              ▼
                                              [5. Classification (Softmax)]
                                                              │
                                                              ▼
                                              [6. Risk Score Generated]
                                                              │
                                                              ▼
                                              [7. Risk Language Label] ──▶ [Farmer sees result]
```

### 2.2 Component map

```
┌─────────────┐      HTTPS/REST      ┌──────────────┐      loads      ┌──────────────┐
│  Frontend    │ ───────────────────▶ │  Backend      │ ──────────────▶ │  Model        │
│  (React)     │ ◀─────────────────── │  (FastAPI)    │ ◀────────────── │  (TF/Keras)   │
│  Record UI   │      JSON result      │  /analyze     │    inference    │  .h5 / .tflite│
└─────────────┘                       └──────────────┘                 └──────────────┘
                                              │
                                              ▼
                                      ┌──────────────┐
                                      │ Preprocessing │
                                      │  (Librosa)    │
                                      └──────────────┘
```

Three deployable units: **frontend** (static site), **backend** (API + model server — can be one process for a hackathon), and the **model artifact**, versioned separately so it can be swapped without a full redeploy.

---

## Part 3 — Data Pipeline

### 3.1 Datasets
| Dataset | Size | Labels | Role |
|---|---|---|---|
| SmartEars | ~6,000 clips | Healthy / Sick / None | Primary training set |
| Poultry Vocalization Dataset | 346 WAV files | Healthy / Unhealthy / Noise | Held-out validation / robustness check |

The two datasets use different label vocabularies, so the first real engineering task is a **label mapping layer** — collapse both into one schema before anything touches the model:

```python
LABEL_MAP = {
    # SmartEars
    "healthy": "healthy",
    "sick": "elevated_respiratory",
    "none": "no_bird_sound",
    # Poultry Vocalization Dataset
    "Healthy": "healthy",
    "Unhealthy": "elevated_respiratory",
    "Noise": "no_bird_sound",
}
```

Decide early whether "no_bird_sound" is a third output class or filtered out before the model ever sees it (recommended for v1 — keep the CNN binary, and treat "no clear bird audio" as a rule-based signal-energy check in preprocessing).

### 3.2 Preprocessing
```python
# ml/preprocessing/audio_utils.py
import librosa
import numpy as np

SAMPLE_RATE = 22050
CLIP_SECONDS = 30
N_MELS = 128
HOP_LENGTH = 512

def load_and_clean(file_path, sr=SAMPLE_RATE, duration=CLIP_SECONDS):
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    y, _ = librosa.effects.trim(y, top_db=20)          # drop leading/trailing silence
    if len(y) < sr * duration:                          # pad short clips
        y = np.pad(y, (0, sr * duration - len(y)))
    y = librosa.util.normalize(y)
    return y, sr

def to_mel_spectrogram(y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)  # 0–1 normalize
    return mel_db.astype(np.float32)
```

This produces a fixed-shape `(128, ~1292, 1)` tensor per 30s clip at 22.05kHz — pin this shape once and reuse it in both training and the API, or you'll hit silent shape-mismatch bugs at inference time.

### 3.3 Augmentation & class balance
Real coops are noisy (fans, wind, other animals) in ways your training set won't fully reflect, so build in robustness early:
- **Augmentation:** time-shift, pitch-shift (±2 semitones), background-noise injection (fan/wind clips), SpecAugment-style frequency/time masking on the spectrogram itself.
- **Class balance:** check the healthy/sick split in SmartEars before training — if it's skewed, use class weights in the loss function rather than naive oversampling, so you don't overfit to duplicated clips.

### 3.4 Data directory convention
```
data/
├── raw/                    # untouched original files
│   ├── smartears/
│   └── poultry_vocalization/
├── processed/
│   ├── train/{healthy,elevated_respiratory}/
│   ├── val/{healthy,elevated_respiratory}/
│   └── test/{healthy,elevated_respiratory}/   ← held out from Poultry Vocalization Dataset
└── spectrograms/            # cached .npy mel-spectrograms, keyed by clip id
```
Cache extracted spectrograms as `.npy` — recomputing Librosa mel-spectrograms on every training run is the easiest place to waste build time.

---

## Part 4 — Model Development

### 4.1 Architecture
A lightweight CNN is the right call: you don't have millions of labeled clips, and inference needs to feel instant on a phone-triggered API call.

```python
# ml/models/cnn_model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape, num_classes=2):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall")],
    )
    return model
```

`GlobalAveragePooling2D` instead of `Flatten` + a large Dense layer keeps the parameter count small and makes the model less sensitive to exact clip length — useful since real recordings won't always be a clean 30.0s.

### 4.2 Training
- **Environment:** Google Colab + T4 GPU.
- **Split:** train/val from SmartEars (e.g. 80/20, stratified by class). Keep the Poultry Vocalization Dataset entirely held out as your test set — it's the honest generalization number, not something you tune against.
- **Class weights:** computed from the train split, passed to `model.fit(..., class_weight=...)`.
- **Callbacks:** `EarlyStopping(monitor="val_recall", mode="max", patience=5)`. For a health-screening tool, optimize for **recall on the "elevated_respiratory" class**, not raw accuracy — a missed sick bird is a worse outcome than a false alarm.
- **Checkpointing:** save the best-val-recall model, not just best-val-loss.

### 4.3 Evaluation
Report at minimum:
- Confusion matrix on the held-out Poultry Vocalization Dataset — your only true out-of-distribution test.
- Precision/recall/F1 per class.
- Sensitivity (recall on "sick") called out explicitly, since it drives the risk-score design in 4.5.

### 4.4 Benchmarking
Compare against the Hugging Face chicken-vocalization CNN baseline, same held-out set, same metrics. Frame this as "does a deployment-optimized model hold up against an existing baseline" rather than a claim of beating it — your differentiator is accessibility, not raw model novelty (consistent with your own "Access, Not Reinvention" positioning).

### 4.5 From softmax to a risk label
This is the "6 → 7" step in your pipeline. Keep the mapping simple and conservative:

```python
def to_risk_label(prob_elevated: float) -> dict:
    score = round(prob_elevated * 100, 1)
    if score >= 70:
        level = "high"
        message = "Elevated respiratory sounds detected — isolate the flock and consult a veterinarian."
    elif score >= 40:
        level = "moderate"
        message = "Some signs of respiratory stress. Monitor closely over the next 24–48 hours."
    else:
        level = "low"
        message = "Flock sounds healthy. No signs of respiratory distress detected."
    return {"risk_score": score, "risk_level": level, "message": message}
```

Treat 70/40 as starting thresholds to tune against your held-out set, not fixed values. Always pair the result with a disclaimer string ("Not a diagnosis — consult a vet if symptoms persist") — this matters ethically and for the demo, since it's screening, not diagnosis.

### 4.6 Export
Save two artifacts: a full Keras `.h5`/`SavedModel` for the FastAPI server now, and (roadmap item) a `.tflite` conversion for future on-device inference.

---

## Part 5 — Backend (FastAPI)

### 5.1 Structure
```
backend/
├── app/
│   ├── main.py
│   ├── inference.py
│   ├── schemas.py
│   └── config.py
├── saved_models/
│   └── flockcare_cnn.h5
├── requirements.txt
└── Dockerfile
```

### 5.2 Core service
```python
# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os

from .inference import analyze_clip

app = FastAPI(title="FlockCare API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten before real deployment
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

MAX_FILE_MB = 15

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ("audio/wav", "audio/webm", "audio/mpeg", "audio/mp4"):
        raise HTTPException(400, "Unsupported audio format.")

    contents = await file.read()
    if len(contents) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {MAX_FILE_MB}MB).")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = analyze_clip(tmp_path)
    finally:
        os.remove(tmp_path)

    return result
```

```python
# app/inference.py
import numpy as np
import tensorflow as tf
from ml.preprocessing.audio_utils import load_and_clean, to_mel_spectrogram
from ml.models.risk import to_risk_label

_model = tf.keras.models.load_model("saved_models/flockcare_cnn.h5")
LABELS = ["healthy", "elevated_respiratory"]

def analyze_clip(path: str) -> dict:
    y, sr = load_and_clean(path)
    mel = to_mel_spectrogram(y, sr)
    x = mel[np.newaxis, ..., np.newaxis]
    probs = _model.predict(x, verbose=0)[0]
    prob_elevated = float(probs[LABELS.index("elevated_respiratory")])
    return to_risk_label(prob_elevated)
```

### 5.3 Notes
- Load the model **once at import time**, not per-request — this is the difference between a snappy demo and a slow one.
- Return a structured 4xx error (not a bare 500) for corrupt or too-short audio; check clip length against a minimum (e.g. 5s) before running inference — garbage audio in is a garbage risk score out.
- `/health` costs nothing to add and is worth having for the deployment step in Part 8.

---

## Part 6 — Frontend (React)

### 6.1 Screens
Matches your mockup: **Record → Analyzing → Result**, plus a lightweight **History** view (local state is fine for a hackathon — no auth needed for v1).

```
frontend/src/
├── screens/
│   ├── RecordScreen.jsx
│   ├── AnalyzingScreen.jsx
│   └── ResultScreen.jsx
├── components/
│   ├── RiskBadge.jsx
│   └── RecordButton.jsx
├── i18n/
│   ├── en.json
│   └── <regional-language>.json
├── api.js
└── App.jsx
```

### 6.2 Recording
```jsx
// RecordScreen.jsx (core logic)
const [recording, setRecording] = useState(false);
const recorderRef = useRef(null);

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  const chunks = [];
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: "audio/webm" });
    onRecordingComplete(blob);
  };
  recorder.start();
  recorderRef.current = recorder;
  setRecording(true);
  setTimeout(() => recorder.stop(), 30000); // auto-stop at 30s, matches model input
}

function stopRecording() {
  recorderRef.current?.stop();
  setRecording(false);
}
```

### 6.3 Calling the API
```js
// api.js
const API_BASE = import.meta.env.VITE_API_BASE_URL;

export async function analyzeRecording(blob) {
  const form = new FormData();
  form.append("file", blob, "coop_audio.webm");
  const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: form });
  if (!res.ok) throw new Error("Analysis failed");
  return res.json();
}
```

### 6.4 Result screen & multilingual output
Keep display copy per `risk_level` (`low` / `moderate` / `high`) in the i18n JSON files, keyed the same way as the backend's response, so the backend owns the risk logic and the frontend owns translation:

```json
// i18n/en.json (excerpt)
{
  "low": "Flock sounds healthy. No signs of respiratory distress detected.",
  "moderate": "Some signs of respiratory stress. Monitor closely for 24–48 hours.",
  "high": "Elevated respiratory sounds detected. Isolate the flock and consult a veterinarian.",
  "disclaimer": "This is a screening tool, not a diagnosis."
}
```

### 6.5 UX notes for the target user
Your farmer persona likely has variable literacy and a low-end Android phone: lean on icons and color (the green/amber/red pattern from your mockup) over dense text, keep the whole flow to three taps, and give the record button a large hit target with a visible countdown so a first-time user isn't left guessing whether it's working.

---

## Part 7 — API Contract

Fix this early — it's the seam between your two workstreams.

| Endpoint | Method | Request | Response |
|---|---|---|---|
| `/health` | GET | — | `{ "status": "ok" }` |
| `/analyze` | POST | `multipart/form-data`, field `file` (audio, ≤15MB, ≥5s) | see below |

**`/analyze` success response:**
```json
{
  "risk_score": 87.0,
  "risk_level": "high",
  "message": "Elevated respiratory sounds detected — isolate the flock and consult a veterinarian.",
  "disclaimer": "This is a screening tool, not a diagnosis."
}
```

**Error response (4xx):**
```json
{ "detail": "Unsupported audio format." }
```

---

## Part 8 — Deployment

### 8.1 Suggested stack for a demo-ready deployment
- **Backend:** containerize with Docker, deploy to Render / Railway / Fly.io — any of these give you a public HTTPS URL with minimal config, which matters if evaluators will test it live on their own phones.
- **Frontend:** static build deployed to Vercel/Netlify, with `VITE_API_BASE_URL` pointed at the backend URL.
- **Model artifact:** ship inside the backend Docker image for v1 (simplest); move to object storage behind a model registry once you're iterating on model versions independently of API code.

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 8.2 Connectivity reality check
Your target users are in low-connectivity rural settings, which cuts against a cloud-only design. Call this out explicitly as a known v1 limitation and roadmap item (Part 12) rather than something the demo needs to solve — cloud inference is the right tradeoff for now, since it's faster to build and lets you iterate on the model without shipping app updates.

---

## Part 9 — Testing & Validation

### 9.1 Automated checks worth having before a demo
- **Preprocessing unit test:** a known clip in → expected spectrogram shape out.
- **API contract test:** valid audio in → 200 with all expected keys; corrupt file in → clean 4xx, not a 500.
- **Model sanity check:** one clearly-healthy and one clearly-sick sample from the test set both classify correctly — a fast smoke test to catch a broken model load before a live demo.

### 9.2 Model validation
The Poultry Vocalization Dataset held-out test (Part 4.3) is your primary validation signal, since it's a genuinely different data source than SmartEars.

### 9.3 Field validation (post-MVP, but worth planning now)
The deck already names this as the next phase:
1. Partner with a local poultry cooperative or veterinary college.
2. Collect labeled recordings from real coops, ideally with vet-confirmed health status per flock.
3. Re-tune the decision threshold on this data — real coop noise (fans, other animals, wind) will very likely shift the optimal cutoff from what you tune on clean public datasets.
4. Track false-negative rate specifically in the field pilot — that number determines whether this is safe to hand to a farmer as a decision aid.

---

## Part 10 — Roadmap

### 10.1 If this is time-boxed (hackathon build order)
Priority order to always have something demoable:
1. Preprocessing pipeline + cached spectrograms (Part 3).
2. Baseline CNN trained on SmartEars only, no augmentation yet (Part 4.1–4.2) — get *a* model working end-to-end before optimizing it.
3. Minimal FastAPI `/analyze` endpoint wired to that model (Part 5).
4. Minimal frontend: record button → result screen, no polish (Part 6).
5. **End-to-end smoke test** — this is the milestone that de-risks the demo.
6. Only after that: augmentation, benchmarking, multilingual output, UI polish, deployment.

### 10.2 Post-MVP roadmap
- **Phase 2:** augmentation + class-balancing refinement, benchmarking vs. the Hugging Face baseline, multilingual output, deployment to a public URL.
- **Phase 3:** field validation with a cooperative/vet partner (Part 9.3), threshold re-tuning on real-world data.
- **Phase 4:** on-device (TFLite) inference for offline use, expanded disease coverage, SMS/IVR fallback for feature phones.

---

## Part 11 — Risks & Mitigations

| Risk | Mitigation |
|---|---|
| False negatives (missed sick birds) — the costliest failure mode for a health tool | Optimize threshold for recall over accuracy (4.2); always pair results with a "consult a vet" disclaimer |
| Real coop noise not represented in public datasets | Background-noise augmentation (3.3); field validation (9.3) |
| Model bias from limited dataset diversity (species, region, mic hardware) | Treat the held-out dataset test as a floor, not a ceiling; plan regional data collection before wider rollout |
| Phone mic quality varies widely across low-end Android devices | Normalize aggressively in preprocessing (3.2); test on a few real low-end devices before the demo, not just a laptop mic |
| Over-claiming diagnostic accuracy | Consistent "screening, not diagnosis" language in both the API response and UI (4.5, 6.4) |

---

## Part 12 — Future Enhancements (beyond v1)
- On-device inference (TFLite) for fully offline screening.
- SMS/IVR-based results for farmers without smartphones.
- "Find nearest vet" integration, as shown in your mockup — a lightweight directory lookup by region.
- Multi-disease differentiation beyond a binary healthy/elevated-respiratory signal.
- Community-level dashboard aggregating anonymized regional results into an early-warning heatmap for local disease spread.

---

## Appendix — Suggested Repo Structure
```
flockcare/
├── data/
├── ml/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── backend/
│   └── app/
├── frontend/
│   └── src/
├── docs/
│   └── implementation.md   ← this file
└── README.md
```
