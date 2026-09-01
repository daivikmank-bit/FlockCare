---
title: FlockCare Bioacoustic API
emoji: 🐔
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
---

# FlockCare

*A stethoscope for the backyard flock — smartphone-only, AI-powered bioacoustic respiratory disease screening for smallholder poultry flocks.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![LiteRT](https://img.shields.io/badge/Google_LiteRT-TFLite-FFA800.svg)](https://ai.google.dev/edge/litert)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-6.x-646CFF.svg)](https://vitejs.dev)
[![Pytest](https://img.shields.io/badge/Pytest-59%20passed-brightgreen.svg)](https://docs.pytest.org)
[![Vitest](https://img.shields.io/badge/Vitest-49%20passed-brightgreen.svg)](https://vitest.dev)

---

## 1. Overview & Key Capabilities

FlockCare analyzes audio recordings (~15–30 seconds) captured via a standard smartphone microphone to detect early acoustic indicators of poultry respiratory distress (such as Infectious Bronchitis, Newcastle Disease, Chronic Respiratory Disease, Infectious Coryza, and Aspergillosis) before visible physical symptoms or mortality events occur.

### Highlights:
- **Ultra-Lightweight Inference Engine**: Powered by Google LiteRT / TFLite (~10MB wheel, **1.8ms inference per window**, and ~35MB RAM footprint), allowing deployment on free cloud tiers or edge devices.
- **Explainable AI (XAI)**:
  - **Grad-CAM Saliency Heatmaps**: Dynamic convolutional feature attention heatmaps overlaid on log-mel spectrograms with interactive opacity controls.
  - **Directional SHAP Feature Attribution**: Positive and negative biomarker contribution waterfall explaining the neural risk score.
  - **Quantitative Bioacoustic Biomarkers**: Tracheal Rale Power %, Spectral Centroid (Hz), Spectral Flatness, and Acoustic Event Density %.
- **5-Disease Differential Engine**: Probability matching across major avian respiratory pathogens combined with clinical physical coop symptom checklists.
- **Multi-Tier Safety Gating**:
  - **RMS Energy Floor**: Filters silence and non-avian ambient noise before neural inference.
  - **Multi-Window Duration Consistency**: Requires $\ge 15\text{s}$ ($\ge 3$ consecutive 5s windows) to prevent false positives from transient clicks.
  - **Mahalanobis Out-of-Distribution (OOD) Gating**: Detects and flags anomalous non-coop acoustic environments (human speech, rain, engine rumble).
- **Editorial Luxury UI (Hers-Inspired)**:
  - Warm ivory palette (`#FAF8F5`), classic serif typography (`Playfair Display` + `Plus Jakarta Sans`), and authentic agricultural visuals.
  - Topic-based diagnostic navigation: *Executive Overview*, *Expected Diseases*, *Acoustic Saliency*, *Biomarkers & SHAP*, and *Veterinary Care Plan*.
  - **One-Click Printable Clinical PDF Report** designed for licensed avian veterinarians.

---

## 2. End-to-End System Architecture

```
                               ┌────────────────────────────────────────────────────────┐
                               │                 Farmer Audio Capture                   │
                               │        Smartphone Web Browser (WebM / WAV / MP4)       │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
                               ┌────────────────────────────────────────────────────────┐
                               │             FastAPI Backend Ingestion                  │
                               │        pydub/ffmpeg & soxr fast resample (22.05kHz)    │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
                               ┌────────────────────────────────────────────────────────┐
                               │           Acoustic Preprocessing & Gating              │
                               │   - RMS & Peak Energy Pre-filter (>0.005 threshold)    │
                               │   - 5-Second Window Slicing (hop: 512, 128 mel bands)   │
                               │   - Log-Mel Spectrogram Shape: (128, 216, 1)           │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
                               ┌────────────────────────────────────────────────────────┐
                               │        Google LiteRT / TFLite Neural Core              │
                               │   - Deep 2D-CNN (Tracheal rale band: 1.5 - 4.5 kHz)   │
                               │   - Mahalanobis OOD Embedding Anomaly Gate             │
                               │   - Temporal Multi-Window Risk Aggregation             │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
                               ┌────────────────────────────────────────────────────────┐
                               │           Explainability & Differential AI             │
                               │   - Grad-CAM Convolutional Saliency Heatmaps           │
                               │   - SHAP Directional Biomarker Attribution Waterfall   │
                               │   - 5-Pathogen Avian Differential Diagnostic Engine    │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
                               ┌────────────────────────────────────────────────────────┐
                               │            FlockCare Editorial Frontend                │
                               │   - Topic-Based Diagnostic Dashboard Screens           │
                               │   - Synchronized Audio Waveform Player with Scrubbing  │
                               │   - One-Click Printable Clinical Vet Report            │
                               └────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
flockcare/
├── data/
│   ├── raw/                       # Raw audio files (.wav)
│   │   ├── smartears/             # SmartEars dataset (healthy, sick, none)
│   │   └── poultry_vocalization/ # Held-out test dataset (healthy, unhealthy, noise)
│   └── spectrograms/              # Extracted 2D mel-spectrogram arrays (.npy)
│       ├── train/                 # (128, 216) spectrograms for CNN training
│       ├── test/                  # (128, 216) spectrograms for held-out evaluation
│       ├── train_manifest.csv     # Training manifest (clip_id, source_path, label, dataset)
│       └── test_manifest.csv      # Test manifest (window_id, file_id, source_path, label, dataset)
├── ml/
│   ├── preprocessing/             # Audio processing & feature extraction pipeline
│   │   ├── labels.py              # Label mapping & energy pre-filter
│   │   ├── audio_utils.py         # Audio loading, chunking (5s), mel-spectrogram, aggregation
│   │   ├── augment.py             # Pitch shift, time shift, noise addition, SpecAugment
│   │   ├── build_train_set.py     # SmartEars dataset builder
│   │   ├── build_test_set.py      # Poultry Vocalization dataset builder
│   │   ├── generate_sample_data.py# Synthetic test audio generator
│   │   └── verify_pipeline.py     # Preprocessing verification script
│   ├── models/                    # Model architecture & inference core
│   │   ├── cnn_model.py           # Deep 2D-CNN architecture definition
│   │   ├── ood_gate.py            # Mahalanobis distance OOD gating
│   │   ├── risk.py                # Temporal window risk aggregation
│   │   └── export.py              # TFLite conversion & quantization utilities
│   ├── training/                  # Model training pipeline
│   │   ├── data.py                # tf.data pipeline with SpecAugment
│   │   ├── train.py               # Model training script with early stopping
│   │   └── verify_training_pipeline.py # End-to-end training verification
│   ├── evaluation/                # Benchmarking & metrics
│   │   ├── evaluate.py            # Held-out file-grouped evaluation
│   │   └── baseline_model.py      # Logistic regression baseline benchmark
│   └── saved_models/              # Serialized model weights & embeddings
│       ├── flockcare_cnn.h5       # Trained Keras neural network
│       ├── flockcare_cnn.tflite   # Quantized LiteRT production model (~10MB)
│       └── ood_reference.npz      # In-distribution embedding distribution
├── backend/                       # Production FastAPI Service
│   ├── app/
│   │   ├── main.py                # FastAPI endpoints & CORS configuration
│   │   ├── inference.py           # LiteRT / TFLite inference runner & threadpool
│   │   ├── explainability.py      # Grad-CAM, SHAP attribution, and acoustic biomarkers
│   │   ├── disease_differential.py# 5-disease avian differential diagnostic engine
│   │   ├── conversion.py          # Multi-format audio decoding (WebM/Opus, MP4, WAV)
│   │   ├── schemas.py             # Pydantic request/response data contracts
│   │   └── config.py              # Environment configuration & thresholds
│   ├── requirements.txt           # Production backend dependencies
│   └── Dockerfile                 # Standalone backend container definition
├── frontend/                      # Editorial React + Vite Frontend
│   ├── public/                    # Static assets, branding logo & icons
│   ├── src/
│   │   ├── screens/               # Topic-based diagnostic screens
│   │   │   ├── LandingScreen.jsx  # Hero landing page with feature cards & modals
│   │   │   ├── SignInScreen.jsx   # Authentication with guest farmer bypass
│   │   │   ├── RecordScreen.jsx   # Audio recording with live visualizer
│   │   │   ├── AnalyzingScreen.jsx# Acoustic analysis progress screen
│   │   │   └── ResultScreen.jsx   # Results hub & topic view switcher
│   │   ├── components/            # Reusable UI components
│   │   │   ├── AudioPlaybackBar.jsx         # Waveform audio player with scrubbing
│   │   │   ├── SpectrogramViewer.jsx        # Mel-spectrogram & Grad-CAM overlay
│   │   │   ├── BiomarkerChart.jsx           # SHAP waterfall & biomarker graphs
│   │   │   ├── DiseaseDifferentialCard.jsx  # Disease differential cards & checklist
│   │   │   └── VetReportModal.jsx           # Printable Clinical PDF Report
│   │   ├── lib/                   # API clients, audio recorder, and history storage
│   │   ├── i18n/                  # Multilingual localization dictionaries
│   │   └── test/                  # Vitest component test suites (14 suites, 49 tests)
│   ├── package.json
│   └── vite.config.js
├── tests/                         # Pytest Backend & ML Test Suite (59 tests)
│   ├── test_preprocessing.py      # Audio loading, mel-spectrogram, chunking tests
│   ├── test_cnn_model.py          # CNN layer shape & gradient tests
│   ├── test_risk.py               # Aggregation & temporal scoring tests
│   ├── test_ood_gate.py           # Mahalanobis distance OOD gating tests
│   ├── test_explainability.py     # Grad-CAM & biomarker extraction tests
│   ├── test_backend.py            # FastAPI API contract & audio conversion tests
│   └── test_model_export.py       # TFLite inference consistency tests
├── Dockerfile                     # Root deployment Dockerfile (Hugging Face / Cloud Run)
├── requirements.txt               # Top-level dependencies
├── pyproject.toml                 # Package configuration
├── deploy_and_beyond.md           # Production deployment & judge presentation guide
├── Data_processing.md             # Data processing & audio feature extraction guide
├── training.md                    # CNN model training & baseline benchmarks guide
├── Backend.md                     # FastAPI backend architecture guide
├── frontend.md                    # React frontend architecture & design system guide
└── implementation.md              # 12-part master implementation blueprint
```

---

## 4. Quickstart Guide

### 4.1 Prerequisites
- **Python**: 3.10 or higher
- **Node.js**: 18.x or 20.x
- **System Audio Codecs**: `ffmpeg` (for multi-format audio decoding)
  - macOS: `brew install ffmpeg`
  - Linux (Debian/Ubuntu): `sudo apt-get install -y ffmpeg`

---

### 4.2 Backend Setup & Execution

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   pip install -r backend/requirements.txt
   ```

3. **Start the FastAPI backend server:**
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   - API Welcome: `http://localhost:8000/`
   - Interactive OpenAPI Docs: `http://localhost:8000/docs`
   - Health Check: `http://localhost:8000/health`

---

### 4.3 Frontend Setup & Execution

1. **Navigate to the frontend directory and install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Start the Vite development server:**
   ```bash
   npm run dev
   ```
   - Open your browser at `http://localhost:5173`.
   - Ensure the backend is running on `http://localhost:8000`.

---

### 4.4 ML Training & Preprocessing Pipeline (Optional)

1. **Generate Synthetic Sample Audio:**
   ```bash
   python3 ml/preprocessing/generate_sample_data.py
   ```

2. **Extract Log-Mel Spectrograms:**
   ```bash
   python3 ml/preprocessing/build_train_set.py
   python3 ml/preprocessing/build_test_set.py
   ```

3. **Train the 2D-CNN & Export LiteRT / TFLite Model:**
   ```bash
   python3 ml/training/train.py
   python3 ml/models/export.py
   ```

4. **Benchmark Model & Evaluate Held-Out Data:**
   ```bash
   python3 ml/evaluation/baseline_model.py
   python3 ml/evaluation/evaluate.py
   ```

---

## 5. Automated Test Suites

The repository contains end-to-end automated test suites for both backend/ML and frontend systems.

### 5.1 Run Backend & ML Pytest Suite (59 Tests)
```bash
pytest -v
```
*Covers audio normalization, mel-spectrogram tensors, SpecAugment, CNN architectures, Mahalanobis OOD gating, Grad-CAM saliency, SHAP attribution, audio format conversions (WebM/WAV/MP3), and FastAPI endpoints.*

### 5.2 Run Frontend Vitest Suite (49 Tests across 14 Suites)
```bash
cd frontend
npm test
```
*Covers landing pages, guest farmer authentication, audio recording with Web Audio API mocks, topic result screens, Grad-CAM heatmap overlays, SHAP charts, disease differential cards, and clinical PDF report generation.*

---

## 6. Key API Endpoints

### `POST /analyze`
Analyzes an uploaded audio recording and returns comprehensive screening diagnostics.
- **Request**: Multipart Form (`file`: WebM, OGG, MP4, WAV, MP3, AAC, FLAC)
- **Response Schema (`AnalyzeResponse`)**:
```json
{
  "risk_score": 78.4,
  "risk_level": "high",
  "message": "High acoustic indicators of respiratory distress detected in the flock.",
  "disclaimer": "FlockCare is an AI screening tool and does not replace veterinary diagnosis.",
  "windows_analyzed": 6,
  "status": "calibrated",
  "ood_score": 1.42,
  "windows_detail": [
    {
      "window_index": 0,
      "start_sec": 0.0,
      "end_sec": 5.0,
      "risk_score": 82.1,
      "ood_score": 1.38,
      "is_ood": false,
      "spectrogram_image": "data:image/png;base64,...",
      "heatmap_image": "data:image/png;base64,...",
      "biomarkers": {
        "tracheal_rale_power_pct": 24.6,
        "spectral_centroid_hz": 2840.5,
        "spectral_flatness": 0.038,
        "acoustic_event_density_pct": 32.0
      }
    }
  ],
  "overall_biomarkers": {
    "tracheal_rale_power_pct": 22.8,
    "spectral_centroid_hz": 2790.2,
    "spectral_flatness": 0.035,
    "acoustic_event_density_pct": 28.5
  },
  "feature_importance": [
    {
      "feature_name": "Tracheal Rale Band Power (1.5-4.5 kHz)",
      "value": "22.8%",
      "impact": 38.5,
      "direction": "increases_risk",
      "clinical_significance": "Excess acoustic energy in the avian tracheal resonance frequency band."
    }
  ],
  "disease_differential": {
    "flock_clinical_status": "Severe Respiratory Distress Pattern",
    "primary_concern": "Infectious Bronchitis (IBV) / Infectious Coryza",
    "differentials": [
      {
        "disease_id": "ibv",
        "name": "Infectious Bronchitis Virus (IBV)",
        "pathogen": "Coronavirus",
        "likelihood": "High",
        "probability_pct": 84,
        "acoustic_rationale": "High-frequency tracheal rales and snoring sounds in the 2.0-4.0 kHz band.",
        "is_notifiable": false,
        "key_symptoms": ["Tracheal rales / coughing", "Watery eyes", "Egg shell defects"],
        "biosecurity_actions": ["Isolate affected flock", "Improve coop ventilation", "Administer electrolytes"]
      }
    ]
  }
}
```

### `GET /health`
Liveness check returning server and model readiness.
```json
{
  "status": "ok"
}
```

---

## 7. Bioacoustic Specifications & Preprocessing Parameters

| Parameter | Value | Description |
|---|---|---|
| **Target Sample Rate (`TARGET_SR`)** | `22,050 Hz` | Standard bioacoustic audio rate capturing poultry vocalizations up to 11 kHz |
| **Window Length (`WINDOW_SEC`)** | `5.0 seconds` | Duration of each individual temporal analysis window |
| **Mel Filter Banks (`N_MELS`)** | `128` | Frequency resolution emphasizing $1.5–4.5\text{ kHz}$ tracheal rale zones |
| **Hop Length (`HOP_LENGTH`)** | `512 samples` | Frame shift giving 216 time bins per 5s window (~23ms hop) |
| **FFT Size (`N_FFT`)** | `2048 samples` | Spectral frequency resolution (~10.7 Hz per bin) |
| **Spectrogram Tensor Shape** | `(128, 216, 1)` | Standardized 2D matrix fed to the CNN model |
| **Energy Pre-filter Threshold** | `RMS > 0.005` | Rejects silent/non-signal audio chunks before neural inference |
| **Minimum Audio Duration** | `15.0 seconds` | Enforces $\ge 3$ consecutive windows for temporal consistency |

---

## 8. Deployment Pathways

### Pathway A: Hugging Face Spaces (Docker)
1. Push this repository to a Docker Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. The included `Dockerfile` and Space metadata in `README.md` will automatically build the environment and expose port `7860`.

### Pathway B: Render / GCP Cloud Run / Fly.io (LiteRT)
- Deploy using the Google LiteRT engine (`ai-edge-litert`).
- Requires $< 512\text{ MB}$ RAM and starts in under 2 seconds.

### Pathway C: Frontend on Vercel / Netlify
- Set the root directory to `frontend/`.
- Set `VITE_API_BASE_URL` to your production backend endpoint.
- HTTPS is automatically provisioned, ensuring seamless browser microphone access.

*For detailed deployment instructions and presentation scripts, see [`deploy_and_beyond.md`](file:///Users/daivikmankame/FlockCare/FlockCare/deploy_and_beyond.md).*

---

## 9. Comprehensive Documentation Sitemap

| Document | Purpose |
|---|---|
| [`deploy_and_beyond.md`](file:///Users/daivikmankame/FlockCare/FlockCare/deploy_and_beyond.md) | Production deployment guide, validation benchmarks, and judge presentation script |
| [`Data_processing.md`](file:///Users/daivikmankame/FlockCare/FlockCare/Data_processing.md) | Audio feature extraction, Mel-spectrogram generation, and augmentation specs |
| [`training.md`](file:///Users/daivikmankame/FlockCare/FlockCare/training.md) | 2D-CNN model architecture, training routines, OOD gating, and benchmark results |
| [`Backend.md`](file:///Users/daivikmankame/FlockCare/FlockCare/Backend.md) | FastAPI backend service, audio conversion pipeline, and schema specs |
| [`frontend.md`](file:///Users/daivikmankame/FlockCare/FlockCare/frontend.md) | Topic-based luxury UI architecture, audio recording, and XAI components |
| [`implementation.md`](file:///Users/daivikmankame/FlockCare/FlockCare/implementation.md) | Complete 12-part master engineering roadmap and design blueprint |

---

## 10. License & Disclaimers

FlockCare is provided for flock management screening and bioacoustic research. It is designed to assist smallholder poultry producers and agricultural extensions in the early detection of respiratory health risks. It does not replace definitive microbiological, serological, or veterinary clinical diagnosis.
