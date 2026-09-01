# FlockCare — Deployment, Verification & Operational Roadmap

*Production deployment guide, validation benchmarks, and presentation guide for FlockCare — the explainable avian bioacoustic intelligence platform for flock respiratory screening.*

---

## Executive Summary & What Has Been Built

FlockCare v1 is fully implemented, thoroughly tested, and ready for immediate deployment:

1. **Acoustic Neural Core & Safety Gating**:
   - Deep 2D Convolutional Neural Network trained on log-mel spectrogram windows ($1.5–4.5\text{ kHz}$ tracheal rale frequency bands).
   - RMS & peak energy pre-filtering rejecting silence/ambient noise before inference.
   - Mahalanobis-distance Out-of-Distribution (OOD) gating to flag anomalous acoustic environments.
   - Minimum $\ge 15\text{s}$ ($\ge 3$ consecutive windows) duration floor for multi-window consistency.

2. **Explainable AI (XAI) & Clinical Differentials**:
   - **Grad-CAM Saliency**: Real-time convolutional feature attention heatmaps overlaid on Mel-spectrograms with an interactive opacity slider.
   - **Directional SHAP Attribution**: Positive/negative feature impact waterfall chart explaining the neural risk score.
   - **Quantitative Bioacoustic Biomarkers**: Tracheal Rale Power %, Spectral Centroid (Hz), Spectral Flatness, and Acoustic Event Density %.
   - **5-Disease Differential Engine**: Probability matching for Infectious Bronchitis (IBV), Chronic Respiratory Disease (CRD), Infectious Coryza, Newcastle Disease (NDV), and Aspergillosis, paired with coop physical symptom checklists.

3. **Topic-Based Multi-Page Interface (Hers iOS Luxury Aesthetic)**:
   - Classic serif typography (`Playfair Display` + `Plus Jakarta Sans`), warm ivory palette (`#FAF8F5`), and authentic agricultural photography.
   - Starting Landing Page with 2-column photo category cards & feature preview modals.
   - Controlled Sign-In & Farm Registration with guest evaluation bypass.
   - Dedicated Topic Results Pages: *Executive Overview*, *Expected Diseases*, *Acoustic Saliency*, *Biomarkers & SHAP*, and *Veterinary Care Plan*.
   - One-click formatted Printable Clinical PDF Report for licensed avian veterinarians.

4. **Rigorous Test Coverage**:
   - **49 / 49 Vitest Frontend Tests Passing** across 14 test suites.
   - **59 / 59 Pytest Backend Tests Passing** across model, audio preprocessing, OOD gating, and API contracts.
   - Fast production bundle compilation (**190ms**).

---

## Part 1 — Deployment Guide

### 1.1 Backend Deployment Options

TensorFlow's memory footprint is the primary resource constraint. Choose one of the following two deployment pathways:

#### Pathway A: Hugging Face Spaces (Recommended for Demo & Production — 16 GB Free RAM)
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Select **Docker** as the Space SDK (Blank template).
3. In the Space's `README.md`, add YAML frontmatter:
   ```yaml
   ---
   title: FlockCare Bioacoustic API
   emoji: 🐔
   colorFrom: green
   colorTo: yellow
   sdk: docker
   app_port: 7860
   ---
   ```
4. Update the Dockerfile entrypoint or port configuration to listen on port `7860`:
   ```dockerfile
   EXPOSE 7860
   CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
   ```
5. Connect your GitHub repository (`https://github.com/daivikmank-bit/FlockCare.git`) or push directly.
6. Note the public URL: `https://<username>-flockcare.hf.space`.

#### Pathway B: Render / GCP Cloud Run / Fly.io (with TFLite Optimization)
1. For lower RAM hosting tiers ($< 512\text{ MB}$ RAM):
   - Export the model to TFLite format (`ml/saved_models/flockcare_cnn.tflite`).
   - Use `tflite-runtime` instead of full `tensorflow` in `requirements.txt`.
2. For GCP Cloud Run:
   ```bash
   gcloud run deploy flockcare-api \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 1Gi
   ```

---

### 1.2 Frontend Deployment (Vercel / Netlify)

1. Connect the GitHub repository to **Vercel** or **Netlify**:
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Node Version**: `18.x` or `20.x`
2. **Environment Variables**:
   - Set `VITE_API_BASE_URL` to your live backend endpoint (e.g. `https://<username>-flockcare.hf.space` or your custom domain).
3. Both Vercel and Netlify automatically provide **HTTPS**, which is mandatory for browser `navigator.mediaDevices.getUserMedia` microphone access.

---

### 1.3 Pre-Demo Verification Checklist

- [ ] **Warm up the backend**: Send a `GET /health` request 2 minutes before presenting to avoid idle container cold starts.
- [ ] **CORS Configuration**: In `backend/app/main.py`, ensure your Vercel/Netlify production frontend domain is listed in `allow_origins`.
- [ ] **Cross-Device Microphone Test**: Test recording on both an iOS device (Safari) and Android/Chrome to verify WebM/Opus streaming and permission prompts.
- [ ] **File Upload Fallback**: Keep a sample `.wav` recording on your test device to demonstrate instant file analysis if live ambient noise is unavailable during presentations.

---

## Part 2 — Live Presentation & Judge Demonstration Script

### 2-Minute Judge Walkthrough Flow

1. **Introduction (15s)**:
   - *"FlockCare is an AI-powered bioacoustic screening platform that detects respiratory distress in poultry flocks before physical symptoms appear, preventing flock-wide mortality."*
   - Highlight the **Hers-inspired editorial design** and official FlockCare brand crest.

2. **Starting Screen & Feature Modals (20s)**:
   - Tap on **"Respiratory Health"** or **"Disease Differential"** category photo cards to show the interactive feature modals and AI specifications.
   - Click **"Get started"** to transition to the Sign-In screen.

3. **Seamless Authentication (15s)**:
   - Demonstrate entering farm credentials or click **"Continue as Guest Farmer"** to enter the screening dashboard.

4. **Coop Audio Recording & Safety Filters (30s)**:
   - Tap the microphone button to record audio, showing real-time pulsing audio level visualization and multi-window progress tracking.
   - Note the server-side safety checks: RMS energy floor preventing silent uploads, and 15s / 3-window multi-window validation.

5. **Explainable AI & Topic-Based Diagnostics (40s)**:
   - Walk through the **Executive Overview**: Risk Index score, clinical summary, and audio player with synchronized time scrubbing.
   - Tap **"Expected Avian Diseases"**: Show differential matching for IBV, CRD, Coryza, and interactive symptom checkboxes.
   - Tap **"Acoustic Saliency"**: Switch between 5-second windows and adjust the **Grad-CAM attention heatmap slider** over the Mel-spectrogram.
   - Tap **"Biomarkers & SHAP"**: Explain the positive/negative feature attribution waterfall chart.
   - Tap **"Export Vet Report"**: Open the formatted **Printable Clinical PDF Report** designed for veterinarians.

---

## Part 3 — Technical Risk Matrix & Mitigations

| Risk | Potential Impact | Production Mitigation in FlockCare |
|---|---|---|
| **Ambient Non-Bird Noise** (fans, tractor hum) | False positives on background sounds | RMS pre-filter, spectral centroid biomarker thresholding, and SpecAugment frequency masking. |
| **Out-of-Domain Audio** (human speech, rain) | Misleading risk predictions | Mahalanobis OOD embedding gate flags uncalibrated recordings with an alert banner. |
| **Short / Cut-off Clips** | Incomplete screening | Server enforces $\ge 15\text{s}$ ($\ge 3$ full 5s windows) floor before accepting inference requests. |
| **Browser WebM Inconsistencies** | Player timestamp bugs (`NaN`/`Infinity`) | Robust duration sanitization with exact fallback to `windows_analyzed * 5.0s`. |
| **Misinterpretation of Output** | Producer delaying veterinary care | Prominent clinical disclaimers and one-click direct veterinarian locator. |

---

## Part 4 — Post-MVP Roadmap & Future Vision

### Phase 1: Cooperative & Veterinary Field Pilot
- Partner with poultry veterinary extensions and broiler cooperatives.
- Collect multi-season acoustic datasets across diverse ventilation setups (tunnel ventilation, open-sided coops, free-range pastures).
- Calibrate risk score thresholds against PCR/ELISA-confirmed pathogen panels.

### Phase 2: Fully Offline Edge Deployment
- Package the TFLite quantized acoustic model for zero-internet on-device screening via Progressive Web App (PWA) and Flutter mobile application.

### Phase 3: Regional Flock Health Early Warning Radar
- Anonymized, GPS-aggregated reporting system providing livestock health authorities and regional veterinarians with an early-warning outbreak radar for rapid containment of virulent avian strains (NDV, Avian Influenza).
