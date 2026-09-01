# FlockCare Frontend

*Editorial Luxury Web Application for Avian Bioacoustic Respiratory Screening.*

Built with **React 19**, **Vite**, and custom **Vanilla CSS** following a warm, editorial luxury aesthetic inspired by Hers iOS (`Playfair Display`, `Plus Jakarta Sans`, `#FAF8F5` palette).

---

## Key Features

- **Starting Screen & Feature Preview Modals**: Dual-column photographic category cards with interactive deep-dive modals explaining bioacoustic AI capabilities.
- **Authentication with Guest Bypass**: Seamless sign-in supporting farm credentials or instant single-click evaluation via "Continue as Guest Farmer".
- **Coop Microphone Recorder**: Real-time audio waveform/pulsing level visualization, 15-second minimum duration enforcement, and multi-window countdown.
- **Topic-Based Diagnostic Screens**:
  - **Executive Overview**: Master Risk Score, flock status badge, and synchronized audio waveform player with time scrubbing.
  - **Expected Diseases**: 5-disease differential matching (IBV, CRD, Coryza, NDV, Aspergillosis) with interactive physical symptom checklists.
  - **Acoustic Saliency**: Multi-window log-mel spectrogram viewer with interactive Grad-CAM convolutional attention heatmap overlay slider.
  - **Biomarkers & SHAP**: Directional feature attribution waterfall (+/- % impact) and quantitative acoustic biomarker metrics.
  - **Veterinary Care Plan**: Immediate biosecurity checklist and one-click printable clinical PDF report.
- **Comprehensive Vitest Suite**: 49 tests passing across 14 component and utility suites.

---

## Quickstart

### 1. Install Dependencies
```bash
npm install
```

### 2. Start Development Server
```bash
npm run dev
```
The app will be available at `http://localhost:5173`. Ensure the FastAPI backend is running on `http://localhost:8000` (or configure `VITE_API_BASE_URL` in `.env`).

### 3. Run Automated Tests
```bash
npm test
```

### 4. Build for Production
```bash
npm run build
```

---

## Directory Structure

```
frontend/
├── public/                 # Static branding assets, logo, and SVGs
├── src/
│   ├── screens/            # Screen views
│   │   ├── LandingScreen.jsx
│   │   ├── SignInScreen.jsx
│   │   ├── RecordScreen.jsx
│   │   ├── AnalyzingScreen.jsx
│   │   └── ResultScreen.jsx
│   ├── components/         # Reusable UI components
│   │   ├── AudioPlaybackBar.jsx
│   │   ├── SpectrogramViewer.jsx
│   │   ├── BiomarkerChart.jsx
│   │   ├── DiseaseDifferentialCard.jsx
│   │   └── VetReportModal.jsx
│   ├── lib/                # API client, audio recorder & history helpers
│   ├── i18n/               # Multilingual localization dictionaries
│   ├── test/               # Vitest test files (49 tests)
│   ├── App.jsx             # Main router & screen coordinator
│   ├── index.css           # Global typography, color tokens & design system
│   └── App.css             # Component-level styling
├── package.json
└── vite.config.js
```
