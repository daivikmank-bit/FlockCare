# FlockCare

*A stethoscope for the backyard flock — smartphone-only, AI-powered respiratory disease screening for smallholder poultry flocks.*

---

## 1. Overview
FlockCare analyzes audio recordings (~30 seconds) captured on a smartphone to detect early acoustic indicators of poultry respiratory diseases (such as Newcastle disease, infectious bronchitis, and avian influenza) before visual symptoms appear.

## 2. Repository Structure
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
│   │   ├── build_test_set.py      # Poultry Vocalization dataset builder (grouped by file_id)
│   │   ├── generate_sample_data.py# Synthetic test audio generator
│   │   └── verify_pipeline.py     # Part 3.9 verification checklist script
│   ├── models/                    # CNN architecture & model definitions (Part 4)
│   ├── training/                  # Training routines & data loaders (Part 4)
│   └── evaluation/                # Model evaluation & baseline benchmarking (Part 4)
├── backend/                       # FastAPI backend service (Part 5)
├── frontend/                      # React frontend application (Part 6)
├── tests/                         # Pytest test suite
│   └── test_preprocessing.py
├── Data_processing.md             # Detailed Part 3 specification
├── training.md                    # Detailed Part 4 specification
├── implementation.md              # Full 12-part master implementation plan
├── requirements.txt
└── pyproject.toml
```

---

## 3. Quickstart: Data Processing & Preprocessing

### 3.1 Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

### 3.2 Running Preprocessing Pipeline
1. **Generate Synthetic Sample Audio (or place real datasets in `data/raw/`):**
   ```bash
   python3 ml/preprocessing/generate_sample_data.py
   ```

2. **Extract Mel-Spectrograms for SmartEars Training Set:**
   ```bash
   python3 ml/preprocessing/build_train_set.py
   ```

3. **Extract Mel-Spectrograms for Held-Out Test Set (Grouped by `file_id`):**
   ```bash
   python3 ml/preprocessing/build_test_set.py
   ```

4. **Run Verification Checklist:**
   ```bash
   python3 ml/preprocessing/verify_pipeline.py
   ```

### 3.3 Running Automated Tests
```bash
pytest -v
```

---

## 4. Preprocessing Specifications
- **Sample Rate (`TARGET_SR`):** `22050 Hz`
- **Window Length (`WINDOW_SEC`):** `5 seconds`
- **Mel Bands (`N_MELS`):** `128`
- **Hop Length (`HOP_LENGTH`):** `512`
- **Target Time Frames (`TARGET_FRAMES`):** `216`
- **Output Tensor Shape per Window:** `(128, 216, 1)` (Normalized float32 $[0, 1]$)
- **Label Schema:** `healthy` ($0$) and `elevated_respiratory` ($1$)
