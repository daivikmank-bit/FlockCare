# FlockCare — Part 3: Data Pipeline & Preprocessing

*Standalone build guide. This file is self-contained — everything needed to build the data pipeline is here, so it can be handed to a coding assistant on its own without the rest of the implementation plan.*

**Goal of this part:** turn two public poultry-audio datasets into a clean, fixed-shape spectrogram dataset ready for CNN training in Part 4.

**Output shape this part hands off to Part 4:** every training example is a `(128, 216, 1)` mel-spectrogram — one 5-second audio window, 128 mel bands, ~216 time frames. Part 4's model input layer must match this exactly.

---

## 3.0 A correction worth knowing before you start

The original plan assumed one ~30-second clip per prediction. The real datasets don't support that directly:
- **SmartEars** (your primary training set) is made of 5-second clips, not 30-second ones.
- **Poultry Vocalization Dataset** (your held-out test set) has variable-length files, 5–60 seconds each.

So the design here is: **train on 5-second windows**, and at inference time (farmer's ~30s recording), **slice into six 5-second windows and aggregate** the six predictions into one flock-level result. This is a better design anyway — a cough might only show up in 1 of 6 windows, and per-window scoring catches that instead of averaging it away in one long clip.

---

## 3.1 Environment setup
```bash
pip install librosa numpy pandas soundfile tqdm
```
`soundfile` is librosa's backend for reading `.wav` reliably; `tqdm` is just for progress bars while processing thousands of clips.

---

## 3.2 Datasets — exact sources

| Dataset | Source | Size | Labels |
|---|---|---|---|
| SmartEars | Mendeley Data — `data.mendeley.com/datasets/dy6gtvt4mk` (Huang, Zhang, Cuan & Fang) | 6,000 five-second clips, 2,000 per class | Healthy / Sick / None (no chicken sound) |
| Poultry Vocalization Signal Dataset for Early Disease Detection | Mendeley Data — `data.mendeley.com/datasets/zp4nf2dxbh` (Aworinde, Adebayo, Akinwunmi et al., 2023) | 346 `.wav` files (139 healthy / 121 unhealthy / 86 noise), 5–60s each, originally 96kHz/24-bit | Healthy / Unhealthy / Noise |

**To download:** open each Mendeley page, use the "Download All" button. No account is typically required for public datasets, but you may need to accept a terms-of-use click-through. Check the license shown on each page (Mendeley datasets are commonly CC BY 4.0, but confirm per-dataset) — and cite both datasets in your own writeup/repo README (author names above).

**Caveat:** the exact folder/file naming inside each zip isn't something I can verify without downloading it myself — after you unzip, `ls` the top level and adjust the glob patterns in 3.6/3.7 to match what's actually there rather than assuming the layout below is pixel-perfect.

### Expected raw layout (verify after unzip, adjust if it differs)
```
data/raw/smartears/
├── healthy/     (~2000 files)
├── sick/        (~2000 files)
└── none/        (~2000 files)

data/raw/poultry_vocalization/
├── healthy/     (139 files)
├── unhealthy/   (121 files)
└── noise/       (86 files)
```

---

## 3.3 Label mapping
Both datasets' folder names collapse cleanly into one shared 3-class schema (no key collisions — both map "healthy" to "healthy"):

```python
# ml/preprocessing/labels.py
LABEL_MAP = {
    "healthy": "healthy",
    "sick": "elevated_respiratory",
    "unhealthy": "elevated_respiratory",
    "none": "no_bird_sound",
    "noise": "no_bird_sound",
}
```
Decide now (matches Part 1.3 scope): keep the model binary for v1 by dropping `no_bird_sound` samples entirely during training, and instead catch "no clear bird audio" with a cheap signal-energy check before the clip ever reaches the CNN. That keeps Part 4's model simple.

```python
def has_bird_signal(y, energy_threshold=0.01):
    """Cheap pre-filter: rejects near-silent / non-bird recordings before spectrogram/CNN."""
    import numpy as np
    return np.sqrt(np.mean(y**2)) > energy_threshold
```

---

## 3.4 Windowing (the key design piece)

```python
# ml/preprocessing/audio_utils.py
import librosa
import numpy as np

TARGET_SR = 22050
WINDOW_SEC = 5
N_MELS = 128
HOP_LENGTH = 512
TARGET_FRAMES = int(np.ceil(WINDOW_SEC * TARGET_SR / HOP_LENGTH))  # ~216

def load_audio(file_path, target_sr=TARGET_SR):
    # librosa resamples automatically — source files at 96kHz become 22.05kHz here
    y, sr = librosa.load(file_path, sr=target_sr)
    y, _ = librosa.effects.trim(y, top_db=20)
    y = librosa.util.normalize(y)
    return y

def chunk_audio(y, sr=TARGET_SR, window_sec=WINDOW_SEC):
    """Split audio into non-overlapping windows; pad the final short window."""
    window = int(window_sec * sr)
    if len(y) <= window:
        return [np.pad(y, (0, max(0, window - len(y))))]
    chunks = [y[i:i + window] for i in range(0, len(y) - window + 1, window)]
    remainder = y[len(chunks) * window:]
    if len(remainder) > window * 0.3:  # keep a meaningful tail, drop tiny scraps
        chunks.append(np.pad(remainder, (0, window - len(remainder))))
    return chunks

def to_mel_spectrogram(y_chunk, sr=TARGET_SR, n_mels=N_MELS, hop_length=HOP_LENGTH):
    mel = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)

def fix_length(mel, target_frames=TARGET_FRAMES):
    """Librosa's frame count can drift by ±1 depending on exact input length — pin it."""
    if mel.shape[1] < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])))
    else:
        mel = mel[:, :target_frames]
    return mel
```

**Inference-time aggregation** (what Part 5's `/analyze` endpoint will call after getting one probability per window):
```python
def aggregate_window_predictions(window_probs, elevated_idx=1):
    """window_probs: list of softmax outputs, one per 5s window in the farmer's recording."""
    elevated = [p[elevated_idx] for p in window_probs]
    return {
        "max_prob": max(elevated),                                   # catches a single bad window
        "frac_flagged": sum(p > 0.5 for p in elevated) / len(elevated),
    }
```
Recommend Part 4/5 use `max_prob` as the primary risk-score input, not the mean — a mean would dilute a real cough that only appears in one of six windows, which is the wrong failure mode for a health screen (see Part 11 risk table: false negatives are the costly error).

---

## 3.5 Build script — training set (SmartEars)
```python
# ml/preprocessing/build_train_set.py
import os, glob, uuid, csv
from audio_utils import load_audio, chunk_audio, to_mel_spectrogram, fix_length
import numpy as np

RAW_DIRS = {
    "data/raw/smartears/healthy": "healthy",
    "data/raw/smartears/sick": "elevated_respiratory",
    # "none" intentionally excluded — handled by the pre-filter in 3.3, not the CNN
}
OUT_DIR = "data/spectrograms/train"
manifest = []

for folder, label in RAW_DIRS.items():
    for path in glob.glob(os.path.join(folder, "*.wav")):
        y = load_audio(path)
        for chunk in chunk_audio(y):
            mel = fix_length(to_mel_spectrogram(chunk))
            clip_id = str(uuid.uuid4())
            out_path = os.path.join(OUT_DIR, label, f"{clip_id}.npy")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, mel)
            manifest.append([clip_id, path, label, "smartears"])

os.makedirs("data/spectrograms", exist_ok=True)
with open("data/spectrograms/train_manifest.csv", "w", newline="") as f:
    csv.writer(f).writerows([["clip_id", "source_path", "label", "dataset"]] + manifest)

print(f"Wrote {len(manifest)} training windows.")
```

## 3.6 Build script — held-out test set (Poultry Vocalization Dataset)
Keep this dataset in its own manifest, and group windows by source file — you'll need that grouping in Part 4's evaluation to test the *aggregation* logic (3.4), not just raw per-window accuracy, since that's what actually mirrors how the deployed app will use a farmer's recording.

```python
# ml/preprocessing/build_test_set.py
import os, glob, uuid, csv
from audio_utils import load_audio, chunk_audio, to_mel_spectrogram, fix_length
import numpy as np

TEST_DIRS = {
    "data/raw/poultry_vocalization/healthy": "healthy",
    "data/raw/poultry_vocalization/unhealthy": "elevated_respiratory",
    "data/raw/poultry_vocalization/noise": "no_bird_sound",
}
OUT_DIR = "data/spectrograms/test"
manifest = []

for folder, label in TEST_DIRS.items():
    for path in glob.glob(os.path.join(folder, "*.wav")):
        file_id = str(uuid.uuid4())  # groups all windows from this one source file
        y = load_audio(path)
        for w_idx, chunk in enumerate(chunk_audio(y)):
            mel = fix_length(to_mel_spectrogram(chunk))
            out_path = os.path.join(OUT_DIR, label, f"{file_id}_{w_idx}.npy")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, mel)
            manifest.append([f"{file_id}_{w_idx}", file_id, path, label, "poultry_vocalization"])

with open("data/spectrograms/test_manifest.csv", "w", newline="") as f:
    csv.writer(f).writerows([["window_id", "file_id", "source_path", "label", "dataset"]] + manifest)

print(f"Wrote {len(manifest)} test windows from {len(set(r[1] for r in manifest))} source files.")
```

---

## 3.7 Augmentation (training set only — never augment the test set)
```python
# ml/preprocessing/augment.py
import numpy as np, librosa

def pitch_shift(y, sr, n_steps_range=(-2, 2)):
    n_steps = np.random.uniform(*n_steps_range)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def time_shift(y, shift_max=0.2):
    shift = int(len(y) * np.random.uniform(-shift_max, shift_max))
    return np.roll(y, shift)

def add_noise(y, noise_clip, snr_db_range=(5, 15)):
    """noise_clip: a loaded background-noise sample (fan/wind), same length or longer than y."""
    snr_db = np.random.uniform(*snr_db_range)
    noise = noise_clip[:len(y)]
    sig_power = np.mean(y**2)
    noise_power = np.mean(noise**2) + 1e-8
    scale = np.sqrt(sig_power / (10**(snr_db/10) * noise_power))
    return y + scale * noise

def spec_augment(mel, freq_mask=12, time_mask=20):
    mel = mel.copy()
    f0 = np.random.randint(0, mel.shape[0] - freq_mask)
    mel[f0:f0+freq_mask, :] = 0
    t0 = np.random.randint(0, max(1, mel.shape[1] - time_mask))
    mel[:, t0:t0+time_mask] = 0
    return mel
```
Apply `pitch_shift`/`time_shift`/`add_noise` on raw audio before spectrogram extraction; apply `spec_augment` on the spectrogram itself. For a hackathon build order, this is safe to skip for the first end-to-end pass and add once the baseline model is training — don't let augmentation block getting a working pipeline.

---

## 3.8 Class balance check
Run this after 3.5, before touching Part 4:
```python
import pandas as pd
df = pd.read_csv("data/spectrograms/train_manifest.csv")
print(df["label"].value_counts())
```
Feed this split into Part 4's `class_weight` argument if it's skewed — don't assume it's balanced just because SmartEars advertises equal per-class raw clip counts, since chunking can change that (some clips may produce more usable windows than others after silence trimming).

---

## 3.9 Verification checklist before moving to Part 4
- [ ] `train_manifest.csv` has both `healthy` and `elevated_respiratory` rows, roughly balanced (or you've computed class weights)
- [ ] Every saved `.npy` has identical shape `(128, 216)` — spot check with `np.load(path).shape` on a handful of files
- [ ] No NaN/Inf: `np.isnan(mel).any()` and `np.isinf(mel).any()` both `False` on a sample
- [ ] Load 2–3 raw clips and their processed spectrograms side by side (e.g. `librosa.display.specshow`) to sanity-check labels weren't scrambled during folder walking
- [ ] `test_manifest.csv` file_ids are correctly grouped (each source file's windows share a `file_id` prefix) — this is what Part 4's aggregation evaluation depends on
- [ ] Train and test data live in fully separate directories — no SmartEars leakage into the test manifest

---

## Handoff to Part 4
Part 4 (model training) should expect:
- Input tensor shape: `(128, 216, 1)` per window
- Two classes: `healthy`, `elevated_respiratory`
- Training data: `data/spectrograms/train/` + `train_manifest.csv`
- Held-out test data: `data/spectrograms/test/` + `test_manifest.csv`, evaluated **both** per-window and per-file (via `aggregate_window_predictions` from 3.4, grouped by `file_id`)

Say the word when you're ready to move to Part 4 and I'll build that one out the same way — standalone, grounded, no assumptions carried over that aren't written down here.
