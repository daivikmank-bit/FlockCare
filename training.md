# FlockCare — Part 4: Model Training

*Standalone build guide. Assumes Part 3's output exists: `data/spectrograms/train/` + `train_manifest.csv`, `data/spectrograms/test/` + `test_manifest.csv`, each spectrogram shaped `(128, 216)`, 5-second windows, two classes (`healthy`, `elevated_respiratory`).*

**Goal of this part:** train the CNN, evaluate it honestly (per-window and per-file), benchmark it against the real public baseline, and export a model file Part 5's API can load.

---

## 4.0 Recap of what Part 3 hands you
- `.npy` spectrograms, shape `(128, 216)`, one per 5-second audio window
- `train_manifest.csv`: `clip_id, source_path, label, dataset` — all from SmartEars
- `test_manifest.csv`: `window_id, file_id, source_path, label, dataset` — all from Poultry Vocalization Dataset, windows grouped by `file_id` so you can reconstruct per-file predictions
- Two classes for the CNN: `healthy` (0), `elevated_respiratory` (1) — `no_bird_sound`/`noise` rows are excluded from training (handled by the pre-filter in Part 3.3)

---

## 4.1 Environment setup
```bash
pip install tensorflow scikit-learn pandas matplotlib
```
In Colab: **Runtime → Change runtime type → T4 GPU**, then confirm it's actually attached:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```
If that prints an empty list, training will still run — just much slower. Don't skip this check; it's a common silent time-sink.

---

## 4.2 Loading the data
With a few thousand 5s windows at `(128, 216)` float32, this comfortably fits in memory (Colab's default RAM is enough) — no need for a streaming pipeline.

```python
# ml/training/data.py
import pandas as pd
import numpy as np
import os

LABEL_TO_IDX = {"healthy": 0, "elevated_respiratory": 1}

def load_split(manifest_path, spec_dir, id_col="clip_id"):
    df = pd.read_csv(manifest_path)
    df = df[df["label"].isin(LABEL_TO_IDX)].reset_index(drop=True)  # drop no_bird_sound/noise rows
    X = np.stack([
        np.load(os.path.join(spec_dir, row.label, f"{getattr(row, id_col)}.npy"))
        for row in df.itertuples()
    ])
    y = df["label"].map(LABEL_TO_IDX).values
    return X[..., np.newaxis], y, df
```

```python
X_train_full, y_train_full, train_df = load_split(
    "data/spectrograms/train_manifest.csv", "data/spectrograms/train", id_col="clip_id"
)
```

---

## 4.3 Train/val split
Split within SmartEars only — the Poultry Vocalization Dataset stays fully held out for 4.7/4.8.
```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
)
```

## 4.4 Class weights
Don't assume balance just because SmartEars advertises equal per-class raw clip counts — chunking in Part 3 can shift that.
```python
from sklearn.utils.class_weight import compute_class_weight

weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(weights))
print(class_weight)
```

---

## 4.5 Model architecture
Matches the `(128, 216, 1)` input Part 3 produces:
```python
# ml/models/cnn_model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(128, 216, 1), num_classes=2):
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
`GlobalAveragePooling2D` instead of `Flatten` keeps parameter count small given a limited dataset, and makes the model robust to the odd off-by-one frame count Part 3's `fix_length` might still let through.

---

## 4.6 Training
```python
import tensorflow as tf
from ml.models.cnn_model import build_model

model = build_model()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_recall", mode="max", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "ml/saved_models/flockcare_cnn_best.h5", monitor="val_recall", mode="max", save_best_only=True
    ),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    class_weight=class_weight,
    callbacks=callbacks,
)
```
Monitoring `val_recall` (not `val_loss` or `val_accuracy`) matters here specifically because a missed sick bird is the costly error for a health-screening tool — optimize for catching it, not for the highest overall accuracy number.

---

## 4.7 Evaluation — per-window (on the held-out Poultry Vocalization Dataset)
```python
from sklearn.metrics import classification_report, confusion_matrix

X_test, y_test, test_df = load_split(
    "data/spectrograms/test_manifest.csv", "data/spectrograms/test", id_col="window_id"
)

probs = model.predict(X_test)
preds = probs.argmax(axis=1)

print(classification_report(y_test, preds, target_names=["healthy", "elevated_respiratory"]))
print(confusion_matrix(y_test, preds))
```
This is your out-of-distribution number — the Poultry Vocalization Dataset is a genuinely different recording setup than SmartEars, so this is a real generalization check, not a training-set echo.

## 4.8 Evaluation — per-file (aggregated) — the number that matters most
Per-window accuracy is a proxy. The number that actually reflects the deployed product is: given a farmer's recording sliced into windows, does the *aggregated* result get the flock right?

```python
test_df = test_df.copy()
test_df["healthy_prob"] = probs[:, 0]
test_df["elevated_prob"] = probs[:, 1]

file_level = test_df.groupby("file_id").agg(
    label=("label", "first"),
    max_elevated_prob=("elevated_prob", "max"),   # matches Part 3's aggregate_window_predictions
).reset_index()

file_level["pred"] = (file_level["max_elevated_prob"] > 0.5).astype(int)
y_file_true = file_level["label"].map({"healthy": 0, "elevated_respiratory": 1})

print(classification_report(y_file_true, file_level["pred"]))
print(confusion_matrix(y_file_true, file_level["pred"]))
```
Report **this** number as your headline result in the demo/writeup, not the raw per-window one — it's what a farmer would actually experience.

---

## 4.9 Benchmarking against the real Hugging Face baseline

This is `IceKhoffi/chicken-vocalization-classifier` — a PyTorch CNN trained on the same Poultry Vocalization Dataset, classifying Healthy / Noise / Unhealthy. Its preprocessing is different from yours (1.5s clips, not 5s), so give it its own path rather than feeding it your `.npy` files.

```bash
pip install torch huggingface_hub
```

**Reconstructed architecture** (layer shapes come straight from the model card — if these are wrong, loading the state dict will throw a shape-mismatch error immediately, which is a useful sanity check):
```python
# ml/evaluation/baseline_model.py
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

class BaselineChickenCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def load_baseline():
    model = BaselineChickenCNN(num_classes=3)
    weights_path = hf_hub_download(
        repo_id="IceKhoffi/chicken-vocalization-classifier",
        filename="Chiken_CNN_Disease_Detection_Model.pth",
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model
```

**Baseline's own preprocessing** (must match exactly — these are its documented training-time settings, not yours):
```python
import librosa
import numpy as np

BASELINE_SR = 22050
BASELINE_WAV_SIZE = int(1.5 * BASELINE_SR)
BASELINE_N_MELS = 128
BASELINE_N_FFT = 2648
BASELINE_HOP = 256

def baseline_preprocess(y):
    y = y[:BASELINE_WAV_SIZE]
    if len(y) < BASELINE_WAV_SIZE:
        y = np.pad(y, (0, BASELINE_WAV_SIZE - len(y)))
    mel = librosa.feature.melspectrogram(
        y=y, sr=BASELINE_SR, n_mels=BASELINE_N_MELS, n_fft=BASELINE_N_FFT, hop_length=BASELINE_HOP
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)
```

**Running it on the same held-out files:**
```python
import glob, torch

baseline = load_baseline()
BASELINE_IDX_TO_LABEL = {0: "healthy", 1: "noise", 2: "unhealthy"}  # confirm this ordering against the model's training notebook before trusting it

results = []
for label_folder, true_label in [("healthy", "healthy"), ("unhealthy", "elevated_respiratory")]:
    for path in glob.glob(f"data/raw/poultry_vocalization/{label_folder}/*.wav"):
        y, _ = librosa.load(path, sr=BASELINE_SR)
        mel = baseline_preprocess(y)
        x = torch.tensor(mel).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, T)
        with torch.no_grad():
            pred_idx = baseline(x).argmax(dim=1).item()
        results.append({"path": path, "true": true_label, "baseline_pred": BASELINE_IDX_TO_LABEL[pred_idx]})
```
Drop `noise`-predicted files before scoring (fair 2-class comparison against your binary model), then run the same `classification_report` as 4.8.

**One flag before you trust this section:** the class-index ordering (`0=healthy, 1=noise, 2=unhealthy`) is my best inference from the model card's stated training order, not something explicitly confirmed in a `label2id` mapping on the page. Verify it against the linked `CHBD_Vocalization_Analysis.ipynb` notebook in that repo before reporting numbers from it — an inverted mapping would silently flip your benchmark result.

**How to frame this in your writeup:** your positioning is "Access, Not Reinvention" — the point of this comparison is showing your deployment-optimized model holds up against an existing baseline, not claiming to beat a model that isn't solving the same accessibility problem.

---

## 4.10 Softmax → risk label
Carried over from the plan, now wired to real predictions:
```python
def to_risk_label(prob_elevated: float) -> dict:
    score = round(prob_elevated * 100, 1)
    if score >= 70:
        level, message = "high", "Elevated respiratory sounds detected — isolate the flock and consult a veterinarian."
    elif score >= 40:
        level, message = "moderate", "Some signs of respiratory stress. Monitor closely over the next 24–48 hours."
    else:
        level, message = "low", "Flock sounds healthy. No signs of respiratory distress detected."
    return {"risk_score": score, "risk_level": level, "message": message}
```
Treat 70/40 as a starting point — tune against the per-file results in 4.8, not the raw per-window numbers.

---

## 4.11 Export
```python
model.save("ml/saved_models/flockcare_cnn.h5")
```
Roadmap item, not required now — TFLite export for future on-device inference:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("ml/saved_models/flockcare_cnn.tflite", "wb") as f:
    f.write(tflite_model)
```

---

## 4.12 Verification checklist before moving to Part 5
- [ ] Val recall on `elevated_respiratory` reported explicitly, not just accuracy
- [ ] Both per-window (4.7) and per-file/aggregated (4.8) metrics reported — the per-file number is your headline result
- [ ] Confusion matrix included alongside precision/recall/F1
- [ ] Baseline comparison run on the same held-out files, with the class-index mapping verified against the source notebook (not assumed)
- [ ] Round-trip test: `tf.keras.models.load_model("ml/saved_models/flockcare_cnn.h5")` reproduces identical predictions on a spot-check batch
- [ ] `flockcare_cnn.h5` exists somewhere Part 5's backend can actually read it from

---

## Handoff to Part 5
- **Model file:** `ml/saved_models/flockcare_cnn.h5`
- **Input:** `(128, 216, 1)` mel-spectrogram per 5-second window — same `load_audio` / `chunk_audio` / `to_mel_spectrogram` / `fix_length` functions from Part 3
- **Output:** softmax over `[healthy, elevated_respiratory]`
- **Inference flow the API must implement:** farmer's ~30s clip → `chunk_audio` → spectrogram each window → `model.predict` per window → `aggregate_window_predictions` (max prob, from Part 3.4) → `to_risk_label` (4.10)

Say the word when you're ready for Part 5 (FastAPI backend) and I'll build that one the same way.
