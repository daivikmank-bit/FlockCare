"""PyTorch baseline model benchmark (IceKhoffi/chicken-vocalization-classifier)."""

import glob
import os
from typing import Any, Dict, List, Optional
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn


BASELINE_SR = 22050
BASELINE_WAV_SIZE = int(1.5 * BASELINE_SR)
BASELINE_N_MELS = 128
BASELINE_N_FFT = 2648
BASELINE_HOP = 256

# Verified against model logits / training notebook: 0=unhealthy, 1=noise, 2=healthy
BASELINE_IDX_TO_LABEL = {0: "unhealthy", 1: "noise", 2: "healthy"}


class BaselineChickenCNN(nn.Module):
    """Reconstructed PyTorch CNN architecture matching IceKhoffi/chicken-vocalization-classifier."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        out = self.classifier(feat)
        return out


def baseline_preprocess(y: np.ndarray) -> np.ndarray:
    """
    Preprocess raw audio according to the baseline model's 1.5s mel-spectrogram parameters.
    
    Args:
        y: Raw 1D audio waveform array at 22050 Hz.
        
    Returns:
        mel_db: Float32 mel-spectrogram of shape (128, ~130).
    """
    y = y[:BASELINE_WAV_SIZE]
    if len(y) < BASELINE_WAV_SIZE:
        y = np.pad(y, (0, BASELINE_WAV_SIZE - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=BASELINE_SR,
        n_mels=BASELINE_N_MELS,
        n_fft=BASELINE_N_FFT,
        hop_length=BASELINE_HOP,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def load_baseline(
    repo_id: str = "IceKhoffi/chicken-vocalization-classifier",
    filename: str = "Chicken_CNN_Disease_Detection_Model.pth",
    device: str = "cpu",
) -> BaselineChickenCNN:
    """
    Download (or load cached) weights from Hugging Face and instantiate the baseline PyTorch model.
    Falls back gracefully to an unweighted model if offline.
    """
    model = BaselineChickenCNN(num_classes=3)
    try:
        from huggingface_hub import hf_hub_download
        try:
            weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception:
            # Fallback to alternate typo spelling if any
            weights_path = hf_hub_download(repo_id=repo_id, filename="Chiken_CNN_Disease_Detection_Model.pth")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Successfully loaded baseline weights from {weights_path}")
    except Exception as e:
        print(f"Note: Could not download Hugging Face baseline weights ({e}). Using initialized baseline model structure.")

    model.to(device)
    model.eval()
    return model


def run_baseline_benchmark(
    baseline_model: Optional[BaselineChickenCNN] = None,
    raw_poultry_dir: str = "data/raw/poultry_vocalization",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run baseline model evaluation across raw Poultry Vocalization Dataset WAV files.
    Filters out 'noise' predictions for fair comparison against the binary FlockCare CNN.
    
    Args:
        baseline_model: Optional pre-loaded BaselineChickenCNN model.
        raw_poultry_dir: Root directory of poultry vocalization raw dataset.
        device: 'cpu' or 'cuda'/'mps'.
        
    Returns:
        Dictionary with benchmark results, classification report, and file predictions.
    """
    if baseline_model is None:
        baseline_model = load_baseline(device=device)

    baseline_model.eval()

    results: List[Dict[str, Any]] = []
    folder_mapping = [
        ("healthy", "healthy"),
        ("unhealthy", "elevated_respiratory"),
    ]

    for folder_name, true_label in folder_mapping:
        pattern = os.path.join(raw_poultry_dir, folder_name, "*.wav")
        file_paths = sorted(glob.glob(pattern))

        for path in file_paths:
            try:
                y, _ = librosa.load(path, sr=BASELINE_SR)
                mel = baseline_preprocess(y)
                # Shape: (1, 1, 128, T)
                x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = baseline_model(x)
                    pred_idx = logits.argmax(dim=1).item()

                pred_label = BASELINE_IDX_TO_LABEL.get(pred_idx, "unknown")
                results.append({
                    "path": path,
                    "filename": os.path.basename(path),
                    "true_label": true_label,
                    "baseline_raw_pred": pred_label,
                    "baseline_pred_idx": pred_idx,
                })
            except Exception as ex:
                print(f"Error processing {path} for baseline: {ex}")

    df_results = pd.DataFrame(results)
    if len(df_results) == 0:
        return {
            "status": "no_files_found",
            "message": f"No WAV files found in {raw_poultry_dir}",
            "df_results": df_results,
        }

    # Map baseline unhealthy -> elevated_respiratory
    df_results["baseline_mapped_pred"] = df_results["baseline_raw_pred"].replace({
        "unhealthy": "elevated_respiratory"
    })

    # Drop noise predictions for fair 2-class comparison
    valid_df = df_results[df_results["baseline_mapped_pred"].isin(["healthy", "elevated_respiratory"])].copy()

    if len(valid_df) > 0 and len(valid_df["true_label"].unique()) > 1:
        label_to_int = {"healthy": 0, "elevated_respiratory": 1}
        y_true = valid_df["true_label"].map(label_to_int).values
        y_pred = valid_df["baseline_mapped_pred"].map(label_to_int).values

        report_text = classification_report(
            y_true,
            y_pred,
            target_names=["healthy", "elevated_respiratory"],
            zero_division=0.0,
        )
        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=["healthy", "elevated_respiratory"],
            output_dict=True,
            zero_division=0.0,
        )
        cm = confusion_matrix(y_true, y_pred).tolist()
    else:
        report_text = "Insufficient class variation in valid predictions."
        report_dict = {}
        cm = []

    return {
        "status": "success",
        "total_files_evaluated": len(df_results),
        "valid_files_evaluated": len(valid_df),
        "noise_predictions_dropped": len(df_results) - len(valid_df),
        "classification_report_text": report_text,
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "results_df": df_results,
    }
