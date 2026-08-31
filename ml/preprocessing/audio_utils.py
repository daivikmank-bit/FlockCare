"""Audio loading, normalization, windowing, and mel-spectrogram transformation utilities."""

import os
from typing import Dict, List, Union
import librosa
import numpy as np

# Audio and spectrogram parameter constants
TARGET_SR: int = 22050
WINDOW_SEC: int = 5
N_MELS: int = 128
HOP_LENGTH: int = 512
TARGET_FRAMES: int = int(np.ceil(WINDOW_SEC * TARGET_SR / HOP_LENGTH))  # Exactly 216 frames


def load_audio(file_path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    Load an audio file, resample to target_sr, trim leading/trailing silence,
    and normalize peak amplitude.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # librosa resamples automatically (e.g. 96kHz -> 22.05kHz)
    y, _ = librosa.load(file_path, sr=target_sr)

    # Trim leading/trailing silence (< 20dB relative to peak)
    if len(y) > 0:
        y, _ = librosa.effects.trim(y, top_db=20)

    # Normalize amplitude
    if len(y) > 0 and np.max(np.abs(y)) > 0:
        y = librosa.util.normalize(y)
    elif len(y) == 0:
        y = np.zeros(int(WINDOW_SEC * target_sr), dtype=np.float32)

    return y.astype(np.float32)


def chunk_audio(
    y: np.ndarray,
    sr: int = TARGET_SR,
    window_sec: int = WINDOW_SEC,
    tail_threshold: float = 0.3,
) -> List[np.ndarray]:
    """
    Split audio into non-overlapping windows of window_sec seconds.
    If the recording is shorter than one window, pads to full window length.
    If a remainder exceeds tail_threshold (30% by default), pads it to a full window.
    Drops insignificant trailing scraps.
    """
    window = int(window_sec * sr)
    if len(y) <= window:
        padded = np.pad(y, (0, max(0, window - len(y))))
        return [padded.astype(np.float32)]

    chunks = [y[i : i + window] for i in range(0, len(y) - window + 1, window)]
    remainder = y[len(chunks) * window :]

    if len(remainder) > window * tail_threshold:
        padded_remainder = np.pad(remainder, (0, window - len(remainder)))
        chunks.append(padded_remainder)

    return [c.astype(np.float32) for c in chunks]


def to_mel_spectrogram(
    y_chunk: np.ndarray,
    sr: int = TARGET_SR,
    n_mels: int = N_MELS,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """
    Compute log Mel-spectrogram from audio chunk and normalize to [0, 1] range.
    Output shape: (n_mels, time_frames).
    """
    mel = librosa.feature.melspectrogram(
        y=y_chunk,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    denom = mel_db.max() - mel_db.min() + 1e-8
    mel_norm = (mel_db - mel_db.min()) / denom
    return mel_norm.astype(np.float32)


def fix_length(mel: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """
    Pin the time frame dimension to exactly target_frames (216)
    by zero-padding or truncating.
    """
    if mel.shape[1] < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])))
    elif mel.shape[1] > target_frames:
        mel = mel[:, :target_frames]
    return mel.astype(np.float32)


def process_audio_file(
    file_path: str,
    target_sr: int = TARGET_SR,
    window_sec: int = WINDOW_SEC,
    n_mels: int = N_MELS,
    hop_length: int = HOP_LENGTH,
    target_frames: int = TARGET_FRAMES,
) -> List[np.ndarray]:
    """
    Complete single-file preprocessing pipeline:
    1. Load, trim, normalize audio
    2. Split into 5-second chunks
    3. Generate normalized mel spectrogram for each chunk
    4. Pin frames to exactly target_frames
    Returns a list of 2D numpy arrays with shape (n_mels, target_frames), i.e., (128, 216).
    """
    y = load_audio(file_path, target_sr=target_sr)
    chunks = chunk_audio(y, sr=target_sr, window_sec=window_sec)
    spectrograms = []
    for chunk in chunks:
        mel = to_mel_spectrogram(chunk, sr=target_sr, n_mels=n_mels, hop_length=hop_length)
        mel_fixed = fix_length(mel, target_frames=target_frames)
        spectrograms.append(mel_fixed)
    return spectrograms


def aggregate_window_predictions(
    window_probs: Union[List[List[float]], np.ndarray],
    elevated_idx: int = 1,
) -> Dict[str, float]:
    """
    Aggregate per-window probabilities into a file/flock-level risk summary.
    Uses max_prob to avoid washing out single-window coughs/wheezes.
    """
    if len(window_probs) == 0:
        return {"max_prob": 0.0, "frac_flagged": 0.0, "mean_prob": 0.0}

    elevated = [p[elevated_idx] for p in window_probs]
    return {
        "max_prob": float(max(elevated)),
        "frac_flagged": float(sum(p > 0.5 for p in elevated) / len(elevated)),
        "mean_prob": float(sum(elevated) / len(elevated)),
    }
