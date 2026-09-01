"""Inference engine for FlockCare audio classification and out-of-distribution gating."""

import io
import os
from typing import Any, Dict, Optional, Tuple
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf

from ml.models.ood_gate import (
    build_embedding_model,
    evaluate_window_ood,
    load_ood_reference,
)
from ml.models.risk import to_risk_label
from ml.preprocessing.audio_utils import (
    TARGET_SR,
    chunk_audio,
    fix_length,
    to_mel_spectrogram,
)
from .config import MIN_AUDIO_SECONDS, MODEL_PATH, OOD_REF_PATH
from .conversion import ConversionError, convert_to_wav_bytes

LABELS = ["healthy", "elevated_respiratory"]


class AudioTooShortError(Exception):
    """Raised when the audio recording duration is below MIN_AUDIO_SECONDS."""
    pass


class InsufficientSignalError(Exception):
    """Raised when the audio is silent or energy is too low to detect flock vocalizations."""
    pass


# Global model and embedder singletons (loaded once at import/startup time)
_model: Optional[tf.keras.Model] = None
_emb_model: Optional[tf.keras.Model] = None
_ood_ref: Optional[Dict[str, np.ndarray]] = None
_ood_threshold: Optional[float] = None


def get_model() -> tf.keras.Model:
    """Loads or returns the cached CNN model singleton."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model weight file not found: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def get_ood_gate() -> Tuple[Optional[tf.keras.Model], Optional[Dict[str, np.ndarray]], Optional[float]]:
    """Loads or returns cached embedding model and OOD reference."""
    global _emb_model, _ood_ref, _ood_threshold
    if _emb_model is None:
        model = get_model()
        _emb_model = build_embedding_model(model)

    if _ood_ref is None and os.path.exists(OOD_REF_PATH):
        try:
            _ood_ref, _ood_threshold = load_ood_reference(OOD_REF_PATH)
        except Exception as e:
            print(f"Warning: Failed to load OOD reference from {OOD_REF_PATH}: {e}")
            _ood_ref, _ood_threshold = None, None

    return _emb_model, _ood_ref, _ood_threshold


# Initialize models at module load time
try:
    get_model()
    get_ood_gate()
except Exception as _e:
    print(f"Model initialization note: {_e}")


def _load_audio_bytes(raw_bytes: bytes) -> np.ndarray:
    """
    Decodes raw audio bytes (WebM, Ogg, MP4, WAV), downmixes to mono,
    resamples to TARGET_SR, and trims silence.
    """
    wav_bytes = convert_to_wav_bytes(raw_bytes)  # ConversionError propagates to API handler
    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")

    if y.ndim > 1:
        y = y.mean(axis=1)  # downmix stereo to mono

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    return y.astype(np.float32)


def analyze_audio_bytes(raw_bytes: bytes) -> Dict[str, Any]:
    """
    Full inference pipeline:
    1. Decode audio to mono @ TARGET_SR
    2. Validate minimum duration threshold (>= 15s for multi-window safety)
    3. Window audio into 5-second chunks (>= 3 windows)
    4. Pre-filter acoustic energy on raw chunks (max-window RMS >= MIN_RMS_ENERGY)
    5. Peak normalize individual windows
    6. Compute mel-spectrograms
    7. Evaluate acoustic OOD gating with conservative ANY-window rule
    8. Predict elevated respiratory risk
    9. Map to clinical risk tier and return schema
    """
    from .config import MIN_RMS_ENERGY

    y = _load_audio_bytes(raw_bytes)

    # 1. Enforce minimum duration (15 seconds with slight tolerance for boundary trimming)
    min_samples = int((MIN_AUDIO_SECONDS - 1.0) * TARGET_SR)
    if len(y) < min_samples:
        raise AudioTooShortError(
            f"Recording too short ({len(y)/TARGET_SR:.1f}s) -- need at least {MIN_AUDIO_SECONDS}s of coop audio for reliable multi-window screening."
        )

    # 2. Chunk into 5s windows
    windows = chunk_audio(y)
    if len(windows) < 3:
        raise AudioTooShortError(
            f"Recording produced only {len(windows)} window(s) -- at least 3 acoustic windows (~15s) are required for reliable screening."
        )

    # 3. Per-window energy pre-filter: check if AT LEAST ONE window clears the vocalization energy floor
    window_rms_values = [float(np.sqrt(np.mean(w**2))) for w in windows]
    max_window_rms = max(window_rms_values) if window_rms_values else 0.0

    if max_window_rms < MIN_RMS_ENERGY or np.max(np.abs(y)) < 0.01:
        raise InsufficientSignalError(
            f"No flock vocalizations detected (peak window RMS {max_window_rms:.4f} < floor {MIN_RMS_ENERGY}). "
            "The recording appears silent or the phone was placed too far from the birds."
        )

    # 4. Peak normalize each window and extract mel spectrograms
    norm_windows = [librosa.util.normalize(w) if np.max(np.abs(w)) > 0 else w for w in windows]
    specs = np.stack([fix_length(to_mel_spectrogram(w)) for w in norm_windows])[..., np.newaxis]

    model = get_model()
    emb_model, ood_ref, ood_threshold = get_ood_gate()

    # OOD Gating check
    status = "calibrated"
    warning = None
    ood_score = None

    if emb_model is not None and ood_ref is not None and ood_threshold is not None:
        ood_res = evaluate_window_ood(emb_model, ood_ref, ood_threshold, specs)
        ood_score = float(ood_res["mean_score"])
        if ood_res["is_out_of_range"]:
            status = "out_of_range"
            warning = "Recording conditions differ from the calibrated setup — result may be less reliable."

    # Model inference
    probs = model.predict(specs, verbose=0)
    elevated_probs = probs[:, LABELS.index("elevated_respiratory")]
    max_prob = float(elevated_probs.max())

    result = to_risk_label(max_prob)
    result["status"] = status
    result["warning"] = warning
    result["ood_score"] = ood_score
    result["disclaimer"] = "This is a screening tool, not a diagnosis."
    result["windows_analyzed"] = len(windows)

    if status == "out_of_range":
        result["message"] = (
            f"{result['message']} (Caution: Acoustic characteristics suggest unfamiliar microphone or ambient noise)."
        )

    return result
