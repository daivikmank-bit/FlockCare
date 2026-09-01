"""Inference pipeline for audio preprocessing, model prediction, and OOD gating.
Supports ultra-lightweight LiteRT (TFLite) and standard Keras models.
"""

import io
import os
from typing import Dict, Any, Optional, Tuple, List
import librosa
import numpy as np
import soundfile as sf

from ml.preprocessing.audio_utils import chunk_audio, to_mel_spectrogram, fix_length
from ml.models.risk import to_risk_label
from .config import MODEL_PATH, TFLITE_PATH, OOD_REF_PATH, MIN_AUDIO_SECONDS
from .conversion import convert_to_wav_bytes, ConversionError
from .explainability import (
    compute_gradcam_heatmap,
    compute_batch_gradcam_heatmaps,
    extract_acoustic_biomarkers,
    generate_spectrogram_and_heatmap_images,
    compute_feature_importance,
)
from .disease_differential import generate_disease_differential

TARGET_SR = 22050
LABELS = ["healthy", "elevated_respiratory"]


class AudioTooShortError(Exception):
    """Raised when the audio recording duration is below MIN_AUDIO_SECONDS."""
    pass


class InsufficientSignalError(Exception):
    """Raised when the audio is silent or energy is too low to detect flock vocalizations."""
    pass


# Global singletons
_interpreter = None
_input_details = None
_output_details = None
_model = None
_emb_model = None
_ood_ref: Optional[Dict[str, np.ndarray]] = None
_ood_threshold: Optional[float] = None


def get_interpreter():
    """Loads or returns cached LiteRT (TFLite) interpreter singleton."""
    global _interpreter, _input_details, _output_details
    if _interpreter is None and os.path.exists(TFLITE_PATH):
        try:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            _interpreter = Interpreter(model_path=TFLITE_PATH)
            _interpreter.allocate_tensors()
            _input_details = _interpreter.get_input_details()
            _output_details = _interpreter.get_output_details()
        except Exception as e:
            print(f"Warning: Failed to load LiteRT interpreter: {e}")
            _interpreter = None
    return _interpreter


def get_model():
    """Loads or returns cached Keras model singleton if TensorFlow is installed."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            return None
        try:
            import tensorflow as tf
            _model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"Keras model load note: {e}")
            _model = None
    return _model


def get_ood_gate():
    """Loads or returns cached embedding model and OOD reference."""
    global _emb_model, _ood_ref, _ood_threshold
    if _emb_model is None:
        model = get_model()
        if model is not None:
            try:
                from ml.models.ood_gate import build_embedding_model
                _emb_model = build_embedding_model(model)
            except Exception as e:
                print(f"Embedding model build note: {e}")
                _emb_model = None

    if _ood_ref is None and os.path.exists(OOD_REF_PATH):
        try:
            from ml.models.ood_gate import load_ood_reference
            _ood_ref, _ood_threshold = load_ood_reference(OOD_REF_PATH)
        except Exception as e:
            print(f"Warning: Failed to load OOD reference from {OOD_REF_PATH}: {e}")
            _ood_ref, _ood_threshold = None, None

    return _emb_model, _ood_ref, _ood_threshold


# Pre-warm models on module load
try:
    get_interpreter()
    get_model()
    get_ood_gate()
except Exception as _e:
    print(f"Model initialization note: {_e}")


def _load_audio_bytes(raw_bytes: bytes) -> np.ndarray:
    """
    Decodes raw audio bytes (WebM, Ogg, MP4, WAV), downmixes to mono,
    resamples to TARGET_SR, and returns raw float32 audio.
    """
    wav_bytes = convert_to_wav_bytes(raw_bytes)
    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")

    if y.ndim > 1:
        y = y.mean(axis=1)  # downmix stereo to mono

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_lq")

    return y.astype(np.float32)


def analyze_audio_bytes(raw_bytes: bytes) -> Dict[str, Any]:
    """
    Full inference & Explainable AI pipeline:
    1. Decode audio to mono @ TARGET_SR
    2. Validate minimum duration threshold (>= 15s for multi-window safety)
    3. Window audio into 5-second chunks (>= 3 windows)
    4. Pre-filter acoustic energy on raw chunks (max-window RMS >= MIN_RMS_ENERGY)
    5. Peak normalize individual windows & compute mel-spectrograms
    6. Evaluate acoustic OOD gating with conservative ANY-window rule
    7. Predict elevated respiratory risk per window and aggregated
    8. Compute Grad-CAM saliency heatmaps & acoustic biomarkers per window
    9. Run Avian Disease Differential matching engine
    10. Map to clinical risk tier and return rich Explainable AI response
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
    specs_2d = [fix_length(to_mel_spectrogram(w)) for w in norm_windows]
    specs = np.stack(specs_2d)[..., np.newaxis]

    interpreter = get_interpreter()
    model = get_model()
    emb_model, ood_ref, ood_threshold = get_ood_gate()

    # 5. OOD Gating check
    status = "calibrated"
    warning = None
    ood_score = None
    window_ood_scores = [0.0] * len(windows)
    window_is_ood = [False] * len(windows)

    if emb_model is not None and ood_ref is not None and ood_threshold is not None:
        try:
            from ml.models.ood_gate import evaluate_window_ood
            ood_res = evaluate_window_ood(emb_model, ood_ref, ood_threshold, specs)
            ood_score = float(ood_res["mean_score"])
            window_ood_scores = [float(s) for s in ood_res["scores"]]
            window_is_ood = [bool(s > ood_threshold) for s in ood_res["scores"]]
            if ood_res["is_out_of_range"]:
                status = "out_of_range"
                warning = "Recording conditions differ from the calibrated setup — result may be less reliable."
        except Exception as e:
            print(f"OOD gating evaluation note: {e}")

    # 6. Model inference (LiteRT fast path or Keras fallback)
    if interpreter is not None:
        probs_list = []
        for i in range(len(specs)):
            interpreter.set_tensor(_input_details[0]["index"], specs[i:i+1])
            interpreter.invoke()
            out = interpreter.get_tensor(_output_details[0]["index"])
            probs_list.append(out[0])
        probs = np.array(probs_list)
    elif model is not None:
        probs = model.predict(specs, verbose=0)
    else:
        raise RuntimeError("No inference engine available (LiteRT or Keras model not loaded).")

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

    # 7. Explainable AI: Fast Batched Grad-CAM heatmaps & Single-Pass Biomarkers
    cam_heatmaps = compute_batch_gradcam_heatmaps(model, specs, class_id=1)

    windows_detail = []
    flock_biomarkers_accum = {
        "rale_intensity_pct": [],
        "spectral_centroid_hz": [],
        "spectral_flatness": [],
        "event_density_pct": [],
    }

    for idx, (w_audio, spec_2d, cam_heatmap, w_prob, w_ood, w_is_ood_flag) in enumerate(
        zip(norm_windows, specs_2d, cam_heatmaps, elevated_probs, window_ood_scores, window_is_ood)
    ):
        spec_b64, cam_b64 = generate_spectrogram_and_heatmap_images(spec_2d, cam_heatmap)

        # Biomarkers
        bm = extract_acoustic_biomarkers(w_audio, sr=TARGET_SR)
        for k in flock_biomarkers_accum:
            flock_biomarkers_accum[k].append(bm[k])

        windows_detail.append({
            "window_index": idx,
            "start_sec": round(idx * 5.0, 1),
            "end_sec": round((idx + 1) * 5.0, 1),
            "risk_score": round(float(w_prob) * 100.0, 1),
            "ood_score": round(float(w_ood), 3),
            "is_ood": w_is_ood_flag,
            "spectrogram_image": spec_b64,
            "heatmap_image": cam_b64,
            "biomarkers": bm,
        })

    # 8. Flock-level aggregated biomarkers
    overall_biomarkers = {
        "rale_intensity_pct": round(float(np.mean(flock_biomarkers_accum["rale_intensity_pct"])), 1),
        "spectral_centroid_hz": round(float(np.mean(flock_biomarkers_accum["spectral_centroid_hz"])), 1),
        "spectral_flatness": round(float(np.mean(flock_biomarkers_accum["spectral_flatness"])), 4),
        "event_density_pct": round(float(np.mean(flock_biomarkers_accum["event_density_pct"])), 1),
    }

    # 9. SHAP-style Feature Importance
    feature_importance = compute_feature_importance(overall_biomarkers, result["risk_score"])

    # 10. Expected Avian Disease Differential Diagnosis
    disease_diff = generate_disease_differential(result["risk_score"], overall_biomarkers, status)

    result["windows_detail"] = windows_detail
    result["overall_biomarkers"] = overall_biomarkers
    result["feature_importance"] = feature_importance
    result["disease_differential"] = disease_diff

    return result
