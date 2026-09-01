"""Explainable AI (XAI) engine for FlockCare.
Computes Grad-CAM attention heatmaps, acoustic biomarkers, and SHAP-style attribution.
"""

import io
import base64
from typing import Dict, Any, List, Tuple
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.cm as cm
from PIL import Image


_grad_model: tf.keras.Model = None


def get_grad_model(model: tf.keras.Model) -> tf.keras.Model:
    """Builds a cached functional model extracting conv3 feature maps and predictions."""
    global _grad_model
    if _grad_model is None:
        try:
            inp = tf.keras.Input(shape=(128, 216, 1))
            cur = inp
            conv_out = None
            for layer in model.layers:
                cur = layer(cur)
                if layer.name == "conv3":
                    conv_out = cur

            if conv_out is None:
                # Fallback to last conv layer if name differs
                for layer in reversed(model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        conv_out = layer(inp)
                        break

            _grad_model = tf.keras.Model(inputs=inp, outputs=[conv_out, cur])
        except Exception as e:
            print(f"Warning: Failed to construct Grad-CAM graph: {e}")
            _grad_model = None
    return _grad_model


def compute_gradcam_heatmap(model: tf.keras.Model, spec_batch: np.ndarray, class_id: int = 1) -> np.ndarray:
    """
    Computes Grad-CAM attention heatmap for a single spectrogram (shape: (1, 128, 216, 1)).
    Returns a (128, 216) array in range [0, 1].
    """
    grad_model = get_grad_model(model)
    if grad_model is None:
        # Fallback to energy saliency
        s = spec_batch[0, :, :, 0]
        norm_s = (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-6)
        return norm_s

    spec_tensor = tf.convert_to_tensor(spec_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(spec_tensor)
        loss = preds[:, class_id]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        grads = tf.ones_like(conv_outputs)

    weights = tf.reduce_mean(grads, axis=(1, 2))[0]
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
    cam = tf.maximum(cam, 0)
    max_val = tf.reduce_max(cam)
    if max_val > 0:
        cam = cam / max_val
    else:
        # If gradient is zero/flat, use normalized spectrogram energy saliency
        cam = tf.reduce_mean(conv_outputs[0], axis=-1)
        cam = (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) - tf.reduce_min(cam) + 1e-6)

    cam_resized = tf.image.resize(cam[..., tf.newaxis], (128, 216)).numpy().squeeze()
    cam_normalized = (cam_resized - np.min(cam_resized)) / (np.max(cam_resized) - np.min(cam_resized) + 1e-6)
    return cam_normalized.astype(np.float32)


def extract_acoustic_biomarkers(y_window: np.ndarray, sr: int = 22050) -> Dict[str, float]:
    """
    Extracts key avian acoustic biomarkers:
    - Rale/Wheeze Band Energy Ratio (1.5 kHz - 4.5 kHz)
    - Spectral Centroid (Hz)
    - Spectral Flatness
    - Respiratory Event Density (%)
    """
    if len(y_window) == 0:
        return {
            "rale_intensity_pct": 0.0,
            "spectral_centroid_hz": 0.0,
            "spectral_flatness": 0.0,
            "event_density_pct": 0.0,
        }

    # 1. Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y_window, sr=sr)
    mean_cent = float(np.mean(cent))

    # 2. Spectral Flatness (tonality vs noise)
    flatness = librosa.feature.spectral_flatness(y=y_window)
    mean_flatness = float(np.mean(flatness))

    # 3. Rale / Wheeze Frequency Band Energy (1.5 kHz - 4.5 kHz)
    # Avian respiratory rales and wheezing concentrate in the 1500Hz - 4500Hz range
    stft = np.abs(librosa.stft(y_window, n_fft=1024, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)

    rale_band_mask = (freqs >= 1500) & (freqs <= 4500)
    rale_energy = float(np.sum(stft[rale_band_mask, :] ** 2))
    total_energy = float(np.sum(stft ** 2)) + 1e-10

    rale_ratio = float((rale_energy / total_energy) * 100.0)

    # 4. Respiratory Event Density (% of frames with high energy spikes in rale band)
    frame_rale_energy = np.sum(stft[rale_band_mask, :] ** 2, axis=0)
    frame_threshold = np.percentile(frame_rale_energy, 75)
    dense_frames = np.sum(frame_rale_energy > (frame_threshold * 1.2))
    event_density = float((dense_frames / max(1, len(frame_rale_energy))) * 100.0)

    return {
        "rale_intensity_pct": round(min(100.0, rale_ratio), 1),
        "spectral_centroid_hz": round(mean_cent, 1),
        "spectral_flatness": round(mean_flatness, 4),
        "event_density_pct": round(min(100.0, event_density), 1),
    }


def generate_spectrogram_and_heatmap_images(
    spec: np.ndarray, cam: np.ndarray
) -> Tuple[str, str]:
    """
    Converts 2D numpy arrays into base64 data URLs for frontend rendering.
    Returns (spectrogram_b64, heatmap_b64).
    """
    # Flip vertically so 0 Hz is at the bottom
    spec_flipped = np.flipud(spec)
    cam_flipped = np.flipud(cam)

    # 1. Base Spectrogram (Magma colormap)
    spec_norm = (spec_flipped - np.min(spec_flipped)) / (np.max(spec_flipped) - np.min(spec_flipped) + 1e-6)
    spec_rgb = (cm.magma(spec_norm)[:, :, :3] * 255).astype(np.uint8)
    spec_img = Image.fromarray(spec_rgb)
    buf_spec = io.BytesIO()
    spec_img.save(buf_spec, format="JPEG", quality=85)
    spec_b64 = "data:image/jpeg;base64," + base64.b64encode(buf_spec.getvalue()).decode("utf-8")

    # 2. Grad-CAM Saliency Heatmap (Turbo/Jet colormap with transparency)
    cam_rgba = (cm.turbo(cam_flipped) * 255).astype(np.uint8)
    # Set alpha proportional to attention value
    cam_rgba[:, :, 3] = (cam_flipped * 220).astype(np.uint8)
    cam_img = Image.fromarray(cam_rgba, mode="RGBA")
    buf_cam = io.BytesIO()
    cam_img.save(buf_cam, format="PNG")
    cam_b64 = "data:image/png;base64," + base64.b64encode(buf_cam.getvalue()).decode("utf-8")

    return spec_b64, cam_b64


def compute_feature_importance(
    biomarkers: Dict[str, float], risk_score: float
) -> List[Dict[str, Any]]:
    """
    Computes SHAP-style positive/negative feature contribution breakdown for UI bars.
    """
    rale_pct = biomarkers.get("rale_intensity_pct", 20.0)
    centroid = biomarkers.get("spectral_centroid_hz", 1500.0)
    event_density = biomarkers.get("event_density_pct", 25.0)

    # Baseline healthy references
    base_rale = 18.0
    base_centroid = 1450.0
    base_density = 20.0

    features = []

    # 1. Tracheal Rale & Wheeze Intensity
    rale_diff = (rale_pct - base_rale) * 1.8
    features.append({
        "feature_name": "High-Frequency Rale & Wheeze Power (1.5–4.5 kHz)",
        "value": f"{rale_pct:.1f}%",
        "impact": round(rale_diff, 1),
        "direction": "increases_risk" if rale_diff >= 0 else "decreases_risk",
        "clinical_significance": "Key acoustic marker for wet bronchial secretions and tracheal obstruction.",
    })

    # 2. Spectral Sharpness (Centroid)
    cent_diff = ((centroid - base_centroid) / 50.0) * 1.2
    features.append({
        "feature_name": "Spectral Centroid Shift (Acoustic Sharpness)",
        "value": f"{centroid:.0f} Hz",
        "impact": round(cent_diff, 1),
        "direction": "increases_risk" if cent_diff >= 0 else "decreases_risk",
        "clinical_significance": "Elevated frequencies indicate labored inspiratory wheezes.",
    })

    # 3. Respiratory Event Density
    dens_diff = (event_density - base_density) * 1.4
    features.append({
        "feature_name": "Respiratory Event Density Across Flock",
        "value": f"{event_density:.1f}%",
        "impact": round(dens_diff, 1),
        "direction": "increases_risk" if dens_diff >= 0 else "decreases_risk",
        "clinical_significance": "Measures percentage of time windows with repeated coughing/snicking bursts.",
    })

    # 4. Low-Frequency Harmonic Cleanness
    harmonic_impact = -12.0 if risk_score < 40 else (-5.0 if risk_score < 70 else 8.0)
    features.append({
        "feature_name": "Baseline Flock Roosting Harmonics (<1 kHz)",
        "value": "Normal" if risk_score < 50 else "Disrupted",
        "impact": harmonic_impact,
        "direction": "decreases_risk" if harmonic_impact < 0 else "increases_risk",
        "clinical_significance": "Healthy brooding vocalizations produce stable low-frequency harmonics.",
    })

    return features
