"""Out-of-distribution (OOD) acoustic quality and domain gating module.

Uses the CNN's GlobalAveragePooling2D embedding representations to compute
Mahalanobis distance against the calibrated in-distribution reference.
Flags audio recordings whose acoustic characteristics deviate from calibrated conditions.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf


def get_gap_layer_name(trained_model: tf.keras.Model) -> str:
    """Finds the name of the GlobalAveragePooling2D layer in the model."""
    for layer in trained_model.layers:
        if isinstance(layer, (tf.keras.layers.GlobalAveragePooling2D, tf.keras.layers.GlobalAvgPool2D)):
            return layer.name
        if "gap" in layer.name.lower() or "global_average_pooling" in layer.name.lower():
            return layer.name
    # Fallback to layer index before classification head
    return trained_model.layers[-4].name


def build_embedding_model(trained_model: tf.keras.Model) -> tf.keras.Model:
    """
    Constructs a feature extractor model that outputs the GlobalAveragePooling2D
    embedding vector (e.g. 64-dimensional) from the trained CNN.
    """
    gap_layer_name = get_gap_layer_name(trained_model)
    gap_output = trained_model.get_layer(gap_layer_name).output
    inputs = trained_model.inputs if hasattr(trained_model, "inputs") and trained_model.inputs else trained_model.input
    embedding_model = tf.keras.Model(inputs=inputs, outputs=gap_output, name="cnn_embedder")
    return embedding_model


def fit_ood_reference(
    embedding_model: tf.keras.Model,
    X_train: np.ndarray,
    reg_epsilon: float = 1e-4,
) -> Dict[str, np.ndarray]:
    """
    Fits Gaussian reference distribution (mean and regularized pseudo-inverse covariance)
    on the in-distribution training data embeddings.
    """
    embeddings = embedding_model.predict(X_train, batch_size=64, verbose=0)
    if embeddings.ndim == 1:
        embeddings = embeddings[:, np.newaxis]

    mean = np.mean(embeddings, axis=0)
    cov = np.cov(embeddings, rowvar=False)

    if cov.ndim == 0:
        cov = np.array([[cov]])
    elif cov.ndim == 1:
        cov = np.diag(cov)

    with np.errstate(all="ignore"):
        # Regularize covariance to avoid singularity
        cov_reg = cov + reg_epsilon * np.eye(cov.shape[0])
        U, s, Vt = np.linalg.svd(cov_reg)
        s_inv = np.where(s > 1e-6, 1.0 / s, 0.0)
        cov_inv = np.dot(Vt.T * s_inv, U.T)

    return {"mean": mean.astype(np.float32), "cov_inv": cov_inv.astype(np.float32)}


def mahalanobis(
    embedding: np.ndarray,
    ref: Dict[str, np.ndarray],
) -> float:
    """
    Computes Mahalanobis distance from embedding vector to reference distribution:
    d_M = sqrt((x - mu)^T * Sigma^{-1} * (x - mu))
    """
    diff = np.asarray(embedding).flatten() - ref["mean"].flatten()
    with np.errstate(all="ignore"):
        dist_sq = float(np.dot(diff, np.dot(ref["cov_inv"], diff)))
    return float(np.sqrt(max(0.0, dist_sq)))


def calibrate_threshold(
    embedding_model: tf.keras.Model,
    ref: Dict[str, np.ndarray],
    X_val: np.ndarray,
    percentile: float = 99.0,
) -> float:
    """
    Calibrates OOD threshold at a target percentile (default 99th percentile)
    of in-distribution validation distance scores.
    """
    val_embeddings = embedding_model.predict(X_val, batch_size=64, verbose=0)
    scores = [mahalanobis(e, ref) for e in val_embeddings]
    threshold = float(np.percentile(scores, percentile))
    return threshold


def save_ood_reference(
    ref: Dict[str, np.ndarray],
    threshold: float,
    filepath: str = "ml/saved_models/ood_reference.npz",
) -> None:
    """Saves mean, cov_inv, and threshold to an .npz file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    np.savez(
        filepath,
        mean=ref["mean"],
        cov_inv=ref["cov_inv"],
        threshold=float(threshold),
    )


def load_ood_reference(
    filepath: str = "ml/saved_models/ood_reference.npz",
) -> Tuple[Dict[str, np.ndarray], float]:
    """Loads mean, cov_inv, and calibrated threshold from .npz file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"OOD reference file not found at: {filepath}")
    data = np.load(filepath, allow_pickle=True)
    ref = {"mean": data["mean"], "cov_inv": data["cov_inv"]}
    threshold = float(data["threshold"])
    return ref, threshold


def evaluate_window_ood(
    embedding_model: tf.keras.Model,
    ref: Dict[str, np.ndarray],
    threshold: float,
    window_spec: np.ndarray,
) -> Dict[str, Union[bool, float, str]]:
    """
    Evaluates a single spectrogram window or batch of windows for a recording.
    Conservative aggregation: if ANY window in the recording trips the threshold,
    the entire recording is flagged as 'out_of_range' to prevent silent false alarms.
    """
    if window_spec.ndim == 3:
        # Single window (128, 216, 1) -> (1, 128, 216, 1)
        window_spec = window_spec[np.newaxis, ...]

    embeddings = embedding_model.predict(window_spec, verbose=0)
    scores = [mahalanobis(e, ref) for e in embeddings]
    mean_score = float(np.mean(scores))
    max_score = float(np.max(scores))

    # Conservative rule: ANY window trips -> whole recording is out of range
    is_out_of_range = bool(max_score > threshold)

    status = "out_of_range" if is_out_of_range else "calibrated"
    message = (
        "Recording conditions differ from the calibrated setup — result may be unreliable."
        if is_out_of_range
        else "Audio characteristics match calibrated recording setup."
    )

    return {
        "is_out_of_range": is_out_of_range,
        "status": status,
        "mean_score": round(mean_score, 3),
        "max_score": round(max_score, 3),
        "threshold": round(threshold, 3),
        "message": message,
    }
