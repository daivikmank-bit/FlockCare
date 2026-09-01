"""Unit tests for Explainable AI (XAI) and Disease Differential features."""

import numpy as np
import pytest
import tensorflow as tf

from backend.app.explainability import (
    compute_gradcam_heatmap,
    extract_acoustic_biomarkers,
    generate_spectrogram_and_heatmap_images,
    compute_feature_importance,
)
from backend.app.disease_differential import generate_disease_differential
from backend.app.inference import get_model


def test_compute_gradcam_heatmap_dimensions():
    """Verify Grad-CAM returns a normalized (128, 216) 2D array in [0, 1]."""
    model = get_model()
    spec = np.random.uniform(-40, 20, (1, 128, 216, 1)).astype(np.float32)

    cam = compute_gradcam_heatmap(model, spec, class_id=1)

    assert cam.shape == (128, 216)
    assert np.min(cam) >= 0.0
    assert np.max(cam) <= 1.0


def test_extract_acoustic_biomarkers():
    """Verify biomarker calculation produces bounded, non-empty metrics."""
    sr = 22050
    t = np.linspace(0, 5.0, int(5.0 * sr))
    # Signal with 2500Hz wheeze tone
    y = (0.5 * np.sin(2 * np.pi * 2500 * t)).astype(np.float32)

    bm = extract_acoustic_biomarkers(y, sr=sr)

    assert "rale_intensity_pct" in bm
    assert "spectral_centroid_hz" in bm
    assert "spectral_flatness" in bm
    assert "event_density_pct" in bm
    assert bm["spectral_centroid_hz"] > 1000.0
    assert bm["rale_intensity_pct"] > 50.0  # Dominant 2500Hz tone in 1.5-4.5kHz band


def test_generate_spectrogram_and_heatmap_images():
    """Verify base64 data URLs for spectrogram and Grad-CAM overlay."""
    spec = np.random.uniform(-40, 20, (128, 216)).astype(np.float32)
    cam = np.random.uniform(0, 1, (128, 216)).astype(np.float32)

    spec_b64, cam_b64 = generate_spectrogram_and_heatmap_images(spec, cam)

    assert spec_b64.startswith("data:image/jpeg;base64,")
    assert cam_b64.startswith("data:image/png;base64,")


def test_disease_differential_healthy_flock():
    """Verify low risk flock produces healthy status and low likelihood across all diseases."""
    biomarkers = {
        "rale_intensity_pct": 12.0,
        "spectral_centroid_hz": 1200.0,
        "spectral_flatness": 0.005,
        "event_density_pct": 10.0,
    }

    diff = generate_disease_differential(risk_score=15.0, biomarkers=biomarkers)

    assert "Healthy" in diff["flock_clinical_status"]
    assert len(diff["differentials"]) >= 5
    assert all(d["likelihood"] == "Low" for d in diff["differentials"])


def test_disease_differential_elevated_risk():
    """Verify elevated risk flock identifies primary differentials (e.g. IBV / CRD) and checklists."""
    biomarkers = {
        "rale_intensity_pct": 65.0,
        "spectral_centroid_hz": 2400.0,
        "spectral_flatness": 0.02,
        "event_density_pct": 70.0,
    }

    diff = generate_disease_differential(risk_score=85.0, biomarkers=biomarkers)

    assert "Active Respiratory Distress" in diff["flock_clinical_status"]
    assert len(diff["differentials"]) >= 5
    top_disease = diff["differentials"][0]
    assert top_disease["likelihood"] in ("High", "Moderate")
    assert len(top_disease["key_symptoms"]) >= 3
    assert len(top_disease["biosecurity_actions"]) >= 3
    assert len(diff["overall_biosecurity_advice"]) >= 3


def test_feature_importance_attribution():
    """Verify SHAP-style attribution computes positive/negative factors."""
    biomarkers = {
        "rale_intensity_pct": 55.0,
        "spectral_centroid_hz": 2200.0,
        "spectral_flatness": 0.015,
        "event_density_pct": 60.0,
    }

    features = compute_feature_importance(biomarkers, risk_score=80.0)

    assert len(features) >= 3
    assert any(f["direction"] == "increases_risk" for f in features)
    assert any(f["impact"] > 0 for f in features)
