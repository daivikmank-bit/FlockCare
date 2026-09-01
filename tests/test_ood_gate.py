"""Unit tests for Out-of-Distribution (OOD) gating module (ml/models/ood_gate.py)."""

import os
import numpy as np
import pytest
import tensorflow as tf

from ml.models.cnn_model import build_model
from ml.models.ood_gate import (
    build_embedding_model,
    calibrate_threshold,
    evaluate_window_ood,
    fit_ood_reference,
    load_ood_reference,
    mahalanobis,
    save_ood_reference,
)


@pytest.fixture
def mock_cnn_and_data():
    """Creates a mock CNN and synthetic in-distribution / out-of-distribution batches."""
    input_shape = (128, 216, 1)
    model = build_model(input_shape=input_shape, num_classes=2)

    # In-distribution data: centered around 0.5 with small noise
    np.random.seed(42)
    X_train = np.random.normal(loc=0.5, scale=0.1, size=(50, 128, 216, 1)).astype(np.float32)
    X_val = np.random.normal(loc=0.5, scale=0.1, size=(20, 128, 216, 1)).astype(np.float32)

    # Extreme OOD data: shifted mean and high variance
    X_ood = np.random.normal(loc=5.0, scale=2.0, size=(10, 128, 216, 1)).astype(np.float32)

    return model, X_train, X_val, X_ood


def test_build_embedding_model(mock_cnn_and_data):
    model, _, _, _ = mock_cnn_and_data
    emb_model = build_embedding_model(model)
    assert isinstance(emb_model, tf.keras.Model)
    assert emb_model.output_shape == (None, 64)


def test_fit_and_mahalanobis(mock_cnn_and_data):
    model, X_train, _, _ = mock_cnn_and_data
    emb_model = build_embedding_model(model)

    ref = fit_ood_reference(emb_model, X_train)
    assert "mean" in ref and "cov_inv" in ref
    assert ref["mean"].shape == (64,)
    assert ref["cov_inv"].shape == (64, 64)

    # Distance to exact mean should be 0
    zero_dist = mahalanobis(ref["mean"], ref)
    assert pytest.approx(zero_dist, abs=1e-4) == 0.0

    # Distance to perturbed embedding should be positive
    perturbed = ref["mean"] + np.ones(64) * 0.5
    pos_dist = mahalanobis(perturbed, ref)
    assert pos_dist > 0.0


def test_calibrate_and_evaluate_gate(mock_cnn_and_data):
    model, X_train, X_val, X_ood = mock_cnn_and_data
    emb_model = build_embedding_model(model)

    ref = fit_ood_reference(emb_model, X_train)
    threshold = calibrate_threshold(emb_model, ref, X_val, percentile=95.0)
    assert threshold > 0.0

    # In-distribution sample should pass
    in_dist_res = evaluate_window_ood(emb_model, ref, threshold, X_val[0])
    assert in_dist_res["status"] in ("calibrated", "out_of_range")
    assert in_dist_res["mean_score"] <= threshold * 1.5

    # Severely shifted OOD sample should be flagged out of range
    ood_res = evaluate_window_ood(emb_model, ref, threshold, X_ood[0])
    assert ood_res["is_out_of_range"] is True
    assert ood_res["status"] == "out_of_range"
    assert ood_res["mean_score"] > threshold


def test_save_and_load_ood_reference(tmp_path, mock_cnn_and_data):
    model, X_train, _, _ = mock_cnn_and_data
    emb_model = build_embedding_model(model)
    ref = fit_ood_reference(emb_model, X_train)
    threshold = 4.25

    save_path = os.path.join(tmp_path, "ood_ref.npz")
    save_ood_reference(ref, threshold, save_path)

    loaded_ref, loaded_thresh = load_ood_reference(save_path)
    np.testing.assert_allclose(ref["mean"], loaded_ref["mean"], rtol=1e-5)
    np.testing.assert_allclose(ref["cov_inv"], loaded_ref["cov_inv"], rtol=1e-5)
    assert pytest.approx(loaded_thresh, abs=1e-5) == threshold
