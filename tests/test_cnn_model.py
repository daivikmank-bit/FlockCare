"""Unit tests for CNN model architecture and inference."""

import numpy as np
import pytest
import tensorflow as tf

from ml.models.cnn_model import build_model, SparsePositiveRecall


def test_sparse_positive_recall_metric():
    metric = SparsePositiveRecall(class_id=1, name="recall")

    # True labels: [0, 1, 1, 0, 1] (3 actual positives)
    y_true = np.array([0, 1, 1, 0, 1])
    # Predictions: 2 true positives predicted as class 1, 1 false negative predicted as class 0
    y_pred = np.array([
        [0.9, 0.1],  # Pred 0, True 0 (TN)
        [0.2, 0.8],  # Pred 1, True 1 (TP)
        [0.7, 0.3],  # Pred 0, True 1 (FN)
        [0.8, 0.2],  # Pred 0, True 0 (TN)
        [0.1, 0.9],  # Pred 1, True 1 (TP)
    ])

    metric.update_state(y_true, y_pred)
    res = metric.result().numpy()
    assert res == pytest.approx(2.0 / 3.0)

    # Test reset
    metric.reset_state()
    assert metric.result().numpy() == pytest.approx(0.0)


def test_build_model_architecture():
    model = build_model(input_shape=(128, 216, 1), num_classes=2)

    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 128, 216, 1)
    assert model.output_shape == (None, 2)

    # Check key layer types
    layer_names = [layer.name for layer in model.layers]
    assert any("conv" in name for name in layer_names)
    assert any("bn" in name for name in layer_names)
    assert any("pool" in name for name in layer_names)
    assert any("gap" in name for name in layer_names)
    assert any("dropout" in name for name in layer_names)
    assert any("dense" in name or "output" in name for name in layer_names)


def test_model_forward_pass():
    model = build_model(input_shape=(128, 216, 1), num_classes=2)

    batch_size = 4
    dummy_input = np.random.randn(batch_size, 128, 216, 1).astype(np.float32)

    probs = model(dummy_input, training=False).numpy()

    assert probs.shape == (batch_size, 2)
    # Check that outputs are valid probabilities summing to ~1.0
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_model_loss_and_optimization():
    model = build_model(input_shape=(128, 216, 1), num_classes=2)

    X_dummy = np.random.randn(8, 128, 216, 1).astype(np.float32)
    y_dummy = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    history = model.fit(X_dummy, y_dummy, epochs=2, batch_size=4, verbose=0)
    assert "loss" in history.history
    assert len(history.history["loss"]) == 2
