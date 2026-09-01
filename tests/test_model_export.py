"""Unit tests for model export to H5 and TFLite formats."""

import os
import tempfile
import numpy as np
import pytest
import tensorflow as tf

from ml.models.cnn_model import build_model
from ml.models.export import export_keras_model, export_tflite_model


def test_export_keras_model():
    model = build_model(input_shape=(128, 216, 1), num_classes=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "model.h5")
        saved_path = export_keras_model(model, h5_path)

        assert os.path.exists(saved_path)
        assert os.path.getsize(saved_path) > 0

        # Reload and compare predictions
        loaded_model = tf.keras.models.load_model(saved_path)
        dummy = np.random.randn(2, 128, 216, 1).astype(np.float32)

        pred_orig = model.predict(dummy, verbose=0)
        pred_loaded = loaded_model.predict(dummy, verbose=0)

        assert np.allclose(pred_orig, pred_loaded, atol=1e-5)


def test_export_tflite_model():
    model = build_model(input_shape=(128, 216, 1), num_classes=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        tflite_path = os.path.join(tmpdir, "model.tflite")
        saved_path = export_tflite_model(model, tflite_path)

        assert os.path.exists(saved_path)
        assert os.path.getsize(saved_path) > 0

        # Load TFLite interpreter and run inference
        interpreter = tf.lite.Interpreter(model_path=saved_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        dummy = np.random.randn(1, 128, 216, 1).astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]["index"])
        assert output_data.shape == (1, 2)
        assert np.allclose(output_data.sum(), 1.0, atol=1e-4)
