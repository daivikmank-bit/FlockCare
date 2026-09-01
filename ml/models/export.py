"""Model export utilities for Keras H5 and TFLite formats."""

import os
from typing import Optional
import tensorflow as tf


def export_keras_model(model: tf.keras.Model, output_path: str = "ml/saved_models/flockcare_cnn.h5") -> str:
    """
    Save the trained Keras model artifact for deployment.
    
    Args:
        model: Trained tf.keras.Model instance.
        output_path: Path where the model file should be saved (e.g. .h5 or .keras).
        
    Returns:
        Absolute or relative path to saved model file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    return output_path


def export_tflite_model(
    model: tf.keras.Model,
    output_path: str = "ml/saved_models/flockcare_cnn.tflite",
    quantize: bool = False,
) -> str:
    """
    Convert and export model to TensorFlow Lite format for lightweight on-device inference.
    
    Args:
        model: Trained tf.keras.Model instance.
        output_path: Path where the .tflite file will be written.
        quantize: Whether to apply standard dynamic range quantization.
        
    Returns:
        Path to the written .tflite file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    return output_path
