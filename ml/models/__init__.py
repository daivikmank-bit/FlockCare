"""Models package for FlockCare."""

from ml.models.risk import to_risk_label

try:
    from ml.models.cnn_model import build_model
    from ml.models.export import export_keras_model, export_tflite_model
    __all__ = ["build_model", "to_risk_label", "export_keras_model", "export_tflite_model"]
except ImportError:
    __all__ = ["to_risk_label"]
