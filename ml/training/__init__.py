"""Training package for FlockCare CNN model."""

from ml.training.data import load_split, get_train_val_split, get_class_weights, LABEL_TO_IDX, IDX_TO_LABEL
from ml.training.train import train_model

__all__ = [
    "load_split",
    "get_train_val_split",
    "get_class_weights",
    "train_model",
    "LABEL_TO_IDX",
    "IDX_TO_LABEL",
]
