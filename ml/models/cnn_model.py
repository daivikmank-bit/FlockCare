"""CNN model architecture for acoustic flock screening."""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


@tf.keras.utils.register_keras_serializable(package="flockcare")
class SparsePositiveRecall(tf.keras.metrics.Metric):
    """
    Computes Recall specifically on the positive disease class (class_id=1: 'elevated_respiratory')
    when labels are provided as sparse integer indices (0 or 1).
    
    Standard `tf.keras.metrics.Recall()` expects one-hot or binary probability targets and does not
    correctly compute class-specific recall out of the box with `sparse_categorical_crossentropy`
    and 2-unit softmax outputs. This custom metric extracts `argmax` predictions and calculates
    true positive rate on class 1.
    """

    def __init__(self, class_id: int = 1, name: str = "recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_id = class_id
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.possible_positives = self.add_weight(name="pos", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        preds = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

        is_true_pos_class = tf.equal(y_true, self.class_id)
        is_pred_pos_class = tf.equal(preds, self.class_id)
        true_pos = tf.logical_and(is_true_pos_class, is_pred_pos_class)

        if sample_weight is not None:
            sample_weight = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
            tp_count = tf.reduce_sum(tf.cast(true_pos, tf.float32) * sample_weight)
            pos_count = tf.reduce_sum(tf.cast(is_true_pos_class, tf.float32) * sample_weight)
        else:
            tp_count = tf.reduce_sum(tf.cast(true_pos, tf.float32))
            pos_count = tf.reduce_sum(tf.cast(is_true_pos_class, tf.float32))

        self.true_positives.assign_add(tp_count)
        self.possible_positives.assign_add(pos_count)

    def result(self):
        return tf.math.divide_no_nan(self.true_positives, self.possible_positives)

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.possible_positives.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"class_id": self.class_id})
        return config


def build_model(
    input_shape: Tuple[int, int, int] = (128, 216, 1),
    num_classes: int = 2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Build and compile a lightweight Convolutional Neural Network for mel-spectrogram classification.
    
    Architecture:
      - 3x [Conv2D(same) -> BatchNorm -> MaxPool2D(2)] with 16, 32, 64 filters.
      - GlobalAveragePooling2D to prevent overfitting and handle variable lengths gracefully.
      - Dense(64) with Dropout(0.4) for regularization.
      - Dense(num_classes, softmax) for classification probabilities.
    
    Args:
        input_shape: Shape of the input mel-spectrogram (n_mels, time_frames, channels).
        num_classes: Number of target classes (default: 2 -> healthy vs elevated_respiratory).
        learning_rate: Learning rate for Adam optimizer.
        
    Returns:
        Compiled tf.keras.Model.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="conv1"),
        layers.BatchNormalization(name="bn1"),
        layers.MaxPooling2D((2, 2), name="pool1"),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.MaxPooling2D((2, 2), name="pool2"),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv3"),
        layers.BatchNormalization(name="bn3"),
        layers.MaxPooling2D((2, 2), name="pool3"),

        layers.GlobalAveragePooling2D(name="gap"),
        layers.Dense(64, activation="relu", name="dense1"),
        layers.Dropout(0.4, name="dropout"),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="flockcare_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", SparsePositiveRecall(class_id=1, name="recall")],
    )
    return model
