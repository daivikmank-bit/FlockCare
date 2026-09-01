"""Model training pipeline for FlockCare CNN."""

import os
from typing import Any, Dict, Optional, Tuple
import numpy as np
import tensorflow as tf
from sklearn.metrics import recall_score

from ml.models.cnn_model import build_model
from ml.preprocessing.augment import spec_augment
from ml.training.data import get_class_weights, get_train_val_split, load_split


class ValRecallCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to calculate validation recall specifically on class 1 (elevated_respiratory)
    at the end of each epoch, enabling EarlyStopping and ModelCheckpoint to monitor 'val_recall'.
    """

    def __init__(self, validation_data: Tuple[np.ndarray, np.ndarray]):
        super().__init__()
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if logs is None:
            logs = {}
        probs = self.model.predict(self.X_val, verbose=0)
        preds = np.argmax(probs, axis=1)

        # Calculate recall on elevated_respiratory (label 1)
        # Handle cases where class 1 might be absent or 0 predictions
        try:
            recall = recall_score(self.y_val, preds, pos_label=1, zero_division=0.0)
        except Exception:
            recall = 0.0

        logs["val_recall"] = float(recall)


def train_model(
    train_manifest: str = "data/spectrograms/train_manifest.csv",
    train_spec_dir: str = "data/spectrograms/train",
    model_save_dir: str = "ml/saved_models",
    epochs: int = 40,
    batch_size: int = 32,
    val_split_ratio: float = 0.2,
    learning_rate: float = 1e-3,
    random_state: int = 42,
    monitor_metric: str = "val_loss",
    patience: int = 8,
    use_augmentation: bool = True,
    verbose: int = 1,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Dict[str, Any]]:
    """
    Execute full training routine on SmartEars dataset with early stopping, model checkpointing,
    and optional SpecAugment data augmentation.
    
    Args:
        train_manifest: Path to train_manifest.csv.
        train_spec_dir: Directory with training spectrograms.
        model_save_dir: Directory where final and best weights will be stored.
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
        val_split_ratio: Fraction of data held out for validation (within SmartEars).
        learning_rate: Adam learning rate.
        random_state: Seed for reproducible train/val splitting.
        monitor_metric: Metric to monitor for early stopping ('val_recall' or 'val_loss').
        patience: Epochs of patience before early stopping.
        use_augmentation: Whether to augment training spectrograms with SpecAugment.
        verbose: Verbosity level for training output.
        
    Returns:
        (model, history, metadata)
    """
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "flockcare_cnn_best.h5")
    final_model_path = os.path.join(model_save_dir, "flockcare_cnn.h5")

    # 1. Load data
    print(f"Loading training data from {train_manifest}...")
    X_full, y_full, df_full = load_split(train_manifest, train_spec_dir, id_col="clip_id")
    print(f"Loaded {len(X_full)} samples with shape {X_full.shape}.")

    # 2. Train/val split (Strictly grouped by source recording / file_id to prevent data leakage)
    groups = None
    if "source_path" in df_full.columns:
        groups = df_full["source_path"].values
    elif "file_id" in df_full.columns:
        groups = df_full["file_id"].values

    X_train, X_val, y_train, y_val = get_train_val_split(
        X_full,
        y_full,
        groups=groups,
        test_size=val_split_ratio,
        random_state=random_state,
    )
    if groups is not None:
        print(f"Grouped train split: {len(X_train)} windows | Val split: {len(X_val)} windows (Zero file overlap).")
    else:
        print(f"Train split: {len(X_train)} samples | Val split: {len(X_val)} samples.")

    # 3. Optional SpecAugment on train set only (leaving val set pristine)
    if use_augmentation and len(X_train) > 0:
        print("Applying SpecAugment (Time & Frequency masking) to training set...")
        X_aug_list = []
        y_aug_list = []
        for x_sample, y_sample in zip(X_train, y_train):
            # Original sample
            X_aug_list.append(x_sample)
            y_aug_list.append(y_sample)
            # Augmented sample (strip channel dim for spec_augment, then re-add)
            mel_2d = x_sample[..., 0]
            mel_aug = spec_augment(mel_2d, freq_mask=12, time_mask=20, num_freq_masks=1, num_time_masks=1)
            X_aug_list.append(mel_aug[..., np.newaxis])
            y_aug_list.append(y_sample)

        X_train = np.stack(X_aug_list).astype(np.float32)
        y_train = np.array(y_aug_list, dtype=np.int32)
        print(f"Expanded training set with SpecAugment to {len(X_train)} samples.")

    # 4. Class weights
    class_weights = get_class_weights(y_train)
    print(f"Computed balanced class weights: {class_weights}")

    # 5. Build model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape=input_shape, num_classes=2, learning_rate=learning_rate)

    # 5. Setup callbacks
    mode = "max" if "recall" in monitor_metric or "acc" in monitor_metric else "min"
    val_recall_cb = ValRecallCallback(validation_data=(X_val, y_val))

    callbacks = [
        val_recall_cb,
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode=mode,
            patience=patience,
            restore_best_weights=True,
            verbose=verbose,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor=monitor_metric,
            mode=mode,
            save_best_only=True,
            verbose=verbose,
        ),
    ]

    # 6. Fit model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=min(batch_size, len(X_train)),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=verbose,
    )

    # 7. Save final model
    model.save(final_model_path)
    print(f"Saved final model to: {final_model_path}")
    print(f"Saved best checkpoint to: {best_model_path}")

    # Compute final validation metrics
    val_probs = model.predict(X_val, verbose=0)
    val_preds = np.argmax(val_probs, axis=1)
    val_recall_val = float(recall_score(y_val, val_preds, pos_label=1, zero_division=0.0))

    metadata = {
        "num_train_samples": len(X_train),
        "num_val_samples": len(X_val),
        "class_weights": class_weights,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "val_recall_elevated": val_recall_val,
        "epochs_trained": len(history.epoch),
    }

    return model, history, metadata


if __name__ == "__main__":
    train_model()
