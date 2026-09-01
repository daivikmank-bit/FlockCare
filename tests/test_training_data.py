"""Unit tests for training data loading, train/val split, and class weights."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from ml.training.data import (
    load_split,
    get_train_val_split,
    get_class_weights,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
)


@pytest.fixture
def sample_spec_dataset():
    """Create a temporary spectrogram dataset with manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_dir = os.path.join(tmpdir, "spectrograms")
        os.makedirs(os.path.join(spec_dir, "healthy"), exist_ok=True)
        os.makedirs(os.path.join(spec_dir, "elevated_respiratory"), exist_ok=True)
        os.makedirs(os.path.join(spec_dir, "no_bird_sound"), exist_ok=True)

        rows = []
        # 6 healthy, 4 elevated_respiratory, 2 no_bird_sound (which should be filtered out)
        for i in range(6):
            clip_id = f"healthy_{i}"
            spec = np.random.randn(128, 216).astype(np.float32)
            np.save(os.path.join(spec_dir, "healthy", f"{clip_id}.npy"), spec)
            rows.append({
                "clip_id": clip_id,
                "source_path": f"dummy/healthy_{i}.wav",
                "label": "healthy",
                "dataset": "test_ds",
            })

        for i in range(4):
            clip_id = f"elevated_{i}"
            spec = np.random.randn(128, 216).astype(np.float32)
            np.save(os.path.join(spec_dir, "elevated_respiratory", f"{clip_id}.npy"), spec)
            rows.append({
                "clip_id": clip_id,
                "source_path": f"dummy/elevated_{i}.wav",
                "label": "elevated_respiratory",
                "dataset": "test_ds",
            })

        for i in range(2):
            clip_id = f"noise_{i}"
            spec = np.random.randn(128, 216).astype(np.float32)
            np.save(os.path.join(spec_dir, "no_bird_sound", f"{clip_id}.npy"), spec)
            rows.append({
                "clip_id": clip_id,
                "source_path": f"dummy/noise_{i}.wav",
                "label": "no_bird_sound",
                "dataset": "test_ds",
            })

        manifest_path = os.path.join(tmpdir, "manifest.csv")
        pd.DataFrame(rows).to_csv(manifest_path, index=False)

        yield manifest_path, spec_dir


def test_load_split(sample_spec_dataset):
    manifest_path, spec_dir = sample_spec_dataset
    X, y, df = load_split(manifest_path, spec_dir, id_col="clip_id")

    # 6 healthy + 4 elevated = 10 (2 noise filtered out)
    assert len(X) == 10
    assert len(y) == 10
    assert len(df) == 10
    assert X.shape == (10, 128, 216, 1)
    assert X.dtype == np.float32

    # Verify labels
    assert (y == 0).sum() == 6
    assert (y == 1).sum() == 4
    assert set(df["label"].unique()) == {"healthy", "elevated_respiratory"}


def test_load_split_missing_file_raises(sample_spec_dataset):
    manifest_path, spec_dir = sample_spec_dataset
    with pytest.raises(FileNotFoundError):
        load_split("non_existent_manifest.csv", spec_dir)


def test_get_train_val_split_standard():
    X = np.random.randn(20, 128, 216, 1).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10)

    X_train, X_val, y_train, y_val = get_train_val_split(X, y, test_size=0.2, random_state=42)

    assert len(X_train) == 16
    assert len(X_val) == 4
    assert len(y_train) == 16
    assert len(y_val) == 4
    # Check stratified balance in val split (2 of each class)
    assert (y_val == 0).sum() == 2
    assert (y_val == 1).sum() == 2


def test_get_train_val_split_grouped_prevents_leakage():
    # 4 distinct audio recordings:
    # file_0 (healthy, 3 windows), file_1 (healthy, 3 windows)
    # file_2 (elevated, 3 windows), file_3 (elevated, 3 windows)
    groups = np.array([
        "file_0", "file_0", "file_0",
        "file_1", "file_1", "file_1",
        "file_2", "file_2", "file_2",
        "file_3", "file_3", "file_3",
    ])
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    X = np.random.randn(12, 128, 216, 1).astype(np.float32)

    X_train, X_val, y_train, y_val = get_train_val_split(
        X, y, groups=groups, test_size=0.5, random_state=42
    )

    # 2 files in train (6 windows), 2 files in val (6 windows)
    assert len(X_train) == 6
    assert len(X_val) == 6
    assert (y_train == 0).sum() == 3
    assert (y_train == 1).sum() == 3
    assert (y_val == 0).sum() == 3
    assert (y_val == 1).sum() == 3

    # Check zero overlap in groups (no data leakage)
    train_groups = set(groups[np.isin(np.arange(12), [i for i, x in enumerate(X) if any(np.array_equal(x, xt) for xt in X_train)])])
    val_groups = set(groups[np.isin(np.arange(12), [i for i, x in enumerate(X) if any(np.array_equal(x, xv) for xv in X_val)])])
    assert len(train_groups.intersection(val_groups)) == 0


def test_get_class_weights():
    # Balanced
    y_balanced = np.array([0, 0, 1, 1])
    weights_bal = get_class_weights(y_balanced)
    assert weights_bal[0] == pytest.approx(1.0)
    assert weights_bal[1] == pytest.approx(1.0)

    # Imbalanced: 3 healthy (0) for every 1 elevated (1)
    y_imbalanced = np.array([0, 0, 0, 1])
    weights_imb = get_class_weights(y_imbalanced)
    # Elevated class should have higher weight
    assert weights_imb[1] > weights_imb[0]
    assert weights_imb[1] == pytest.approx(2.0)
    assert weights_imb[0] == pytest.approx(4 / 6)
