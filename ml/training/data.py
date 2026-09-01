"""Dataset loading, splitting, and class weight utilities for spectrogram data."""

import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

LABEL_TO_IDX = {"healthy": 0, "elevated_respiratory": 1}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


def load_split(
    manifest_path: str,
    spec_dir: str,
    id_col: str = "clip_id",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load spectrograms from manifest CSV and return feature array, label array, and cleaned DataFrame.
    
    Filters out non-binary classes (e.g. 'no_bird_sound' / 'noise') so the model trains strictly on
    binary acoustic disease screening ('healthy' vs 'elevated_respiratory').
    
    Args:
        manifest_path: Path to train_manifest.csv or test_manifest.csv.
        spec_dir: Root directory containing spectrograms organised by label subdirectories.
        id_col: Column name identifying the npy file name (e.g. 'clip_id' for train, 'window_id' for test).
        
    Returns:
        X: Spectrogram array with shape (N, 128, 216, 1) and float32 dtype.
        y: Integer labels array with shape (N,).
        df: Filtered pandas DataFrame matching the rows in X and y.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found at: {manifest_path}")

    df = pd.read_csv(manifest_path)
    df = df[df["label"].isin(LABEL_TO_IDX)].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"No valid examples found in {manifest_path} with labels in {list(LABEL_TO_IDX.keys())}")

    specs = []
    for row in df.itertuples():
        spec_id = getattr(row, id_col)
        spec_file = os.path.join(spec_dir, row.label, f"{spec_id}.npy")
        if not os.path.exists(spec_file):
            raise FileNotFoundError(f"Spectrogram file missing: {spec_file}")
        spec = np.load(spec_file).astype(np.float32)
        specs.append(spec)

    X = np.stack(specs)
    # Add channel dimension: (N, 128, 216) -> (N, 128, 216, 1)
    X = X[..., np.newaxis]
    y = df["label"].map(LABEL_TO_IDX).values.astype(np.int32)

    return X, y, df


def get_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split training data into train and validation splits.
    
    CRITICAL FOR ACOUSTIC SCREENING:
    If `groups` (e.g. source audio file paths or recording IDs) is provided, the split is performed
    strictly at the recording/group level, stratified by each recording's label.
    This guarantees that all 5-second windows from a given raw audio clip land exclusively in either
    train OR val, eliminating data leakage from highly correlated adjacent windows.
    
    Args:
        X: Feature tensor of shape (N, 128, 216, 1).
        y: Target label array of shape (N,).
        groups: Optional 1D array of length N identifying the source recording/file for each window.
        test_size: Proportion of the dataset (or unique files) to hold out for validation (default: 0.2).
        random_state: Random seed for reproducibility.
        
    Returns:
        X_train, X_val, y_train, y_val.
    """
    if groups is not None and len(np.unique(groups)) < len(groups):
        # Grouped stratified split by recording / source_path
        groups = np.asarray(groups)
        df_groups = pd.DataFrame({"group": groups, "label": y})
        # Group-level label (first label per recording)
        group_summary = df_groups.groupby("group").agg({"label": "first"}).reset_index()
        
        unique_groups = group_summary["group"].values
        unique_labels = group_summary["label"].values

        unique_classes, counts = np.unique(unique_labels, return_counts=True)
        stratify = unique_labels if np.min(counts) >= 2 else None

        train_groups, val_groups = train_test_split(
            unique_groups,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state,
        )

        train_mask = np.isin(groups, train_groups)
        val_mask = np.isin(groups, val_groups)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        return X_train, X_val, y_train, y_val

    # Standard stratified split when no grouping is provided or each sample is its own group
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = np.min(counts)
    stratify = y if min_count >= 2 else None

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )
    return X_train, X_val, y_train, y_val


def get_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights for training to handle potential class imbalance.
    
    Args:
        y_train: 1D array of class labels.
        
    Returns:
        Dictionary mapping class index to balanced sample weight.
    """
    unique_classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y_train,
    )
    class_weights_dict = {int(cls): float(w) for cls, w in zip(unique_classes, weights)}
    return class_weights_dict
