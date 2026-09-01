"""Unit tests for evaluation metrics, per-window reporting, and per-file aggregation."""

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.evaluate import evaluate_per_window, evaluate_per_file
from ml.models.cnn_model import build_model


def test_evaluate_per_window():
    model = build_model(input_shape=(128, 216, 1), num_classes=2)
    X_test = np.random.randn(8, 128, 216, 1).astype(np.float32)
    y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    results = evaluate_per_window(model, X_test, y_test)

    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    assert "confusion_matrix" in results
    assert "classification_report_text" in results
    assert results["probs"].shape == (8, 2)
    assert len(results["preds"]) == 8


def test_evaluate_per_file_aggregation():
    # Construct a test DataFrame with 2 files (each having 3 windows)
    # File 1 (healthy): window elevated probs = [0.1, 0.2, 0.3] -> max 0.3 <= 0.5 -> Pred: 0 (healthy) -> Correct
    # File 2 (elevated_respiratory): window elevated probs = [0.2, 0.8, 0.4] -> max 0.8 > 0.5 -> Pred: 1 (elevated) -> Correct
    df = pd.DataFrame([
        {"file_id": "file_1", "label": "healthy", "window_id": "w1"},
        {"file_id": "file_1", "label": "healthy", "window_id": "w2"},
        {"file_id": "file_1", "label": "healthy", "window_id": "w3"},
        {"file_id": "file_2", "label": "elevated_respiratory", "window_id": "w4"},
        {"file_id": "file_2", "label": "elevated_respiratory", "window_id": "w5"},
        {"file_id": "file_2", "label": "elevated_respiratory", "window_id": "w6"},
    ])

    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.7, 0.3],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.6, 0.4],
    ])

    res = evaluate_per_file(df, probs, threshold=0.5)

    assert res["num_files"] == 2
    assert res["accuracy"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0
    assert res["confusion_matrix"] == [[1, 0], [0, 1]]

    # Check file-level aggregation details
    file_df = res["file_level_df"]
    f1_row = file_df[file_df["file_id"] == "file_1"].iloc[0]
    f2_row = file_df[file_df["file_id"] == "file_2"].iloc[0]

    assert f1_row["top_k_mean_prob"] == pytest.approx(0.25)
    assert f1_row["pred"] == 0
    assert f2_row["top_k_mean_prob"] == pytest.approx(0.60)
    assert f2_row["pred"] == 1


def test_evaluate_per_file_max_and_positive_count():
    df = pd.DataFrame([
        {"file_id": "file_1", "label": "healthy", "window_id": "w1"},
        {"file_id": "file_1", "label": "healthy", "window_id": "w2"},
        {"file_id": "file_2", "label": "elevated_respiratory", "window_id": "w3"},
        {"file_id": "file_2", "label": "elevated_respiratory", "window_id": "w4"},
    ])
    probs = np.array([
        [0.9, 0.1],
        [0.4, 0.6],  # 1 noisy window
        [0.3, 0.7],
        [0.2, 0.8],  # 2 elevated windows
    ])

    # Under raw max > 0.5: file_1 is flagged positive (FP)
    res_max = evaluate_per_file(df, probs, strategy="max", threshold=0.5)
    assert res_max["file_level_df"][res_max["file_level_df"]["file_id"] == "file_1"]["pred"].iloc[0] == 1

    # Under positive_count (>= 2 windows > 0.5): file_1 is saved (TN) while file_2 is caught (TP)
    res_count = evaluate_per_file(df, probs, strategy="positive_count", top_k=2)
    assert res_count["file_level_df"][res_count["file_level_df"]["file_id"] == "file_1"]["pred"].iloc[0] == 0
    assert res_count["file_level_df"][res_count["file_level_df"]["file_id"] == "file_2"]["pred"].iloc[0] == 1
    assert res_count["accuracy"] == 1.0
