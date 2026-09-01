"""Evaluation routines for per-window and per-file (aggregated) out-of-distribution performance."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf

from ml.training.data import load_split, LABEL_TO_IDX


def evaluate_per_window(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Compute per-window classification metrics on the held-out test split.
    
    Args:
        model: Trained Keras model.
        X_test: Feature array (N, 128, 216, 1).
        y_test: True integer labels (N,).
        target_names: Class names list (default: ['healthy', 'elevated_respiratory']).
        
    Returns:
        Dictionary containing predictions, probabilities, report string, report dict, and confusion matrix.
    """
    if target_names is None:
        target_names = ["healthy", "elevated_respiratory"]

    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)

    report_dict = classification_report(
        y_test,
        preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0.0,
    )
    report_text = classification_report(
        y_test,
        preds,
        target_names=target_names,
        zero_division=0.0,
    )
    cm = confusion_matrix(y_test, preds)

    return {
        "probs": probs,
        "preds": preds,
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, pos_label=1, zero_division=0.0)),
        "recall": float(recall_score(y_test, preds, pos_label=1, zero_division=0.0)),
        "f1": float(f1_score(y_test, preds, pos_label=1, zero_division=0.0)),
        "classification_report_text": report_text,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_per_file(
    test_df: pd.DataFrame,
    probs: np.ndarray,
    strategy: str = "top_k_mean",
    threshold: float = 0.5,
    top_k: int = 2,
    target_names: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Aggregate per-window probabilities to recording/file level.
    
    Strategies:
      - 'top_k_mean': Average the top-k highest elevated probabilities in the file (dampens single-window fluke noise).
      - 'max': Maximum elevated probability across all windows in the file.
      - 'mean': Average elevated probability across all windows.
      - 'positive_count': Flags file if at least `top_k` windows have elevated_prob > threshold.
      
    Args:
        test_df: DataFrame of test windows containing 'file_id' and 'label'.
        probs: Window probabilities of shape (N, 2).
        strategy: Aggregation method ('top_k_mean' | 'max' | 'mean' | 'positive_count').
        threshold: Decision boundary on flock score (default: 0.5).
        top_k: Number of top windows to average for 'top_k_mean' or require for 'positive_count' (default: 2).
        target_names: Names of classes.
        
    Returns:
        Dictionary containing file-level DataFrame, report string, metrics, and confusion matrix.
    """
    if target_names is None:
        target_names = ["healthy", "elevated_respiratory"]

    df = test_df.copy()
    df["healthy_prob"] = probs[:, 0]
    df["elevated_prob"] = probs[:, 1]

    # Group by file_id
    file_records = []
    for file_id, group in df.groupby("file_id"):
        label = group["label"].iloc[0]
        elevated_scores = group["elevated_prob"].values
        k = min(top_k, len(elevated_scores))
        
        # Calculate scores under different strategies
        max_score = float(np.max(elevated_scores))
        mean_score = float(np.mean(elevated_scores))
        sorted_scores = np.sort(elevated_scores)[::-1]
        top_k_score = float(np.mean(sorted_scores[:k]))
        positive_window_count = int(np.sum(elevated_scores > 0.5))

        if strategy == "top_k_mean":
            flock_score = top_k_score
            is_positive = int(flock_score >= threshold)
        elif strategy == "max":
            flock_score = max_score
            is_positive = int(flock_score >= threshold)
        elif strategy == "mean":
            flock_score = mean_score
            is_positive = int(flock_score >= threshold)
        elif strategy == "positive_count":
            flock_score = max_score
            is_positive = int(positive_window_count >= k)
        else:
            flock_score = top_k_score
            is_positive = int(flock_score >= threshold)

        file_records.append({
            "file_id": file_id,
            "label": label,
            "num_windows": len(elevated_scores),
            "max_elevated_prob": max_score,
            "top_k_mean_prob": top_k_score,
            "mean_elevated_prob": mean_score,
            "positive_window_count": positive_window_count,
            "flock_score": flock_score,
            "pred": is_positive,
        })

    file_level = pd.DataFrame(file_records)
    y_file_true = file_level["label"].map(LABEL_TO_IDX).values.astype(int)
    y_file_pred = file_level["pred"].values

    report_dict = classification_report(
        y_file_true,
        y_file_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0.0,
    )
    report_text = classification_report(
        y_file_true,
        y_file_pred,
        target_names=target_names,
        zero_division=0.0,
    )
    cm = confusion_matrix(y_file_true, y_file_pred)

    return {
        "strategy": strategy,
        "threshold": threshold,
        "top_k": top_k,
        "file_level_df": file_level,
        "num_files": len(file_level),
        "accuracy": float(accuracy_score(y_file_true, y_file_pred)),
        "precision": float(precision_score(y_file_true, y_file_pred, pos_label=1, zero_division=0.0)),
        "recall": float(recall_score(y_file_true, y_file_pred, pos_label=1, zero_division=0.0)),
        "f1": float(f1_score(y_file_true, y_file_pred, pos_label=1, zero_division=0.0)),
        "classification_report_text": report_text,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }


def compare_aggregation_strategies(test_df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """Evaluate and compare multiple window aggregation strategies on file-level performance."""
    strategies_to_test = [
        ("max", 0.50, 1, "Raw max > 0.50"),
        ("max", 0.70, 1, "Calibrated max > 0.70"),
        ("max", 0.80, 1, "Conservative max > 0.80"),
        ("top_k_mean", 0.50, 2, "Top-2 Mean > 0.50"),
        ("top_k_mean", 0.60, 2, "Top-2 Mean > 0.60"),
        ("top_k_mean", 0.50, 3, "Top-3 Mean > 0.50"),
        ("positive_count", 0.50, 2, ">= 2 Windows > 0.50"),
        ("mean", 0.40, 1, "Global Mean > 0.40"),
    ]

    rows = []
    for strat, thresh, k, desc in strategies_to_test:
        eval_res = evaluate_per_file(test_df, probs, strategy=strat, threshold=thresh, top_k=k)
        cm = eval_res["confusion_matrix"]
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        rows.append({
            "Description": desc,
            "Strategy": strat,
            "Threshold": thresh,
            "Top-K": k,
            "Accuracy": f"{eval_res['accuracy'] * 100:.1f}%",
            "Precision (Elevated)": f"{eval_res['precision'] * 100:.1f}%",
            "Recall (Elevated)": f"{eval_res['recall'] * 100:.1f}%",
            "F1-Score": f"{eval_res['f1'] * 100:.1f}%",
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        })

    return pd.DataFrame(rows)


def run_full_evaluation(
    model: tf.keras.Model,
    test_manifest: str = "data/spectrograms/test_manifest.csv",
    test_spec_dir: str = "data/spectrograms/test",
    threshold: float = 0.5,
    strategy: str = "top_k_mean",
) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline (both per-window and aggregated per-file) on held-out test data.
    
    Args:
        model: Trained Keras model.
        test_manifest: Path to test_manifest.csv.
        test_spec_dir: Path to test spectrogram directory.
        threshold: Aggregation threshold.
        strategy: Aggregation strategy ('top_k_mean' | 'max' | 'positive_count' | 'mean').
        
    Returns:
        Dictionary containing per_window, per_file evaluation results and strategy comparison table.
    """
    X_test, y_test, test_df = load_split(test_manifest, test_spec_dir, id_col="window_id")

    # 1. Per-window evaluation
    window_eval = evaluate_per_window(model, X_test, y_test)

    # 2. Per-file aggregated evaluation (Headline)
    file_eval = evaluate_per_file(test_df, window_eval["probs"], strategy=strategy, threshold=threshold)

    # 3. Strategy comparison table
    strategy_comparison = compare_aggregation_strategies(test_df, window_eval["probs"])

    print("=================== 1. PER-WINDOW EVALUATION (OOD HELD-OUT) ===================")
    print(window_eval["classification_report_text"])
    print("Per-Window Confusion Matrix:\n", np.array(window_eval["confusion_matrix"]))

    print(f"\n=================== 2. PER-FILE AGGREGATED EVALUATION ({strategy.upper()} @ {threshold}) ===================")
    print(file_eval["classification_report_text"])
    print("Per-File Confusion Matrix:\n", np.array(file_eval["confusion_matrix"]))

    print("\n=================== 3. AGGREGATION STRATEGY COMPARISON ===================")
    print(strategy_comparison.to_string(index=False))

    return {
        "per_window": window_eval,
        "per_file": file_eval,
        "strategy_comparison": strategy_comparison,
    }
