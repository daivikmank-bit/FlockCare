"""Evaluation and benchmarking package for FlockCare."""

from ml.evaluation.evaluate import evaluate_per_window, evaluate_per_file, run_full_evaluation
from ml.evaluation.baseline_model import BaselineChickenCNN, load_baseline, baseline_preprocess, run_baseline_benchmark

__all__ = [
    "evaluate_per_window",
    "evaluate_per_file",
    "run_full_evaluation",
    "BaselineChickenCNN",
    "load_baseline",
    "baseline_preprocess",
    "run_baseline_benchmark",
]
