"""Unit tests for PyTorch baseline model architecture and preprocessing."""

import numpy as np
import pytest
import torch

from ml.evaluation.baseline_model import (
    BaselineChickenCNN,
    baseline_preprocess,
    BASELINE_SR,
    BASELINE_WAV_SIZE,
    BASELINE_N_MELS,
)


def test_baseline_architecture():
    model = BaselineChickenCNN(num_classes=3)
    model.eval()

    # Input: (batch, 1, 128, ~130)
    x = torch.randn(2, 1, 128, 130)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 3)


def test_baseline_preprocess():
    # 2 seconds of synthetic audio at 22050 Hz
    sr = BASELINE_SR
    y = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)).astype(np.float32)

    mel = baseline_preprocess(y)

    assert isinstance(mel, np.ndarray)
    assert mel.shape[0] == BASELINE_N_MELS
    # Window is 1.5s -> ~130 time frames
    assert 120 <= mel.shape[1] <= 140
    assert mel.dtype == np.float32
