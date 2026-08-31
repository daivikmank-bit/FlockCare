"""Comprehensive test suite for the FlockCare ML Preprocessing package."""

import os
import tempfile
import numpy as np
import pytest
import soundfile as sf

from ml.preprocessing.labels import (
    normalize_label,
    has_bird_signal,
    LABEL_MAP,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
)
from ml.preprocessing.audio_utils import (
    TARGET_SR,
    WINDOW_SEC,
    N_MELS,
    HOP_LENGTH,
    TARGET_FRAMES,
    load_audio,
    chunk_audio,
    to_mel_spectrogram,
    fix_length,
    process_audio_file,
    aggregate_window_predictions,
)
from ml.preprocessing.augment import (
    pitch_shift,
    time_shift,
    add_noise,
    spec_augment,
)


class TestLabels:
    def test_label_normalization(self):
        assert normalize_label("healthy") == "healthy"
        assert normalize_label("sick") == "elevated_respiratory"
        assert normalize_label("unhealthy") == "elevated_respiratory"
        assert normalize_label("Healthy") == "healthy"
        assert normalize_label("Unhealthy") == "elevated_respiratory"
        assert normalize_label("none") == "no_bird_sound"
        assert normalize_label("noise") == "no_bird_sound"
        assert normalize_label("Noise") == "no_bird_sound"

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError):
            normalize_label("alien_species")

    def test_label_indices(self):
        assert LABEL_TO_IDX["healthy"] == 0
        assert LABEL_TO_IDX["elevated_respiratory"] == 1
        assert IDX_TO_LABEL[0] == "healthy"
        assert IDX_TO_LABEL[1] == "elevated_respiratory"

    def test_has_bird_signal(self):
        # Empty array
        assert not has_bird_signal(np.array([]))
        # Silence
        silence = np.zeros(22050 * 2)
        assert not has_bird_signal(silence, energy_threshold=0.01)
        # Low noise below threshold
        low_noise = 0.002 * np.random.randn(22050)
        assert not has_bird_signal(low_noise, energy_threshold=0.01)
        # Active bird signal
        active = 0.2 * np.sin(np.linspace(0, 100, 22050))
        assert has_bird_signal(active, energy_threshold=0.01)


class TestAudioUtils:
    @pytest.fixture
    def sample_audio(self):
        # 5 seconds sine tone at 440Hz
        sr = TARGET_SR
        t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        return y.astype(np.float32)

    def test_constants(self):
        assert TARGET_SR == 22050
        assert WINDOW_SEC == 5
        assert N_MELS == 128
        assert HOP_LENGTH == 512
        assert TARGET_FRAMES == 216

    def test_chunk_audio_short(self):
        sr = TARGET_SR
        short_y = np.ones(sr * 2, dtype=np.float32)  # 2 seconds
        chunks = chunk_audio(short_y, sr=sr, window_sec=5)
        assert len(chunks) == 1
        assert len(chunks[0]) == sr * 5  # Padded to full 5s

    def test_chunk_audio_exact(self, sample_audio):
        sr = TARGET_SR
        chunks = chunk_audio(sample_audio, sr=sr, window_sec=5)
        assert len(chunks) == 1
        assert len(chunks[0]) == sr * 5

    def test_chunk_audio_long_with_tail(self):
        sr = TARGET_SR
        # 12 seconds: 5s + 5s + 2s remainder (> 30% of 5s window) -> 3 chunks
        long_y = np.ones(sr * 12, dtype=np.float32)
        chunks = chunk_audio(long_y, sr=sr, window_sec=5)
        assert len(chunks) == 3
        for c in chunks:
            assert len(c) == sr * 5

    def test_chunk_audio_drops_tiny_tail(self):
        sr = TARGET_SR
        # 10.5 seconds: 5s + 5s + 0.5s remainder (10% < 30% of 5s window) -> 2 chunks
        long_y = np.ones(int(sr * 10.5), dtype=np.float32)
        chunks = chunk_audio(long_y, sr=sr, window_sec=5)
        assert len(chunks) == 2
        for c in chunks:
            assert len(c) == sr * 5

    def test_to_mel_spectrogram_and_fix_length(self, sample_audio):
        mel = to_mel_spectrogram(sample_audio, sr=TARGET_SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
        assert mel.shape[0] == 128
        # Output should be normalized in [0, 1]
        assert mel.min() >= 0.0
        assert mel.max() <= 1.0 + 1e-6
        assert mel.dtype == np.float32

        mel_fixed = fix_length(mel, target_frames=TARGET_FRAMES)
        assert mel_fixed.shape == (128, 216)

    def test_fix_length_pad_and_truncate(self):
        short_mel = np.ones((128, 200), dtype=np.float32)
        fixed_short = fix_length(short_mel, target_frames=216)
        assert fixed_short.shape == (128, 216)
        assert (fixed_short[:, 200:] == 0).all()

        long_mel = np.ones((128, 250), dtype=np.float32)
        fixed_long = fix_length(long_mel, target_frames=216)
        assert fixed_long.shape == (128, 216)

    def test_load_and_process_audio_file(self, sample_audio):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            sf.write(temp_path, sample_audio, TARGET_SR)
            loaded_y = load_audio(temp_path, target_sr=TARGET_SR)
            assert len(loaded_y) > 0
            assert np.max(np.abs(loaded_y)) <= 1.0

            specs = process_audio_file(temp_path)
            assert len(specs) >= 1
            for spec in specs:
                assert spec.shape == (128, 216)
                assert not np.isnan(spec).any()
                assert not np.isinf(spec).any()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_aggregate_window_predictions(self):
        probs = [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],  # 1 elevated window
            [0.7, 0.3],
            [0.6, 0.4],
            [0.9, 0.1],
        ]
        agg = aggregate_window_predictions(probs, elevated_idx=1)
        assert agg["max_prob"] == pytest.approx(0.8)
        assert agg["frac_flagged"] == pytest.approx(1 / 6)
        assert agg["mean_prob"] == pytest.approx((0.1 + 0.2 + 0.8 + 0.3 + 0.4 + 0.1) / 6)

    def test_aggregate_empty(self):
        agg = aggregate_window_predictions([])
        assert agg["max_prob"] == 0.0
        assert agg["frac_flagged"] == 0.0


class TestAugmentations:
    @pytest.fixture
    def sample_audio(self):
        sr = TARGET_SR
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        return y.astype(np.float32)

    def test_pitch_shift(self, sample_audio):
        shifted = pitch_shift(sample_audio, sr=TARGET_SR, n_steps_range=(1.0, 2.0))
        assert len(shifted) == len(sample_audio)
        assert not np.allclose(shifted, sample_audio)

    def test_time_shift(self, sample_audio):
        shifted = time_shift(sample_audio, shift_max=0.3)
        assert len(shifted) == len(sample_audio)

    def test_add_noise(self, sample_audio):
        noisy = add_noise(sample_audio, snr_db_range=(10.0, 10.0))
        assert len(noisy) == len(sample_audio)
        assert np.max(np.abs(noisy)) <= 1.0 + 1e-5

    def test_spec_augment(self):
        mel = np.ones((128, 216), dtype=np.float32)
        aug = spec_augment(mel, freq_mask=12, time_mask=20)
        assert aug.shape == (128, 216)
        # Check that some values were masked to 0
        assert (aug == 0.0).any()
