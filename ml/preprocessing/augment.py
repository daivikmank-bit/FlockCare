"""Audio and Mel-spectrogram data augmentations for training robustness."""

from typing import Optional, Tuple
import librosa
import numpy as np


def pitch_shift(
    y: np.ndarray,
    sr: int = 22050,
    n_steps_range: Tuple[float, float] = (-2.0, 2.0),
) -> np.ndarray:
    """Pitch shift audio by a random number of semitones within n_steps_range."""
    n_steps = np.random.uniform(*n_steps_range)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps).astype(np.float32)


def time_shift(y: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
    """Roll/shift audio in time by a random fraction up to shift_max."""
    shift = int(len(y) * np.random.uniform(-shift_max, shift_max))
    return np.roll(y, shift).astype(np.float32)


def add_noise(
    y: np.ndarray,
    noise_clip: Optional[np.ndarray] = None,
    snr_db_range: Tuple[float, float] = (5.0, 15.0),
) -> np.ndarray:
    """
    Inject background acoustic noise (e.g. coop fan, wind) or Gaussian noise
    at a randomly sampled Signal-to-Noise Ratio (SNR in dB).
    """
    snr_db = np.random.uniform(*snr_db_range)
    sig_power = np.mean(y**2) + 1e-8

    if noise_clip is not None and len(noise_clip) > 0:
        if len(noise_clip) < len(y):
            repeats = int(np.ceil(len(y) / len(noise_clip)))
            noise = np.tile(noise_clip, repeats)[: len(y)]
        else:
            start_idx = np.random.randint(0, len(noise_clip) - len(y) + 1)
            noise = noise_clip[start_idx : start_idx + len(y)]
    else:
        # Generate Gaussian noise if no noise audio clip provided
        noise = np.random.randn(len(y)).astype(np.float32)

    noise_power = np.mean(noise**2) + 1e-8
    scale = np.sqrt(sig_power / (10 ** (snr_db / 10.0) * noise_power))
    augmented = y + scale * noise
    return librosa.util.normalize(augmented).astype(np.float32)


def spec_augment(
    mel: np.ndarray,
    freq_mask: int = 12,
    time_mask: int = 20,
    num_freq_masks: int = 1,
    num_time_masks: int = 1,
) -> np.ndarray:
    """
    SpecAugment: Apply frequency and time masking to a 2D mel-spectrogram (n_mels, time_frames).
    """
    mel_aug = mel.copy()
    n_mels, n_frames = mel_aug.shape

    # Frequency masking
    for _ in range(num_freq_masks):
        f = np.random.randint(0, min(freq_mask, n_mels))
        if f > 0:
            f0 = np.random.randint(0, n_mels - f + 1)
            mel_aug[f0 : f0 + f, :] = 0.0

    # Time masking
    for _ in range(num_time_masks):
        t = np.random.randint(0, min(time_mask, n_frames))
        if t > 0:
            t0 = np.random.randint(0, n_frames - t + 1)
            mel_aug[:, t0 : t0 + t] = 0.0

    return mel_aug.astype(np.float32)
