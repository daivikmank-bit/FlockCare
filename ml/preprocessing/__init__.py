"""FlockCare ML Preprocessing package."""

from .labels import LABEL_MAP, LABEL_TO_IDX, IDX_TO_LABEL, has_bird_signal
from .audio_utils import (
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
from .augment import pitch_shift, time_shift, add_noise, spec_augment

__all__ = [
    "LABEL_MAP",
    "LABEL_TO_IDX",
    "IDX_TO_LABEL",
    "has_bird_signal",
    "TARGET_SR",
    "WINDOW_SEC",
    "N_MELS",
    "HOP_LENGTH",
    "TARGET_FRAMES",
    "load_audio",
    "chunk_audio",
    "to_mel_spectrogram",
    "fix_length",
    "process_audio_file",
    "aggregate_window_predictions",
    "pitch_shift",
    "time_shift",
    "add_noise",
    "spec_augment",
]
