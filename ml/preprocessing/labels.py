"""Label definitions, mapping schemas, and audio signal pre-filters."""

import numpy as np

# Canonical mapping for SmartEars and Poultry Vocalization datasets
LABEL_MAP = {
    # SmartEars
    "healthy": "healthy",
    "sick": "elevated_respiratory",
    "none": "no_bird_sound",
    # Poultry Vocalization Dataset
    "Healthy": "healthy",
    "unhealthy": "elevated_respiratory",
    "Unhealthy": "elevated_respiratory",
    "noise": "no_bird_sound",
    "Noise": "no_bird_sound",
}

# Binary classification targets for Part 4 CNN
LABEL_TO_IDX = {
    "healthy": 0,
    "elevated_respiratory": 1,
}

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


def normalize_label(raw_label: str) -> str:
    """Map a raw folder or dataset label to the canonical FlockCare label."""
    clean = raw_label.strip()
    if clean in LABEL_MAP:
        return LABEL_MAP[clean]
    clean_lower = clean.lower()
    if clean_lower in LABEL_MAP:
        return LABEL_MAP[clean_lower]
    raise ValueError(f"Unknown raw label: {raw_label}")


def has_bird_signal(y: np.ndarray, energy_threshold: float = 0.01) -> bool:
    """
    Cheap pre-filter: rejects near-silent / non-bird recordings before spectrogram/CNN.
    Computes Root Mean Square (RMS) energy.
    """
    if y is None or len(y) == 0:
        return False
    rms = np.sqrt(np.mean(y.astype(np.float32) ** 2))
    return bool(rms > energy_threshold)
