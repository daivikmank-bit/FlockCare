"""Generate synthetic acoustic sample WAV files to test the data pipeline before downloading full Mendeley datasets."""

import os
import numpy as np
import soundfile as sf


def generate_audio_signal(
    duration: float = 5.0,
    sr: int = 22050,
    signal_type: str = "healthy",
) -> np.ndarray:
    """
    Synthesizes mock audio:
    - 'healthy': Harmonic tones (chicken vocalizations ~400-1200Hz) with gentle envelope.
    - 'elevated' / 'sick': Harmonic tones mixed with higher-frequency wheezing/rattles (2500-4000Hz).
    - 'noise': Low-frequency ambient rumble and pink/white noise.
    - 'silent': Low-level noise below energy threshold.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    if signal_type == "healthy":
        # Clucks and chirps: bursts of harmonics at ~500Hz, 1000Hz, 1500Hz
        signal = np.zeros_like(t)
        # Add 3 periodic burst clucks
        num_clucks = max(2, int(duration))
        for i in range(num_clucks):
            center = (i + 0.5) * (duration / num_clucks)
            env = np.exp(-((t - center) ** 2) / (2 * 0.15**2))  # 150ms cluck envelope
            cluck = (
                0.6 * np.sin(2 * np.pi * 550 * t)
                + 0.3 * np.sin(2 * np.pi * 1100 * t)
                + 0.1 * np.sin(2 * np.pi * 1650 * t)
            )
            signal += env * cluck
        # Add slight ambient background
        signal += 0.02 * np.random.randn(len(t))

    elif signal_type in ("sick", "elevated", "unhealthy"):
        # Wheeze & rattle: prolonged strained tones with high-frequency noise bursts
        signal = np.zeros_like(t)
        num_bursts = max(2, int(duration))
        for i in range(num_bursts):
            center = (i + 0.5) * (duration / num_bursts)
            env = np.exp(-((t - center) ** 2) / (2 * 0.35**2))  # 350ms prolonged rasp
            wheeze = (
                0.5 * np.sin(2 * np.pi * 650 * t)
                + 0.4 * np.sin(2 * np.pi * 2800 * t)  # High wheezing harmonic
                + 0.3 * np.sin(2 * np.pi * 3600 * t)  # Upper respiratory rattle
            )
            signal += env * wheeze
        signal += 0.03 * np.random.randn(len(t))

    elif signal_type in ("noise", "none"):
        # Coop fan / rumble + wind noise
        rumble = 0.05 * np.sin(2 * np.pi * 60 * t) + 0.03 * np.sin(2 * np.pi * 120 * t)
        noise = 0.04 * np.random.randn(len(t))
        signal = rumble + noise

    elif signal_type == "silent":
        signal = 0.001 * np.random.randn(len(t))

    else:
        signal = 0.05 * np.random.randn(len(t))

    # Normalize within [-0.95, 0.95]
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = 0.9 * (signal / max_val)

    return signal.astype(np.float32)


def generate_sample_datasets(base_raw_dir: str = "data/raw", num_samples_per_class: int = 5):
    """
    Creates mock raw sample datasets matching the exact folder structures:
    data/raw/smartears/{healthy, sick, none}
    data/raw/poultry_vocalization/{healthy, unhealthy, noise}
    """
    sr = 22050

    # 1. SmartEars (5-second clips)
    smartears_config = {
        "healthy": ("healthy", 5.0),
        "sick": ("sick", 5.0),
        "none": ("none", 5.0),
    }

    print("Generating synthetic SmartEars dataset...")
    for folder_name, (sig_type, dur) in smartears_config.items():
        folder_path = os.path.join(base_raw_dir, "smartears", folder_name)
        os.makedirs(folder_path, exist_ok=True)
        for i in range(num_samples_per_class):
            y = generate_audio_signal(duration=dur, sr=sr, signal_type=sig_type)
            file_path = os.path.join(folder_path, f"sample_{folder_name}_{i:03d}.wav")
            sf.write(file_path, y, sr)

    # 2. Poultry Vocalization Dataset (variable length 6s to 12s clips)
    poultry_config = {
        "healthy": ("healthy", 11.0),
        "unhealthy": ("unhealthy", 12.0),
        "noise": ("noise", 8.0),
    }

    print("Generating synthetic Poultry Vocalization dataset...")
    for folder_name, (sig_type, dur) in poultry_config.items():
        folder_path = os.path.join(base_raw_dir, "poultry_vocalization", folder_name)
        os.makedirs(folder_path, exist_ok=True)
        for i in range(num_samples_per_class):
            y = generate_audio_signal(duration=dur, sr=sr, signal_type=sig_type)
            file_path = os.path.join(folder_path, f"sample_{folder_name}_{i:03d}.wav")
            sf.write(file_path, y, sr)

    print(f"Sample datasets generated successfully in '{base_raw_dir}'.")


if __name__ == "__main__":
    generate_sample_datasets()
