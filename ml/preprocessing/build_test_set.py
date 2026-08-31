"""Batch processing script for the held-out Poultry Vocalization test dataset."""

import argparse
import csv
import glob
import os
import uuid
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from ml.preprocessing.audio_utils import (
    load_audio,
    chunk_audio,
    to_mel_spectrogram,
    fix_length,
)

DEFAULT_TEST_DIRS = {
    "data/raw/poultry_vocalization/healthy": "healthy",
    "data/raw/poultry_vocalization/unhealthy": "elevated_respiratory",
    "data/raw/poultry_vocalization/noise": "no_bird_sound",
}


def build_test_set(
    test_dirs: dict = None,
    out_dir: str = "data/spectrograms/test",
    manifest_path: str = "data/spectrograms/test_manifest.csv",
):
    """
    Process raw Poultry Vocalization WAV files into fixed (128, 216) .npy spectrograms,
    preserving file_id grouping for aggregated evaluation in Part 4.
    """
    if test_dirs is None:
        test_dirs = DEFAULT_TEST_DIRS

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)

    manifest = []
    total_files = 0
    total_windows = 0

    print("Building held-out test set from Poultry Vocalization Dataset...")
    for folder, target_label in test_dirs.items():
        if not os.path.exists(folder):
            print(f"Warning: Raw directory '{folder}' not found. Skipping.")
            continue

        wav_files = []
        for ext in ("*.wav", "*.WAV", "*.mp3", "*.flac"):
            wav_files.extend(glob.glob(os.path.join(folder, ext)))
            wav_files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
        wav_files = sorted(list(set(wav_files)))

        print(f"Processing {len(wav_files)} files for label '{target_label}' from {folder}...")
        for path in tqdm(wav_files, desc=f"{target_label}"):
            try:
                file_id = str(uuid.uuid4())
                y = load_audio(path)
                chunks = chunk_audio(y)
                dest_dir = os.path.join(out_dir, target_label)
                os.makedirs(dest_dir, exist_ok=True)

                for w_idx, chunk in enumerate(chunks):
                    mel = fix_length(to_mel_spectrogram(chunk))
                    window_id = f"{file_id}_{w_idx}"
                    out_path = os.path.join(dest_dir, f"{window_id}.npy")
                    np.save(out_path, mel)
                    manifest.append([window_id, file_id, path, target_label, "poultry_vocalization"])
                    total_windows += 1

                total_files += 1
            except Exception as e:
                print(f"Error processing {path}: {e}")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["window_id", "file_id", "source_path", "label", "dataset"])
        writer.writerows(manifest)

    print(
        f"Done! Wrote {total_windows} test windows from {total_files} source files to {out_dir}. "
        f"Manifest saved to {manifest_path}."
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build test spectrogram dataset from Poultry Vocalization Dataset.")
    parser.add_argument("--out-dir", default="data/spectrograms/test", help="Directory for output .npy files")
    parser.add_argument(
        "--manifest", default="data/spectrograms/test_manifest.csv", help="Path for output manifest CSV"
    )
    args = parser.parse_args()

    build_test_set(out_dir=args.out_dir, manifest_path=args.manifest)
