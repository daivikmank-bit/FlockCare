"""Batch processing script for the SmartEars training dataset."""

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
from ml.preprocessing.labels import normalize_label, has_bird_signal

DEFAULT_RAW_DIRS = {
    "data/raw/smartears/healthy": "healthy",
    "data/raw/smartears/sick": "elevated_respiratory",
    # "none" intentionally excluded from CNN training per Part 3.3
}


def build_train_set(
    raw_dirs: dict = None,
    out_dir: str = "data/spectrograms/train",
    manifest_path: str = "data/spectrograms/train_manifest.csv",
    filter_energy: bool = True,
):
    """
    Process raw SmartEars WAV files into fixed (128, 216) .npy spectrograms
    and write the train_manifest.csv.
    """
    if raw_dirs is None:
        raw_dirs = DEFAULT_RAW_DIRS

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)

    manifest = []
    skipped_energy = 0
    total_processed = 0

    print("Building training set from SmartEars...")
    for folder, target_label in raw_dirs.items():
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
                y = load_audio(path)
                if filter_energy and not has_bird_signal(y):
                    skipped_energy += 1
                    continue

                chunks = chunk_audio(y)
                for chunk in chunks:
                    mel = fix_length(to_mel_spectrogram(chunk))
                    clip_id = str(uuid.uuid4())
                    dest_dir = os.path.join(out_dir, target_label)
                    os.makedirs(dest_dir, exist_ok=True)
                    out_path = os.path.join(dest_dir, f"{clip_id}.npy")
                    np.save(out_path, mel)
                    manifest.append([clip_id, path, target_label, "smartears"])
                    total_processed += 1
            except Exception as e:
                print(f"Error processing {path}: {e}")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "source_path", "label", "dataset"])
        writer.writerows(manifest)

    print(
        f"Done! Wrote {total_processed} training spectrogram windows to {out_dir} "
        f"(skipped {skipped_energy} low-energy clips). Manifest saved to {manifest_path}."
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training spectrogram dataset from SmartEars.")
    parser.add_argument("--out-dir", default="data/spectrograms/train", help="Directory for output .npy files")
    parser.add_argument(
        "--manifest", default="data/spectrograms/train_manifest.csv", help="Path for output manifest CSV"
    )
    args = parser.parse_args()

    build_train_set(out_dir=args.out_dir, manifest_path=args.manifest)
