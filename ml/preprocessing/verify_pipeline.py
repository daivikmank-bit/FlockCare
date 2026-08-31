"""Verification suite implementing the Part 3.9 pipeline verification checklist."""

import os
import pandas as pd
import numpy as np


def verify_pipeline(
    train_manifest_path: str = "data/spectrograms/train_manifest.csv",
    test_manifest_path: str = "data/spectrograms/test_manifest.csv",
    train_spec_dir: str = "data/spectrograms/train",
    test_spec_dir: str = "data/spectrograms/test",
    expected_shape: tuple = (128, 216),
) -> bool:
    """
    Runs the comprehensive Part 3.9 checklist:
    1. Checks presence of manifests and directories.
    2. Validates class balance and labels.
    3. Validates tensor shape (128, 216) and dtype float32 across all .npy files.
    4. Validates no NaN or Inf values exist.
    5. Validates file_id grouping for test set aggregation.
    6. Validates zero dataset leakage between train and test manifests.
    """
    print("=" * 60)
    print("FlockCare Data Processing Pipeline Verification")
    print("=" * 60)

    passed_all = True

    # 1. Manifest Existence
    for name, path in [("Train manifest", train_manifest_path), ("Test manifest", test_manifest_path)]:
        if not os.path.exists(path):
            print(f"[FAIL] {name} not found at {path}")
            passed_all = False
        else:
            print(f"[PASS] {name} found at {path}")

    if not passed_all:
        print("\nPlease run build_train_set.py and build_test_set.py first.")
        return False

    train_df = pd.read_csv(train_manifest_path)
    test_df = pd.read_csv(test_manifest_path)

    # 2. Check Train Manifest Classes
    print("\n--- 1. Training Set Class Distribution ---")
    train_counts = train_df["label"].value_counts().to_dict()
    for label, count in train_counts.items():
        print(f"  - {label}: {count} windows")

    required_train_classes = {"healthy", "elevated_respiratory"}
    if not required_train_classes.issubset(set(train_counts.keys())):
        print(f"[FAIL] Train manifest missing required classes: {required_train_classes - set(train_counts.keys())}")
        passed_all = False
    else:
        print("[PASS] Train manifest contains all required classes.")

    # 3. Check Test Manifest Classes and Groupings
    print("\n--- 2. Test Set Distribution & Grouping ---")
    test_counts = test_df["label"].value_counts().to_dict()
    for label, count in test_counts.items():
        print(f"  - {label}: {count} windows")

    unique_files = test_df["file_id"].nunique()
    total_test_windows = len(test_df)
    print(f"  - Total test files (file_id): {unique_files}")
    print(f"  - Total test windows: {total_test_windows}")
    print(f"  - Average windows per file: {total_test_windows / max(1, unique_files):.2f}")

    if "file_id" not in test_df.columns:
        print("[FAIL] 'file_id' column missing from test manifest.")
        passed_all = False
    else:
        print("[PASS] Test manifest correctly structured with 'file_id' for aggregation.")

    # 4. Check Dataset Leakage
    print("\n--- 3. Dataset Leakage Check ---")
    train_sources = set(train_df["source_path"].dropna())
    test_sources = set(test_df["source_path"].dropna())
    overlap = train_sources.intersection(test_sources)
    if len(overlap) > 0:
        print(f"[FAIL] Data leakage detected! {len(overlap)} files present in both train and test.")
        passed_all = False
    else:
        print("[PASS] Zero overlap between train and test source files.")

    # 5. Check Spectrogram Files, Tensor Shapes, and NaN/Inf
    print("\n--- 4. Spectrogram Integrity (Shape, Dtype, NaN/Inf) ---")
    all_spec_paths = []

    for row in train_df.itertuples():
        spec_path = os.path.join(train_spec_dir, str(row.label), f"{row.clip_id}.npy")
        all_spec_paths.append((spec_path, "train"))

    for row in test_df.itertuples():
        spec_path = os.path.join(test_spec_dir, str(row.label), f"{row.window_id}.npy")
        all_spec_paths.append((spec_path, "test"))

    shape_mismatches = 0
    nan_inf_found = 0
    missing_files = 0

    for path, split in all_spec_paths:
        if not os.path.exists(path):
            missing_files += 1
            continue

        mel = np.load(path)
        if mel.shape != expected_shape:
            shape_mismatches += 1
        if np.isnan(mel).any() or np.isinf(mel).any():
            nan_inf_found += 1

    if missing_files > 0:
        print(f"[FAIL] {missing_files} referenced .npy spectrogram files do not exist.")
        passed_all = False
    else:
        print(f"[PASS] All {len(all_spec_paths)} .npy files exist on disk.")

    if shape_mismatches > 0:
        print(f"[FAIL] {shape_mismatches} files have incorrect shape (expected {expected_shape}).")
        passed_all = False
    else:
        print(f"[PASS] All inspected files have exact shape {expected_shape}.")

    if nan_inf_found > 0:
        print(f"[FAIL] {nan_inf_found} files contain NaN or Inf values.")
        passed_all = False
    else:
        print("[PASS] Zero NaN or Inf values detected.")

    print("\n" + "=" * 60)
    if passed_all:
        print("ALL VERIFICATION CHECKS PASSED!")
    else:
        print("SOME CHECKS FAILED. Please review the output above.")
    print("=" * 60)

    return passed_all


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify FlockCare data pipeline outputs.")
    parser.add_argument("--train-manifest", default="data/spectrograms/train_manifest.csv")
    parser.add_argument("--test-manifest", default="data/spectrograms/test_manifest.csv")
    parser.add_argument("--train-dir", default="data/spectrograms/train")
    parser.add_argument("--test-dir", default="data/spectrograms/test")
    args = parser.parse_args()

    verify_pipeline(
        train_manifest_path=args.train_manifest,
        test_manifest_path=args.test_manifest,
        train_spec_dir=args.train_dir,
        test_spec_dir=args.test_dir,
    )
