#!/usr/bin/env python3
"""
Extract validation cases for a specific fold from nnU-Net splits.

Creates symlinks to validation images in a temporary directory for inference.
This ensures we only run inference on held-out data (not training data).
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Extract validation cases for a specific fold"
    )
    parser.add_argument(
        "--splits-file", type=str, required=True,
        help="Path to splits_final.json"
    )
    parser.add_argument(
        "--images-dir", type=str, required=True,
        help="Directory containing training images (imagesTr)"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for symlinks to validation images"
    )
    parser.add_argument(
        "--fold", type=int, required=True,
        help="Fold number (0-4)"
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="Only print case IDs, don't create symlinks"
    )
    args = parser.parse_args()

    # Load splits
    with open(args.splits_file, "r") as f:
        splits = json.load(f)

    if args.fold < 0 or args.fold >= len(splits):
        print(f"Error: fold {args.fold} out of range (0-{len(splits)-1})")
        return 1

    val_cases = splits[args.fold]["val"]
    print(f"Fold {args.fold}: {len(val_cases)} validation cases")

    if args.list_only:
        for case_id in sorted(val_cases):
            print(case_id)
        return 0

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images_dir)

    # Create symlinks
    created = 0
    missing = []
    for case_id in val_cases:
        src = images_dir / f"{case_id}_0000.nii.gz"
        dst = output_dir / f"{case_id}_0000.nii.gz"

        if not src.exists():
            missing.append(case_id)
            continue

        # Remove existing symlink if present
        if dst.exists() or dst.is_symlink():
            dst.unlink()

        dst.symlink_to(src.resolve())
        created += 1

    print(f"Created {created} symlinks in {output_dir}")

    if missing:
        print(f"Warning: {len(missing)} cases not found:")
        for case_id in missing[:5]:
            print(f"  {case_id}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")

    return 0


if __name__ == "__main__":
    exit(main())
