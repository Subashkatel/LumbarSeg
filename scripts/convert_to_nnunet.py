#!/usr/bin/env python3
"""
Convert lumbar muscle segmentation dataset to nnU-Net v2 format.

Input format (current):
    disjoint_all/<sid>/image/<sid>_image.nii.gz
    disjoint_all/<sid>/masks/<sid>_L_ES.nii.gz
    disjoint_all/<sid>/masks/<sid>_R_ES.nii.gz
    disjoint_all/<sid>/masks/<sid>_L_MF.nii.gz
    disjoint_all/<sid>/masks/<sid>_R_MF.nii.gz

Output format (nnU-Net v2):
    nnUNet_raw/Dataset001_LumbarMuscle/imagesTr/<case>_0000.nii.gz
    nnUNet_raw/Dataset001_LumbarMuscle/labelsTr/<case>.nii.gz

Label mapping:
    0 = Background
    1 = L_ES (Left Erector Spinae)
    2 = R_ES (Right Erector Spinae)
    3 = L_MF (Left Multifidus)
    4 = R_MF (Right Multifidus)
"""

import argparse
import json
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
from tqdm import tqdm


LABEL_MAP = {
    "L_ES": 1,
    "R_ES": 2,
    "L_MF": 3,
    "R_MF": 4,
}


def process_subject(args):
    """Process a single subject: copy image and combine masks."""
    sid, src_dir, images_dir, labels_dir = args

    try:
        src_path = Path(src_dir)

        # Source paths
        img_path = src_path / "image" / f"{sid}_image.nii.gz"
        mask_paths = {
            k: src_path / "masks" / f"{sid}_{k}.nii.gz"
            for k in LABEL_MAP.keys()
        }

        # Check all files exist
        if not img_path.exists():
            return sid, False, f"Image not found: {img_path}"

        for name, mpath in mask_paths.items():
            if not mpath.exists():
                return sid, False, f"Mask not found: {mpath}"

        # Copy image with nnU-Net naming convention
        dst_img = Path(images_dir) / f"{sid}_0000.nii.gz"
        shutil.copy2(img_path, dst_img)

        # Load first mask to get shape/affine
        ref_nii = nib.load(mask_paths["L_ES"])
        combined = np.zeros(ref_nii.shape, dtype=np.uint8)

        # Combine masks with priority (MF inside ES, so ES first then MF overwrites)
        # Actually for distinct labels we just assign each voxel to one class
        # If there's overlap, higher label wins (MF > ES) since MF is contained in ES region
        for name in ["L_ES", "R_ES", "L_MF", "R_MF"]:
            mask_nii = nib.load(mask_paths[name])
            mask_data = mask_nii.get_fdata()
            combined[mask_data > 0.5] = LABEL_MAP[name]

        # Save combined label
        dst_label = Path(labels_dir) / f"{sid}.nii.gz"
        combined_nii = nib.Nifti1Image(combined, ref_nii.affine, ref_nii.header)
        combined_nii.header.set_data_dtype(np.uint8)
        nib.save(combined_nii, dst_label)

        return sid, True, None

    except Exception as e:
        return sid, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Convert dataset to nnU-Net format")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to disjoint_all directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to nnUNet_raw/Dataset001_LumbarMuscle")
    parser.add_argument("--train-list", type=str, default=None,
                        help="JSON file with training subjects (optional, uses all if not provided)")
    parser.add_argument("--test-list", type=str, default=None,
                        help="JSON file with test subjects for imagesTs (optional)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    images_tr = output_dir / "imagesTr"
    labels_tr = output_dir / "labelsTr"
    images_ts = output_dir / "imagesTs"

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)

    # Get subject list
    if args.train_list:
        with open(args.train_list) as f:
            train_data = json.load(f)
        subjects = [d["sid"] for d in train_data]
    else:
        # Use all subjects in input directory
        subjects = sorted([d.name for d in input_dir.iterdir()
                          if d.is_dir() and not d.name.startswith(".")])

    print(f"Converting {len(subjects)} subjects to nnU-Net format...")

    # Prepare arguments for parallel processing
    process_args = [
        (sid, input_dir / sid, images_tr, labels_tr)
        for sid in subjects
    ]

    # Process in parallel
    success_count = 0
    failed = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_subject, arg): arg[0]
                   for arg in process_args}

        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="Converting"):
            sid, success, error = future.result()
            if success:
                success_count += 1
            else:
                failed.append((sid, error))

    print(f"\nConversion complete: {success_count}/{len(subjects)} subjects")

    if failed:
        print(f"\nFailed subjects ({len(failed)}):")
        for sid, error in failed:
            print(f"  {sid}: {error}")

    # Handle test set if provided
    if args.test_list:
        with open(args.test_list) as f:
            test_data = json.load(f)
        test_subjects = [d["sid"] for d in test_data]

        print(f"\nCopying {len(test_subjects)} test subjects...")
        for sid in tqdm(test_subjects, desc="Test set"):
            src_img = input_dir / sid / "image" / f"{sid}_image.nii.gz"
            if src_img.exists():
                shutil.copy2(src_img, images_ts / f"{sid}_0000.nii.gz")

    print(f"\nOutput written to: {output_dir}")
    print("\nNext steps:")
    print("  1. Run: python scripts/create_dataset_json.py")
    print("  2. Run: nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity")


if __name__ == "__main__":
    main()
