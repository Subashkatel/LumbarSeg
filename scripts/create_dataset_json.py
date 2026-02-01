#!/usr/bin/env python3
"""
Create dataset.json for nnU-Net v2.

This file describes the dataset structure, modalities, and label mapping.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create nnU-Net dataset.json")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Path to nnUNet_raw/Dataset001_LumbarMuscle")
    parser.add_argument("--dataset-name", type=str, default="LumbarMuscle",
                        help="Name of the dataset")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"

    # Count training cases
    train_images = sorted(images_tr.glob("*_0000.nii.gz"))
    num_training = len(train_images)

    # Build training list
    training_list = []
    for img_path in train_images:
        case_id = img_path.name.replace("_0000.nii.gz", "")
        training_list.append({
            "image": f"./imagesTr/{case_id}_0000.nii.gz",
            "label": f"./labelsTr/{case_id}.nii.gz"
        })

    # Dataset configuration
    dataset_json = {
        "channel_names": {
            "0": "MRI"
        },
        "labels": {
            "background": 0,
            "L_ES": 1,
            "R_ES": 2,
            "L_MF": 3,
            "R_MF": 4
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "name": args.dataset_name,
        "description": "Lumbar paraspinal muscle segmentation (Erector Spinae and Multifidus)",
        "reference": "Internal dataset",
        "licence": "Internal use only",
        "release": "1.0",
        "overwrite_image_reader_writer": "SimpleITKIO"
    }

    # Write dataset.json
    output_path = dataset_dir / "dataset.json"
    with open(output_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Created {output_path}")
    print(f"  - Training cases: {num_training}")
    print(f"  - Channels: 1 (MRI)")
    print(f"  - Labels: 5 (background + 4 muscles)")
    print("\nNext step:")
    print("  nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity")


if __name__ == "__main__":
    main()
