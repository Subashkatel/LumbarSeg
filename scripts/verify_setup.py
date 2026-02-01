#!/usr/bin/env python3
"""
Verify nnU-Net setup and data conversion.

Run this script to ensure everything is correctly configured before training.
"""

import sys
from pathlib import Path


def check_imports():
    """Verify all required packages are installed."""
    print("=" * 60)
    print("1. CHECKING IMPORTS")
    print("=" * 60)

    errors = []

    # Check PyTorch
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"    CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("    ⚠ CUDA not available (OK on login node)")
    except ImportError as e:
        errors.append(f"PyTorch: {e}")
        print(f"  ✗ PyTorch: {e}")

    # Check nnU-Net
    try:
        import nnunetv2
        from importlib.metadata import version
        nnunet_version = version('nnunetv2')
        print(f"  ✓ nnU-Net v2 {nnunet_version}")
    except ImportError as e:
        errors.append(f"nnunetv2: {e}")
        print(f"  ✗ nnunetv2: {e}")

    # Check other dependencies
    for pkg in ['nibabel', 'numpy', 'scipy', 'pandas', 'sklearn', 'matplotlib', 'tqdm']:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {pkg} {version}")
        except ImportError as e:
            errors.append(f"{pkg}: {e}")
            print(f"  ✗ {pkg}: {e}")

    return len(errors) == 0


def check_environment_variables():
    """Verify nnU-Net environment variables are set."""
    print("\n" + "=" * 60)
    print("2. CHECKING ENVIRONMENT VARIABLES")
    print("=" * 60)

    import os

    required_vars = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
    errors = []

    for var in required_vars:
        value = os.environ.get(var)
        if value:
            exists = Path(value).exists()
            status = "✓" if exists else "⚠ (path doesn't exist)"
            print(f"  {status} {var}={value}")
            if not exists:
                errors.append(f"{var} path doesn't exist: {value}")
        else:
            print(f"  ✗ {var} not set")
            errors.append(f"{var} not set")

    if errors:
        print("\n  To set environment variables:")
        print("  export nnUNet_raw=/scratch/gpfs/MARTONOSI/sk2415/ml/nnUNet_raw")
        print("  export nnUNet_preprocessed=/scratch/gpfs/MARTONOSI/sk2415/ml/nnUNet_preprocessed")
        print("  export nnUNet_results=/scratch/gpfs/MARTONOSI/sk2415/ml/nnUNet_results")

    return len(errors) == 0


def check_dataset():
    """Verify dataset structure and files."""
    print("\n" + "=" * 60)
    print("3. CHECKING DATASET")
    print("=" * 60)

    import os
    import json

    base_path = os.environ.get('nnUNet_raw', '/scratch/gpfs/MARTONOSI/sk2415/ml/nnUNet_raw')
    dataset_path = Path(base_path) / "Dataset001_LumbarMuscle"

    errors = []

    # Check dataset.json
    dataset_json = dataset_path / "dataset.json"
    if dataset_json.exists():
        with open(dataset_json) as f:
            config = json.load(f)
        print(f"  ✓ dataset.json found")
        print(f"    - Name: {config.get('name', 'N/A')}")
        print(f"    - Channels: {config.get('channel_names', {})}")
        print(f"    - Labels: {config.get('labels', {})}")
        print(f"    - Training cases: {config.get('numTraining', 0)}")
    else:
        print(f"  ✗ dataset.json not found at {dataset_json}")
        errors.append("dataset.json missing")

    # Check imagesTr
    images_tr = dataset_path / "imagesTr"
    if images_tr.exists():
        images = list(images_tr.glob("*_0000.nii.gz"))
        print(f"  ✓ imagesTr: {len(images)} images found")
    else:
        print(f"  ✗ imagesTr directory not found")
        errors.append("imagesTr missing")

    # Check labelsTr
    labels_tr = dataset_path / "labelsTr"
    if labels_tr.exists():
        labels = list(labels_tr.glob("*.nii.gz"))
        print(f"  ✓ labelsTr: {len(labels)} labels found")
    else:
        print(f"  ✗ labelsTr directory not found")
        errors.append("labelsTr missing")

    # Verify matching
    if images_tr.exists() and labels_tr.exists():
        image_ids = {p.name.replace("_0000.nii.gz", "") for p in images}
        label_ids = {p.name.replace(".nii.gz", "") for p in labels}

        missing_labels = image_ids - label_ids
        missing_images = label_ids - image_ids

        if missing_labels:
            print(f"  ⚠ {len(missing_labels)} images without labels")
            errors.append(f"{len(missing_labels)} images without labels")
        if missing_images:
            print(f"  ⚠ {len(missing_images)} labels without images")
            errors.append(f"{len(missing_images)} labels without images")

        if not missing_labels and not missing_images:
            print(f"  ✓ All {len(image_ids)} cases have matching images and labels")

    return len(errors) == 0


def check_label_values():
    """Verify label files have correct values (0-4)."""
    print("\n" + "=" * 60)
    print("4. CHECKING LABEL VALUES (sampling 3 files)")
    print("=" * 60)

    import os
    import nibabel as nib
    import numpy as np

    base_path = os.environ.get('nnUNet_raw', '/scratch/gpfs/MARTONOSI/sk2415/ml/nnUNet_raw')
    labels_tr = Path(base_path) / "Dataset001_LumbarMuscle" / "labelsTr"

    if not labels_tr.exists():
        print("  ✗ labelsTr not found, skipping")
        return False

    label_files = sorted(labels_tr.glob("*.nii.gz"))[:3]
    expected_labels = {0, 1, 2, 3, 4}
    errors = []

    for label_file in label_files:
        try:
            nii = nib.load(label_file)
            data = nii.get_fdata()
            unique_values = set(np.unique(data).astype(int))

            if unique_values.issubset(expected_labels):
                print(f"  ✓ {label_file.name}: values {sorted(unique_values)}")
            else:
                unexpected = unique_values - expected_labels
                print(f"  ✗ {label_file.name}: unexpected values {unexpected}")
                errors.append(f"{label_file.name} has unexpected values")
        except Exception as e:
            print(f"  ✗ {label_file.name}: error loading - {e}")
            errors.append(str(e))

    return len(errors) == 0


def check_nnunet_commands():
    """Verify nnU-Net CLI commands are available."""
    print("\n" + "=" * 60)
    print("5. CHECKING nnU-Net COMMANDS")
    print("=" * 60)

    import shutil

    commands = [
        'nnUNetv2_plan_and_preprocess',
        'nnUNetv2_train',
        'nnUNetv2_predict',
        'nnUNetv2_find_best_configuration',
    ]

    errors = []
    for cmd in commands:
        path = shutil.which(cmd)
        if path:
            print(f"  ✓ {cmd}")
        else:
            print(f"  ✗ {cmd} not found in PATH")
            errors.append(f"{cmd} not found")

    return len(errors) == 0


def main():
    print("\n" + "=" * 60)
    print("  nnU-Net v2 SETUP VERIFICATION")
    print("=" * 60)

    results = {
        "imports": check_imports(),
        "env_vars": check_environment_variables(),
        "dataset": check_dataset(),
        "label_values": check_label_values(),
        "commands": check_nnunet_commands(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("  ALL CHECKS PASSED - Ready to train!")
        print("  Next: sbatch scripts/preprocess.sbatch")
    else:
        print("  SOME CHECKS FAILED - Please fix issues above")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
