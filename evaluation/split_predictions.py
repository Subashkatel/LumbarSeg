#!/usr/bin/env python3
"""
Split nnU-Net multi-label predictions back to separate binary masks.

Input: nnU-Net prediction with labels 0-4
Output: Separate binary masks for each muscle class
"""

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
from tqdm import tqdm


LABEL_MAP = {
    1: "L_ES",
    2: "R_ES",
    3: "L_MF",
    4: "R_MF",
}


def process_prediction(args):
    """Split a single prediction file into separate masks."""
    pred_path, output_dir = args

    try:
        pred_path = Path(pred_path)
        output_dir = Path(output_dir)

        # Load prediction
        pred_nii = nib.load(pred_path)
        pred_data = pred_nii.get_fdata().astype(np.uint8)

        # Extract case ID
        case_id = pred_path.name.replace(".nii.gz", "")

        # Create output directory for this case
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # Split into separate masks
        for label_val, label_name in LABEL_MAP.items():
            mask = (pred_data == label_val).astype(np.uint8)
            mask_nii = nib.Nifti1Image(mask, pred_nii.affine, pred_nii.header)
            mask_nii.header.set_data_dtype(np.uint8)

            out_path = case_dir / f"{case_id}_{label_name}.nii.gz"
            nib.save(mask_nii, out_path)

        return case_id, True, None

    except Exception as e:
        return pred_path.name, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Split nnU-Net predictions")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing nnU-Net predictions")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for split masks")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all prediction files
    pred_files = sorted(input_dir.glob("*.nii.gz"))
    # Exclude probability files
    pred_files = [p for p in pred_files if not p.name.endswith("_probs.nii.gz")]

    print(f"Processing {len(pred_files)} predictions...")

    process_args = [(p, output_dir) for p in pred_files]

    success_count = 0
    failed = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_prediction, arg): arg[0]
                   for arg in process_args}

        for future in tqdm(as_completed(futures), total=len(futures)):
            case_id, success, error = future.result()
            if success:
                success_count += 1
            else:
                failed.append((case_id, error))

    print(f"\nProcessed: {success_count}/{len(pred_files)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for case_id, error in failed:
            print(f"  {case_id}: {error}")


if __name__ == "__main__":
    main()
