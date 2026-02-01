#!/usr/bin/env python3
"""
Generate visualization overlays comparing ground truth vs predictions.

Creates side-by-side comparison images showing:
- Original grayscale image
- Ground truth segmentation overlay
- Prediction segmentation overlay

Color scheme:
- L_ES (1): Red
- R_ES (2): Blue
- L_MF (3): Green
- R_MF (4): Yellow
"""

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm


LABEL_MAP = {
    1: "L_ES",
    2: "R_ES",
    3: "L_MF",
    4: "R_MF",
}

# RGBA colors for each label (values 0-1)
COLORS = {
    1: [1.0, 0.0, 0.0, 0.6],  # L_ES: Red
    2: [0.0, 0.0, 1.0, 0.6],  # R_ES: Blue
    3: [0.0, 1.0, 0.0, 0.6],  # L_MF: Green
    4: [1.0, 1.0, 0.0, 0.6],  # R_MF: Yellow
}


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to 0-1 range with percentile clipping."""
    p1, p99 = np.percentile(img, [1, 99])
    img_clipped = np.clip(img, p1, p99)
    return (img_clipped - p1) / (p99 - p1 + 1e-8)


def create_color_overlay(mask: np.ndarray) -> np.ndarray:
    """Create RGBA overlay from segmentation mask."""
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    for label_val, color in COLORS.items():
        label_mask = mask == label_val
        for c in range(4):
            overlay[:, :, c] += label_mask * color[c]

    return overlay


def find_representative_slices(mask: np.ndarray, n_slices: int = 3) -> list:
    """Find slices with the most muscle content."""
    # Sum muscle voxels per slice (along z-axis, assuming last dim is z)
    muscle_presence = (mask > 0).sum(axis=(0, 1))

    if muscle_presence.max() == 0:
        # No muscles found, return middle slices
        mid = mask.shape[2] // 2
        return [mid - 1, mid, mid + 1]

    # Get top n_slices with most muscle content
    top_slices = np.argsort(muscle_presence)[-n_slices:]
    return sorted(top_slices)


def generate_visualization(
    image_path: Path,
    gt_path: Path,
    pred_path: Path,
    output_dir: Path,
    case_id: str,
    n_slices: int = 3,
) -> dict:
    """Generate comparison visualization for a single case."""
    try:
        # Load data
        img_nii = nib.load(image_path)
        gt_nii = nib.load(gt_path)
        pred_nii = nib.load(pred_path)

        img = img_nii.get_fdata()
        gt = gt_nii.get_fdata().astype(np.int32)
        pred = pred_nii.get_fdata().astype(np.int32)

        # Normalize image
        img_norm = normalize_image(img)

        # Find representative slices based on ground truth
        slices = find_representative_slices(gt, n_slices)

        output_paths = []

        for slice_idx in slices:
            # Extract 2D slices (axial view)
            img_slice = img_norm[:, :, slice_idx]
            gt_slice = gt[:, :, slice_idx]
            pred_slice = pred[:, :, slice_idx]

            # Create figure with 3 panels
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'{case_id} - Slice {slice_idx}', fontsize=14, fontweight='bold')

            # Panel 1: Original image
            axes[0].imshow(img_slice.T, cmap='gray', origin='lower', aspect='equal')
            axes[0].set_title('Original Image', fontsize=12)
            axes[0].axis('off')

            # Panel 2: Ground Truth overlay
            axes[1].imshow(img_slice.T, cmap='gray', origin='lower', aspect='equal')
            gt_overlay = create_color_overlay(gt_slice)
            axes[1].imshow(gt_overlay.transpose(1, 0, 2), origin='lower', aspect='equal')
            axes[1].set_title('Ground Truth', fontsize=12)
            axes[1].axis('off')

            # Panel 3: Prediction overlay
            axes[2].imshow(img_slice.T, cmap='gray', origin='lower', aspect='equal')
            pred_overlay = create_color_overlay(pred_slice)
            axes[2].imshow(pred_overlay.transpose(1, 0, 2), origin='lower', aspect='equal')
            axes[2].set_title('Prediction', fontsize=12)
            axes[2].axis('off')

            # Add legend
            legend_elements = [
                Patch(facecolor=COLORS[1][:3], alpha=0.6, label='L_ES'),
                Patch(facecolor=COLORS[2][:3], alpha=0.6, label='R_ES'),
                Patch(facecolor=COLORS[3][:3], alpha=0.6, label='L_MF'),
                Patch(facecolor=COLORS[4][:3], alpha=0.6, label='R_MF'),
            ]
            fig.legend(
                handles=legend_elements,
                loc='lower center',
                ncol=4,
                fontsize=10,
                frameon=True,
                bbox_to_anchor=(0.5, 0.02)
            )

            plt.tight_layout(rect=[0, 0.08, 1, 0.96])

            # Save figure
            output_path = output_dir / f"{case_id}_slice{slice_idx:03d}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            output_paths.append(str(output_path))

        return {"case_id": case_id, "status": "success", "outputs": output_paths}

    except Exception as e:
        return {"case_id": case_id, "status": "failed", "error": str(e)}


def generate_montage(
    image_path: Path,
    gt_path: Path,
    pred_path: Path,
    output_dir: Path,
    case_id: str,
) -> dict:
    """Generate a montage showing all views and slices."""
    try:
        # Load data
        img_nii = nib.load(image_path)
        gt_nii = nib.load(gt_path)
        pred_nii = nib.load(pred_path)

        img = img_nii.get_fdata()
        gt = gt_nii.get_fdata().astype(np.int32)
        pred = pred_nii.get_fdata().astype(np.int32)

        # Normalize image
        img_norm = normalize_image(img)

        # Find best slice
        muscle_presence = (gt > 0).sum(axis=(0, 1))
        best_slice = np.argmax(muscle_presence)

        # Create figure: 2 rows (GT, Pred) x 3 columns (Axial, Sagittal, Coronal)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{case_id} - Multi-view Comparison', fontsize=14, fontweight='bold')

        views = [
            ("Axial", lambda x: x[:, :, best_slice], (1, 0, 2)),
            ("Sagittal", lambda x: x[x.shape[0]//2, :, :], (1, 0, 2)),
            ("Coronal", lambda x: x[:, x.shape[1]//2, :], (1, 0, 2)),
        ]

        for col, (view_name, slice_fn, transpose_order) in enumerate(views):
            img_slice = slice_fn(img_norm)
            gt_slice = slice_fn(gt)
            pred_slice = slice_fn(pred)

            # Ground truth row
            axes[0, col].imshow(img_slice.T, cmap='gray', origin='lower', aspect='auto')
            gt_overlay = create_color_overlay(gt_slice)
            axes[0, col].imshow(gt_overlay.transpose(1, 0, 2), origin='lower', aspect='auto')
            axes[0, col].set_title(f'{view_name} - GT', fontsize=11)
            axes[0, col].axis('off')

            # Prediction row
            axes[1, col].imshow(img_slice.T, cmap='gray', origin='lower', aspect='auto')
            pred_overlay = create_color_overlay(pred_slice)
            axes[1, col].imshow(pred_overlay.transpose(1, 0, 2), origin='lower', aspect='auto')
            axes[1, col].set_title(f'{view_name} - Pred', fontsize=11)
            axes[1, col].axis('off')

        # Add legend
        legend_elements = [
            Patch(facecolor=COLORS[1][:3], alpha=0.6, label='L_ES'),
            Patch(facecolor=COLORS[2][:3], alpha=0.6, label='R_ES'),
            Patch(facecolor=COLORS[3][:3], alpha=0.6, label='L_MF'),
            Patch(facecolor=COLORS[4][:3], alpha=0.6, label='R_MF'),
        ]
        fig.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=4,
            fontsize=10,
            frameon=True,
            bbox_to_anchor=(0.5, 0.02)
        )

        plt.tight_layout(rect=[0, 0.06, 1, 0.96])

        output_path = output_dir / f"{case_id}_montage.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return {"case_id": case_id, "status": "success", "output": str(output_path)}

    except Exception as e:
        return {"case_id": case_id, "status": "failed", "error": str(e)}


def process_case(args):
    """Process a single case (for parallel execution)."""
    image_path, gt_path, pred_path, output_dir, case_id, n_slices, montage = args

    results = []

    # Generate slice visualizations
    result = generate_visualization(
        image_path, gt_path, pred_path, output_dir, case_id, n_slices
    )
    results.append(result)

    # Generate montage if requested
    if montage:
        montage_result = generate_montage(
            image_path, gt_path, pred_path, output_dir, case_id
        )
        results.append(montage_result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate GT vs Prediction visualization overlays"
    )
    parser.add_argument(
        "--image-dir", type=str, required=True,
        help="Directory containing original images (*_0000.nii.gz)"
    )
    parser.add_argument(
        "--gt-dir", type=str, required=True,
        help="Directory containing ground truth labels"
    )
    parser.add_argument(
        "--pred-dir", type=str, required=True,
        help="Directory containing predictions"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--n-slices", type=int, default=3,
        help="Number of slices to visualize per case (default: 3)"
    )
    parser.add_argument(
        "--montage", action="store_true",
        help="Also generate multi-view montage for each case"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--cases", type=str, nargs="+", default=None,
        help="Specific case IDs to process (default: all)"
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find prediction files
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    pred_files = [p for p in pred_files if not p.name.endswith("_probs.nii.gz")]

    # Build processing list
    process_args = []
    for pred_path in pred_files:
        case_id = pred_path.name.replace(".nii.gz", "")

        # Filter by specified cases if provided
        if args.cases and case_id not in args.cases:
            continue

        # Find corresponding image and GT
        image_path = image_dir / f"{case_id}_0000.nii.gz"
        gt_path = gt_dir / f"{case_id}.nii.gz"

        if not image_path.exists():
            print(f"Warning: No image found for {case_id}")
            continue
        if not gt_path.exists():
            print(f"Warning: No ground truth found for {case_id}")
            continue

        process_args.append((
            image_path, gt_path, pred_path, output_dir, case_id,
            args.n_slices, args.montage
        ))

    print(f"Processing {len(process_args)} cases...")

    successful = 0
    failed = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_case, arg): arg[4] for arg in process_args}

        for future in tqdm(as_completed(futures), total=len(futures)):
            case_id = futures[future]
            try:
                results = future.result()
                for result in results:
                    if result["status"] == "success":
                        successful += 1
                    else:
                        failed.append((result["case_id"], result.get("error", "Unknown")))
            except Exception as e:
                failed.append((case_id, str(e)))

    print(f"\nVisualization complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(failed)}")
    print(f"  Output directory: {output_dir}")

    if failed:
        print("\nFailed cases:")
        for case_id, error in failed[:10]:
            print(f"  {case_id}: {error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


if __name__ == "__main__":
    main()
