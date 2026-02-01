#!/usr/bin/env python3
"""
Compute segmentation metrics for nnU-Net predictions.

Metrics computed (based on lumbar muscle segmentation literature):
- Dice Similarity Coefficient (DSC)
- Jaccard Index (IoU)
- Hausdorff Distance 95th percentile (HD95)
- Average Symmetric Surface Distance (ASSD)
- Precision (PPV)
- Recall (TPR/Sensitivity)
- Specificity (TNR)
- Relative Volume Error (RVE)

References:
- https://www.sciencedirect.com/science/article/pii/S2950363924000206
- https://onlinelibrary.wiley.com/doi/full/10.1002/jsp2.70003
"""

import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


LABEL_MAP = {
    1: "L_ES",
    2: "R_ES",
    3: "L_MF",
    4: "R_MF",
}


def compute_surface_distances(pred: np.ndarray, gt: np.ndarray, spacing: tuple = (1, 1, 1)):
    """
    Compute surface distances between prediction and ground truth.
    Returns distances from pred surface to gt and vice versa.
    """
    pred_border = pred ^ distance_transform_edt(pred) <= 1
    gt_border = gt ^ distance_transform_edt(gt) <= 1

    # Distance transform of the complement
    dt_gt = distance_transform_edt(~gt, sampling=spacing)
    dt_pred = distance_transform_edt(~pred, sampling=spacing)

    # Surface distances
    pred_to_gt = dt_gt[pred_border]
    gt_to_pred = dt_pred[gt_border]

    return pred_to_gt, gt_to_pred


def compute_hd95(pred: np.ndarray, gt: np.ndarray, spacing: tuple = (1, 1, 1)) -> float:
    """Compute 95th percentile Hausdorff Distance."""
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return np.nan

    try:
        pred_to_gt, gt_to_pred = compute_surface_distances(pred, gt, spacing)
        if len(pred_to_gt) == 0 or len(gt_to_pred) == 0:
            return np.nan
        all_distances = np.concatenate([pred_to_gt, gt_to_pred])
        return np.percentile(all_distances, 95)
    except Exception:
        return np.nan


def compute_assd(pred: np.ndarray, gt: np.ndarray, spacing: tuple = (1, 1, 1)) -> float:
    """Compute Average Symmetric Surface Distance."""
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return np.nan

    try:
        pred_to_gt, gt_to_pred = compute_surface_distances(pred, gt, spacing)
        if len(pred_to_gt) == 0 or len(gt_to_pred) == 0:
            return np.nan
        return (np.mean(pred_to_gt) + np.mean(gt_to_pred)) / 2
    except Exception:
        return np.nan


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient."""
    intersection = np.sum(pred * gt)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(gt) + 1e-8)


def compute_jaccard(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Jaccard index (IoU)."""
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return intersection / (union + 1e-8)


def compute_metrics_for_class(pred: np.ndarray, gt: np.ndarray, spacing: tuple = (1, 1, 1)) -> dict:
    """Compute all metrics for a single class."""
    pred_bin = pred > 0.5
    gt_bin = gt > 0.5

    # Basic counts
    tp = np.sum(pred_bin & gt_bin)
    fp = np.sum(pred_bin & ~gt_bin)
    fn = np.sum(~pred_bin & gt_bin)
    tn = np.sum(~pred_bin & ~gt_bin)

    # Overlap metrics
    dice = compute_dice(pred_bin.astype(float), gt_bin.astype(float))
    jaccard = compute_jaccard(pred_bin.astype(float), gt_bin.astype(float))

    # Classification metrics
    tpr = tp / (tp + fn + 1e-8)  # Sensitivity/Recall
    tnr = tn / (tn + fp + 1e-8)  # Specificity
    ppv = tp / (tp + fp + 1e-8)  # Precision

    # Volume metrics
    pred_vol = np.sum(pred_bin)
    gt_vol = np.sum(gt_bin)
    vol_ratio = pred_vol / (gt_vol + 1e-8)
    rel_vol_error = abs(pred_vol - gt_vol) / (gt_vol + 1e-8)  # Relative Volume Error

    # Surface distance metrics (only if both have content)
    hd95 = compute_hd95(pred_bin, gt_bin, spacing)
    assd = compute_assd(pred_bin, gt_bin, spacing)

    return {
        "dice": dice,
        "jaccard": jaccard,
        "hd95": hd95,
        "assd": assd,
        "tpr": tpr,
        "tnr": tnr,
        "ppv": ppv,
        "vol_ratio": vol_ratio,
        "rel_vol_error": rel_vol_error,
        "pred_vol_mm3": int(pred_vol * np.prod(spacing)),
        "gt_vol_mm3": int(gt_vol * np.prod(spacing)),
    }


def process_case(args):
    """Process a single case."""
    pred_path, gt_path, case_id = args

    try:
        # Load prediction and ground truth
        pred_nii = nib.load(pred_path)
        gt_nii = nib.load(gt_path)

        pred_data = pred_nii.get_fdata()
        gt_data = gt_nii.get_fdata()

        # Get voxel spacing
        spacing = pred_nii.header.get_zooms()[:3]

        results = {"case_id": case_id}

        # Compute metrics for each class
        all_dice = []
        all_hd95 = []
        all_assd = []

        for label_val, label_name in LABEL_MAP.items():
            pred_class = (pred_data == label_val).astype(float)
            gt_class = (gt_data == label_val).astype(float)

            metrics = compute_metrics_for_class(pred_class, gt_class, spacing)
            for metric_name, value in metrics.items():
                results[f"{label_name}_{metric_name}"] = value

            all_dice.append(metrics["dice"])
            if not np.isnan(metrics["hd95"]):
                all_hd95.append(metrics["hd95"])
            if not np.isnan(metrics["assd"]):
                all_assd.append(metrics["assd"])

        # Compute macro averages
        results["dice_macro"] = np.mean(all_dice)
        results["hd95_macro"] = np.mean(all_hd95) if all_hd95 else np.nan
        results["assd_macro"] = np.mean(all_assd) if all_assd else np.nan

        return case_id, results, None

    except Exception as e:
        return case_id, None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Compute segmentation metrics")
    parser.add_argument("--pred-dir", type=str, required=True,
                        help="Directory containing nnU-Net predictions")
    parser.add_argument("--gt-dir", type=str, required=True,
                        help="Directory containing ground truth labels (nnU-Net format)")
    parser.add_argument("--output", type=str, default="metrics.csv",
                        help="Output CSV file")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--wandb", action="store_true",
                        help="Log results to W&B (offline mode)")
    parser.add_argument("--wandb-project", type=str, default="nnunet-lumbar-muscle",
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization overlays after computing metrics")
    parser.add_argument("--viz-output-dir", type=str, default=None,
                        help="Output directory for visualizations (default: pred-dir/visualizations)")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory containing original images (required if --visualize)")
    parser.add_argument("--viz-n-slices", type=int, default=3,
                        help="Number of slices to visualize per case (default: 3)")
    parser.add_argument("--viz-montage", action="store_true",
                        help="Also generate multi-view montage for each case")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    # Find matching prediction and ground truth files
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    pred_files = [p for p in pred_files if not p.name.endswith("_probs.nii.gz")]

    process_args = []
    for pred_path in pred_files:
        case_id = pred_path.name.replace(".nii.gz", "")
        gt_path = gt_dir / f"{case_id}.nii.gz"

        if gt_path.exists():
            process_args.append((pred_path, gt_path, case_id))
        else:
            print(f"Warning: No ground truth found for {case_id}")

    print(f"Processing {len(process_args)} cases...")

    all_results = []
    failed = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_case, arg): arg[2]
                   for arg in process_args}

        for future in tqdm(as_completed(futures), total=len(futures)):
            case_id, results, error = future.result()
            if results:
                all_results.append(results)
            else:
                failed.append((case_id, error))

    # Create DataFrame and save
    df = pd.DataFrame(all_results)
    df = df.sort_values("case_id")
    df.to_csv(args.output, index=False)

    # Print summary
    print(f"\nResults saved to: {args.output}")
    print(f"Processed: {len(all_results)} cases")

    if failed:
        print(f"Failed: {len(failed)}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY METRICS (mean ± std)")
    print("=" * 70)

    # Macro metrics
    print(f"\n{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)

    for metric in ['dice_macro', 'hd95_macro', 'assd_macro']:
        if metric in df.columns:
            values = df[metric].dropna()
            print(f"{metric:<20} {values.mean():>10.4f} {values.std():>10.4f} {values.min():>10.4f} {values.max():>10.4f}")

    # Per-class Dice
    print("\n" + "-" * 70)
    print("Per-class Dice:")
    for label_name in LABEL_MAP.values():
        dice_col = f"{label_name}_dice"
        if dice_col in df.columns:
            values = df[dice_col]
            print(f"  {label_name:<10} {values.mean():>10.4f} ± {values.std():.4f}")

    # Per-class HD95
    print("\nPer-class HD95 (mm):")
    for label_name in LABEL_MAP.values():
        hd95_col = f"{label_name}_hd95"
        if hd95_col in df.columns:
            values = df[hd95_col].dropna()
            if len(values) > 0:
                print(f"  {label_name:<10} {values.mean():>10.4f} ± {values.std():.4f}")

    # Save summary JSON
    summary = {
        "n_cases": len(all_results),
        "metrics": {}
    }

    for metric in ['dice_macro', 'hd95_macro', 'assd_macro']:
        if metric in df.columns:
            values = df[metric].dropna()
            summary["metrics"][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }

    for label_name in LABEL_MAP.values():
        for metric in ['dice', 'hd95', 'assd']:
            col = f"{label_name}_{metric}"
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    summary["metrics"][col] = {
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                    }

    summary_path = Path(args.output).with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # W&B logging
    if args.wandb:
        try:
            import wandb

            os.environ.setdefault('WANDB_MODE', 'offline')

            run_name = args.wandb_run_name or f"eval_{Path(args.pred_dir).name}"

            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={"pred_dir": str(pred_dir), "gt_dir": str(gt_dir)},
            )

            # Log summary metrics
            wandb_metrics = {"n_cases": len(all_results)}
            for metric, values in summary["metrics"].items():
                wandb_metrics[f"{metric}_mean"] = values["mean"]
                wandb_metrics[f"{metric}_std"] = values["std"]

            wandb.log(wandb_metrics)

            # Log per-case table
            wandb.log({"per_case_metrics": wandb.Table(dataframe=df)})

            wandb.finish()
            print(f"\nW&B run logged (offline mode). Sync with: wandb sync {wandb.run.dir}")

        except Exception as e:
            print(f"\nW&B logging failed: {e}")

    # Generate visualizations if requested
    if args.visualize:
        if not args.image_dir:
            print("\nError: --image-dir is required when using --visualize")
        else:
            print("\n" + "=" * 70)
            print("Generating visualizations...")
            print("=" * 70)

            viz_output = args.viz_output_dir or str(pred_dir / "visualizations")

            try:
                from generate_visualizations import (
                    generate_visualization,
                    generate_montage,
                    Path as VizPath,
                )

                viz_output_path = Path(viz_output)
                viz_output_path.mkdir(parents=True, exist_ok=True)
                image_dir_path = Path(args.image_dir)

                viz_success = 0
                viz_failed = 0

                for pred_path, gt_path, case_id in tqdm(process_args, desc="Generating visualizations"):
                    image_path = image_dir_path / f"{case_id}_0000.nii.gz"

                    if not image_path.exists():
                        viz_failed += 1
                        continue

                    try:
                        result = generate_visualization(
                            image_path, gt_path, pred_path,
                            viz_output_path, case_id, args.viz_n_slices
                        )
                        if result["status"] == "success":
                            viz_success += 1
                        else:
                            viz_failed += 1

                        if args.viz_montage:
                            generate_montage(
                                image_path, gt_path, pred_path,
                                viz_output_path, case_id
                            )
                    except Exception as e:
                        viz_failed += 1

                print(f"\nVisualizations generated: {viz_success}")
                print(f"Visualizations failed: {viz_failed}")
                print(f"Output directory: {viz_output}")

            except ImportError as e:
                print(f"\nFailed to import visualization module: {e}")
                print("Make sure generate_visualizations.py is in the same directory")


if __name__ == "__main__":
    main()
