#!/usr/bin/env python3
"""
LumbarSeg - Command Line Interface

Usage:
    lumbarseg -i scan.nii.gz
    lumbarseg -i scan.nii.gz -o results/
    lumbarseg -i scan.nii.gz --gt L_ES.nii R_ES.nii L_Mult.nii R_Mult.nii
"""

import argparse
import sys
from pathlib import Path

from . import __version__
from .python_api import segment, segment_batch


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="lumbarseg",
        description="LumbarSeg - Segment lumbar paraspinal muscles from MRI scans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output Labels:
  0 = Background
  1 = Left Erector Spinae (L_ES)
  2 = Right Erector Spinae (R_ES)
  3 = Left Multifidus (L_MF)
  4 = Right Multifidus (R_MF)

Output Folder Structure:
  output/
  ├── segmentation.nii.gz    Multi-label segmentation (0-4)
  ├── L_ES.nii.gz            Binary mask - Left Erector Spinae
  ├── R_ES.nii.gz            Binary mask - Right Erector Spinae
  ├── L_MF.nii.gz            Binary mask - Left Multifidus
  ├── R_MF.nii.gz            Binary mask - Right Multifidus
  ├── preview.png            Segmentation overlay image
  └── metrics.csv            Evaluation metrics (only with --gt)

Examples:
  lumbarseg -i scan.nii.gz                      # -> scan_segmented/
  lumbarseg -i scan.nii.gz -o results/           # -> results/
  lumbarseg -i scans_folder/ -o results_folder/
  lumbarseg -i scan.nii.gz --fast
  lumbarseg -i scan.nii.gz --device cpu
  lumbarseg -i scan.nii.gz --gt L_ES.nii R_ES.nii L_Mult.nii R_Mult.nii
  lumbarseg -i scan.nii.gz --gt combined_gt.nii.gz
        """,
    )

    # Required arguments
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Input NIfTI file or directory containing NIfTI files",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory (default: <input_name>_segmented/ next to input file)",
    )

    # Optional arguments
    parser.add_argument(
        "-f", "--fold",
        default="all",
        help="Fold(s) to use: 0-4 for single fold, 'all' for ensemble (default: all)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use single fold (fold 0) for faster inference",
    )
    parser.add_argument(
        "-d", "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device for inference (default: auto-detect)",
    )
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Save probability maps",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        nargs="+",
        default=None,
        help="Ground truth mask file(s) for evaluation. Either a single multi-label NIfTI or separate binary masks (L_ES, R_ES, L_Mult, R_Mult)",
    )
    parser.add_argument(
        "--disable-tta",
        action="store_true",
        help="Disable test-time augmentation (8x faster, slightly lower accuracy)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output messages",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"LumbarSeg {__version__}",
    )

    args = parser.parse_args()

    # Handle --fast flag
    fold = "0" if args.fast else args.fold

    # Parse fold argument
    if fold == "all" or fold == "ensemble":
        fold = "all"
    elif fold.isdigit():
        fold = int(fold)

    verbose = not args.quiet

    # Convert gt paths to list or None
    ground_truth = args.gt if args.gt else None

    # Auto-generate output path if not specified
    if args.output is None:
        if args.input.is_dir():
            args.output = args.input.parent / f"{args.input.name}_segmented"
        else:
            # Remove .nii.gz or .nii extension
            stem = args.input.name
            for suffix in ['.nii.gz', '.nii']:
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break
            args.output = args.input.parent / f"{stem}_segmented"

    try:
        # Check if input is file or directory
        if args.input.is_dir():
            # Batch processing
            segment_batch(
                input_dir=args.input,
                output_dir=args.output,
                fold=fold,
                device=args.device,
                verbose=verbose,
            )
        else:
            # Single file
            segment(
                input=args.input,
                output=args.output,
                fold=fold,
                device=args.device,
                verbose=verbose,
                save_probabilities=args.save_probabilities,
                disable_tta=args.disable_tta,
                ground_truth=ground_truth,
            )

        if verbose:
            print("\nDone!")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
