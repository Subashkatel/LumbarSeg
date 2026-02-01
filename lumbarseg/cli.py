#!/usr/bin/env python3
"""
LumbarSeg - Command Line Interface

Usage:
    lumbarseg -i scan.nii.gz -o segmentation.nii.gz
    lumbarseg -i scans/ -o results/
"""

import argparse
import sys
from pathlib import Path

from . import __version__
from .python_api import segment, segment_batch, detect_device, generate_preview, split_segmentation
from .config import LABELS, LABEL_NAMES


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

Examples:
  lumbarseg -i scan.nii.gz -o segmentation.nii.gz
  lumbarseg -i scan.nii.gz -o seg.nii.gz --preview
  lumbarseg -i scans_folder/ -o results_folder/
  lumbarseg -i scan.nii.gz -o seg.nii.gz --fast
  lumbarseg -i scan.nii.gz -o seg.nii.gz --device cpu
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
        required=True,
        help="Output segmentation file or directory",
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
        choices=["cuda", "cpu", "mps"],
        default=None,
        help="Device for inference (default: auto-detect)",
    )
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Save probability maps",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate a PNG preview image showing the segmentation overlay",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Output separate binary mask files for each muscle class (L_ES, R_ES, L_MF, R_MF)",
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
            )

            # Split into separate masks if requested
            if args.split:
                split_files = split_segmentation(
                    segmentation=args.output,
                    verbose=verbose,
                )
                if verbose and split_files:
                    print(f"Split masks saved:")
                    for label_name, path in split_files.items():
                        print(f"  {label_name}: {path}")

            # Generate preview if requested
            if args.preview:
                preview_path = generate_preview(
                    input_image=args.input,
                    segmentation=args.output,
                    verbose=verbose,
                )
                if verbose and preview_path:
                    print(f"Preview saved to: {preview_path}")

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
