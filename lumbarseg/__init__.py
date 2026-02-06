"""
LumbarSeg - Lumbar Paraspinal Muscle Segmentation

Automatic segmentation of lumbar paraspinal muscles from MRI scans using nnU-Net v2.

Segments 4 muscle classes:
  - Label 1: Left Erector Spinae (L_ES)
  - Label 2: Right Erector Spinae (R_ES)
  - Label 3: Left Multifidus (L_MF)
  - Label 4: Right Multifidus (R_MF)

Quick Start:
    # CLI
    lumbarseg -i scan.nii.gz -o results/

    # Python API
    from lumbarseg import segment
    segment("scan.nii.gz", "results/")
"""

__version__ = "1.0.0"
__author__ = "LumbarSeg Team"

from .python_api import segment, evaluate

__all__ = ["segment", "evaluate", "__version__"]
