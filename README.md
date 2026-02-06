# LumbarSeg

Automatic segmentation of lumbar paraspinal muscles from MRI using nnU-Net v2.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![nnU-Net v2](https://img.shields.io/badge/nnU--Net-v2.6+-green.svg)](https://github.com/MIC-DKFZ/nnUNet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

LumbarSeg provides automated segmentation of four lumbar paraspinal muscles from MRI:

| Muscle | Label | Color |
|--------|-------|-------|
| Left Erector Spinae | L_ES (1) | Red |
| Right Erector Spinae | R_ES (2) | Blue |
| Left Multifidus | L_MF (3) | Green |
| Right Multifidus | R_MF (4) | Yellow |

### Performance

| Metric | L_ES | R_ES | L_MF | R_MF | Macro Avg |
|--------|------|------|------|------|-----------|
| Dice Score | 91.7% | 91.7% | 89.7% | 89.2% | **90.6%** |

Trained on 347 subjects with 5-fold cross-validation using nnU-Net v2.

## Quick Start

### Installation

```bash
git clone https://github.com/Subashkatel/LumbarSeg.git
cd LumbarSeg
```

**Option A: Using Conda**
```bash
conda create -n lumbarseg python=3.11
conda activate lumbarseg

# Install PyTorch (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For Mac (CPU only - MPS not supported by nnU-Net):
pip install torch torchvision

# Install LumbarSeg
pip install .
```

**Option B: Using Pixi**
```bash
pixi init
pixi add python=3.11 pytorch torchvision
pixi add pytorch-cuda=12.1 -c pytorch -c nvidia  # For CUDA
pixi run pip install .
```

**Verify Installation:**
```bash
python -m pytest tests/ -v -m "not integration"
```

### Run Segmentation

```bash
# Basic usage: -i is the path to your input NIfTI file
lumbarseg -i /path/to/scan.nii.gz

# Specify where to save output: -o is the path to the output directory
lumbarseg -i /path/to/scan.nii.gz -o /path/to/results/

# CPU or Mac: use --fast and --disable-tta for faster inference (~3min instead of ~20min)
lumbarseg -i /path/to/scan.nii.gz --fast --disable-tta

# CPU or Mac: full accuracy but slower (~20min)
lumbarseg -i /path/to/scan.nii.gz --device cpu

# Batch processing: -i can also be a directory of NIfTI files
lumbarseg -i /path/to/scans_folder/ -o /path/to/results_folder/
```

Model weights are downloaded automatically on first use (~2.5 GB).

### Output

By default, the output folder is named after the input file with `_segmented` appended (e.g., `scan.nii.gz` produces `scan_segmented/`). Use `-o` to specify a custom output directory.

Every run produces an output folder with:

```
scan_segmented/
├── segmentation.nii.gz    # Multi-label segmentation (0-4)
├── L_ES.nii.gz            # Binary mask - Left Erector Spinae
├── R_ES.nii.gz            # Binary mask - Right Erector Spinae
├── L_MF.nii.gz            # Binary mask - Left Multifidus
├── R_MF.nii.gz            # Binary mask - Right Multifidus
├── preview.png            # 8-slice segmentation overlay
└── metrics.csv            # Evaluation metrics (only with --gt)
```

### Ground Truth Comparison

Compare predictions against manual segmentations:

```bash
# Separate binary masks
lumbarseg -i scan.nii.gz --gt L_ES.nii R_ES.nii L_Mult.nii R_Mult.nii

# Single multi-label mask
lumbarseg -i scan.nii.gz --gt ground_truth.nii.gz
```

Output includes a metrics table:

```
Evaluation Results:
==============================================================================
Muscle         Dice  Jaccard  HD95(mm)  ASSD(mm)  Precision   Recall
------------------------------------------------------------------------------
L_ES          95.2%    90.8%     3.21      0.45      96.7%    93.7%
R_ES          94.8%    90.1%     3.54      0.51      96.0%    93.6%
L_MF          93.2%    87.3%     4.12      0.62      93.8%    92.7%
R_MF          93.3%    87.4%     3.89      0.58      94.9%    91.6%
------------------------------------------------------------------------------
Mean          94.1%    88.9%     3.69      0.54      95.4%    92.9%
```

### CLI Options

```
lumbarseg -i INPUT [-o OUTPUT] [options]

Required:
  -i, --input          Input NIfTI file or directory

Optional:
  -o, --output         Output directory (default: <input>_segmented/)
  --gt FILE [FILE...]  Ground truth mask(s) for evaluation
  --fast               Use single fold for faster inference (~90% Dice)
  -d, --device         Device: cuda, cpu (default: auto-detect)
  -f, --fold           Fold(s): 0-4 or 'all' (default: all)
  --disable-tta        Disable test-time augmentation (8x faster)
  --save-probabilities Save probability maps
  -q, --quiet          Suppress output messages
  -v, --version        Show version
```

## Input Requirements

- **Format**: NIfTI (.nii or .nii.gz)
- **Modality**: T1/T2-weighted Inculding Water Fat MRI of lumbar spine
- **Orientation**: Any orientation (auto-reoriented to RAS before inference)

## Platform Support

| Platform | GPU | CPU | Estimated Time (per scan) |
|----------|-----|-----|---------------------------|
| Linux | CUDA | Yes | ~35s (GPU) / ~20min (CPU) |
| macOS | No (MPS disabled) | Yes | ~20min (CPU only) |
| Windows | CUDA | Yes | ~35s (GPU) / ~20min (CPU) |

> **Note:** For optimal results, an NVIDIA GPU with CUDA support is strongly recommended.
> CPU inference works but is significantly slower (~30x). On macOS, only CPU is available
> because nnU-Net does not support Apple's MPS backend. Use `--fast` and `--disable-tta`
> flags to reduce CPU inference time at the cost of slightly lower accuracy.

## Project Structure

```
LumbarSeg/
├── lumbarseg/                  # Python package
│   ├── __init__.py
│   ├── python_api.py           # Core API (segment, evaluate, etc.)
│   ├── cli.py                  # Command-line interface
│   └── config.py               # Configuration and constants
├── tests/                      # Test suite
├── evaluation/                 # Standalone evaluation utilities
├── scripts/                    # Training/SLURM scripts
├── setup.py                    # Package installation
└── README.md
```

### Python API

```python
from lumbarseg import segment, evaluate

# Basic usage
segment("scan.nii.gz", "results/")

# With options
segment("scan.nii.gz", "results/", fold=0, device="cpu")

# With ground truth evaluation
segment("scan.nii.gz", "results/",
        ground_truth=["L_ES.nii", "R_ES.nii", "L_Mult.nii", "R_Mult.nii"])

# Evaluate separately
results = evaluate(
    prediction="results/segmentation.nii.gz",
    ground_truth=["L_ES.nii", "R_ES.nii", "L_Mult.nii", "R_Mult.nii"],
    output_path="results/metrics.csv"
)
```

## Citation

<!-- ```bibtex
@software{lumbarseg2026,
  author = {Subash Katel},
  title = {LumbarSeg: Automatic Lumbar Paraspinal Muscle Segmentation},
  year = {2026},
  url = {https://github.com/Subashkatel/LumbarSeg}
}
``` -->

## References

- [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) - Isensee et al., Nature Methods 2021
- [Lumbar muscle segmentation review](https://www.sciencedirect.com/science/article/pii/S2950363924000206)
- [Paraspinal muscle segmentation](https://onli.nelibrary.wiley.com/doi/full/10.1002/jsp2.70003)

## License

MIT License - see [LICENSE](LICENSE) for details.
