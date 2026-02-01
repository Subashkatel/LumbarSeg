# LumbarSeg

Automatic segmentation of lumbar paraspinal muscles from MRI using nnU-Net v2.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![nnU-Net v2](https://img.shields.io/badge/nnU--Net-v2.6+-green.svg)](https://github.com/MIC-DKFZ/nnUNet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

LumbarSeg provides automated segmentation of four lumbar paraspinal muscles from MRI:

| Muscle | Abbreviation | Description |
|--------|--------------|-------------|
| Left Erector Spinae | L_ES | Large muscle group lateral to spine |
| Right Erector Spinae | R_ES | Large muscle group lateral to spine |
| Left Multifidus | L_MF | Deep muscle medial to ES |
| Right Multifidus | R_MF | Deep muscle medial to ES |

### Performance

| Metric | L_ES | R_ES | L_MF | R_MF | Macro Avg |
|--------|------|------|------|------|-----------|
| Dice Score | 91.7% | 91.7% | 89.7% | 89.2% | **90.6%** |

Trained on 347 subjects with 5-fold cross-validation using nnU-Net v2.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Subashkatel/LumbarSeg.git
cd LumbarSeg
```

**Option A: Using Conda**
```bash
conda create -n lumbarseg python=3.11
conda activate lumbarseg

# Install PyTorch
# For CUDA (Linux/Windows with NVIDIA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For Mac (CPU or MPS):
pip install torch torchvision

# Install LumbarSeg
pip install .
```

**Option B: Using Pixi**
```bash
pixi init
pixi add python=3.11 pytorch torchvision

# For CUDA support, add:
pixi add pytorch-cuda=12.1 -c pytorch -c nvidia

# Install LumbarSeg
pixi run pip install .

# Run commands with pixi
pixi run lumbarseg -i scan.nii.gz -o seg.nii.gz
```

### Run Segmentation

**That's it! Just two commands:**

```bash
# Single file
lumbarseg -i scan.nii.gz -o segmentation.nii.gz

# Batch processing (entire folder)
lumbarseg -i scans_folder/ -o results_folder/
```

Model weights are downloaded automatically on first use (~2.5 GB).

### Python API

```python
from lumbarseg import segment

# Single file
segment("scan.nii.gz", "segmentation.nii.gz")

# With options
segment("scan.nii.gz", "seg.nii.gz", fold=0, device="cpu")
```

### CLI Options

```bash
lumbarseg -i INPUT -o OUTPUT [options]

Required:
  -i, --input      Input NIfTI file or directory
  -o, --output     Output file or directory

Options:
  --fast           Use single fold for faster inference (~90% Dice)
  -d, --device     Device: cuda, cpu, or mps (auto-detected)
  -f, --fold       Fold(s): 0-4 or 'all' for ensemble (default: all)
  -q, --quiet      Suppress output messages
```

### Advanced: Manual nnU-Net Usage

For researchers who want direct nnU-Net access:

```bash
# Install custom trainer
./scripts/install_trainer.sh

# Download weights manually
./scripts/download_weights.sh

# Set environment variables
export nnUNet_results=/path/to/LumbarSeg/nnUNet_results
export nnUNet_raw=/path/to/LumbarSeg/nnUNet_raw
export nnUNet_preprocessed=/path/to/LumbarSeg/nnUNet_preprocessed

# Run nnU-Net directly
nnUNetv2_predict \
    -i /path/to/images \
    -o /path/to/output \
    -d 001 -c 3d_fullres -tr nnUNetTrainerWandb -f 0 1 2 3 4
```

## Input Requirements

- **Format**: NIfTI (.nii or .nii.gz)
- **Modality**: T1-weighted or T2-weighted MRI of lumbar spine
- **Naming**: Any filename (the CLI handles nnU-Net naming conventions automatically)
- **Orientation**: Any orientation (nnU-Net handles reorientation)

Example:
```
input_folder/
├── patient001.nii.gz
├── patient002.nii.gz
└── patient003.nii.gz
```

## Output

Predictions are saved as multi-label NIfTI files with values:
- `0`: Background
- `1`: Left Erector Spinae (L_ES)
- `2`: Right Erector Spinae (R_ES)
- `3`: Left Multifidus (L_MF)
- `4`: Right Multifidus (R_MF)

## Project Structure

```
LumbarSeg/
├── lumbarseg/                  # Main Python package
│   ├── __init__.py             # Package init with segment() function
│   ├── python_api.py           # Python API implementation
│   ├── cli.py                  # Command-line interface
│   └── config.py               # Configuration and constants
├── nnunetv2_trainers/          # Custom nnU-Net trainer
│   └── nnUNetTrainerWandb.py   # Trainer with W&B logging
├── scripts/                    # Training and utility scripts
│   ├── install_trainer.sh      # Install custom trainer
│   ├── download_weights.sh     # Download pre-trained weights
│   ├── train_nnunet.sbatch     # Single-fold training (SLURM)
│   ├── train_nnunet_5fold.sbatch # 5-fold cross-validation (SLURM)
│   ├── eval_crossval.sbatch    # Cross-validation evaluation (SLURM)
│   └── eval_ensemble.sbatch    # Ensemble evaluation (SLURM)
├── evaluation/                 # Evaluation utilities
│   ├── compute_metrics.py      # Dice, HD95, ASSD metrics
│   └── generate_visualizations.py # GT vs Pred overlays
├── setup.py                    # Package installation
├── requirements.txt
├── CLAUDE.md                   # Developer documentation
└── README.md
```

## Training Your Own Model

### Data Preparation

1. Organize your data:
```
data/
├── subject001/
│   ├── image.nii.gz           # MRI volume
│   └── label.nii.gz           # Multi-label mask (0-4)
├── subject002/
│   └── ...
```

2. Convert to nnU-Net format:
```bash
python scripts/convert_to_nnunet.py \
    --input-dir data/ \
    --output-dir nnUNet_raw/Dataset001_LumbarMuscle
```

### Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity -c 3d_fullres
```

### Training

```bash
# Train all 5 folds (recommended)
for fold in 0 1 2 3 4; do
    nnUNetv2_train 001 3d_fullres $fold -tr nnUNetTrainer
done

# Or use SLURM for cluster training
sbatch scripts/train_nnunet_5fold.sbatch
```

## Evaluation

### Compute Metrics

```bash
python evaluation/compute_metrics.py \
    --pred-dir predictions/ \
    --gt-dir ground_truth/ \
    --output metrics.csv
```

### Generate Visualizations

```bash
python evaluation/generate_visualizations.py \
    --image-dir images/ \
    --gt-dir ground_truth/ \
    --pred-dir predictions/ \
    --output-dir visualizations/
```

Generates side-by-side comparisons with color-coded overlays:
- **Red**: Left Erector Spinae
- **Blue**: Right Erector Spinae
- **Green**: Left Multifidus
- **Yellow**: Right Multifidus

## Citation

If you use LumbarSeg in your research, please cite:

```bibtex
@software{lumbarseg2026,
  author = {Your Name},
  title = {LumbarSeg: Automatic Lumbar Paraspinal Muscle Segmentation},
  year = {2026},
  url = {https://github.com/Subashkatel/LumbarSeg}
}
```

## References

- [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) - Isensee et al., Nature Methods 2021
- [Lumbar muscle segmentation review](https://www.sciencedirect.com/science/article/pii/S2950363924000206)
- [Paraspinal muscle segmentation](https://onlinelibrary.wiley.com/doi/full/10.1002/jsp2.70003)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- nnU-Net team at DKFZ for the self-configuring segmentation framework
- Princeton University Research Computing for HPC resources
