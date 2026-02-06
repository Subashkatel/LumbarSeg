"""
LumbarSeg - Configuration and constants.
"""

import os
from pathlib import Path

# Model configuration
DATASET_ID = "001"
DATASET_NAME = "Dataset001_LumbarMuscle"
CONFIG = "3d_fullres"
TRAINER = "nnUNetTrainerWandb"
PLANS = "nnUNetPlans"

# Model weights info (for auto-download)
GITHUB_REPO = "Subashkatel/LumbarSeg"
RELEASE_TAG = "v1.1"
MODEL_FILES = {
    "fold_0": "fold_0_checkpoint_final.pth",
    "fold_1": "fold_1_checkpoint_final.pth",
    "fold_2": "fold_2_checkpoint_final.pth",
    "fold_3": "fold_3_checkpoint_final.pth",
    "fold_4": "fold_4_checkpoint_final.pth",
}

# Label mapping
LABEL_NAMES = {
    0: "BG",
    1: "L_ES",
    2: "R_ES",
    3: "L_MF",
    4: "R_MF",
}


def get_lumbarseg_home() -> Path:
    """Get LumbarSeg home directory for storing models and config."""
    # Check environment variable first
    if "LUMBARSEG_HOME" in os.environ:
        home = Path(os.environ["LUMBARSEG_HOME"])
    else:
        # Default to ~/.lumbarseg
        home = Path.home() / ".lumbarseg"

    home.mkdir(parents=True, exist_ok=True)
    return home


def get_weights_dir() -> Path:
    """Get directory for storing model weights."""
    weights_dir = get_lumbarseg_home() / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir


def get_nnunet_results_dir() -> Path:
    """Get nnU-Net results directory structure."""
    weights_dir = get_weights_dir()

    # Create nnU-Net expected structure
    results_dir = weights_dir / DATASET_NAME / f"{TRAINER}__{PLANS}__{CONFIG}"
    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir
