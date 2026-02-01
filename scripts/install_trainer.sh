#!/bin/bash
# ===========================================
# LumbarSeg - Install Custom nnU-Net Trainer
# ===========================================
#
# This script installs the nnUNetTrainerWandb custom trainer
# into your local nnU-Net v2 installation.
#
# The LumbarSeg model was trained with this custom trainer,
# so you need to install it before running inference.
#
# Usage:
#   ./scripts/install_trainer.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TRAINER_SOURCE="${REPO_ROOT}/nnunetv2_trainers/nnUNetTrainerWandb.py"

echo "=============================================="
echo "LumbarSeg - Installing Custom Trainer"
echo "=============================================="
echo ""

# Find nnU-Net installation
NNUNET_PATH=$(python -c "import nnunetv2; print(nnunetv2.__path__[0])" 2>/dev/null)

if [ -z "$NNUNET_PATH" ]; then
    echo "ERROR: nnU-Net v2 is not installed."
    echo "Please install it first: pip install nnunetv2"
    exit 1
fi

echo "Found nnU-Net at: $NNUNET_PATH"

# Create target directory
TARGET_DIR="${NNUNET_PATH}/training/nnUNetTrainer/variants/training_with_wandb"
mkdir -p "$TARGET_DIR"

# Check if trainer source exists
if [ ! -f "$TRAINER_SOURCE" ]; then
    echo "ERROR: Trainer source not found at: $TRAINER_SOURCE"
    exit 1
fi

# Copy trainer
echo "Installing nnUNetTrainerWandb..."
cp "$TRAINER_SOURCE" "$TARGET_DIR/"

# Create __init__.py if it doesn't exist
INIT_FILE="${TARGET_DIR}/__init__.py"
if [ ! -f "$INIT_FILE" ]; then
    echo "Creating __init__.py..."
    echo "from .nnUNetTrainerWandb import nnUNetTrainerWandb" > "$INIT_FILE"
fi

echo ""
echo "=============================================="
echo "Installation Complete!"
echo "=============================================="
echo ""
echo "Trainer installed to:"
echo "  ${TARGET_DIR}/nnUNetTrainerWandb.py"
echo ""
echo "You can now run inference with:"
echo "  nnUNetv2_predict ... -tr nnUNetTrainerWandb"
echo ""
