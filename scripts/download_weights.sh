#!/bin/bash
# ===========================================
# LumbarSeg - Download Pre-trained Weights
# ===========================================
#
# Downloads trained model checkpoints from GitHub Releases.
# These are required for running inference with the pre-trained model.
#
# Usage:
#   ./scripts/download_weights.sh
#
# The script will create the following directory structure:
#   nnUNet_results/Dataset001_LumbarMuscle/nnUNetTrainerWandb__nnUNetPlans__3d_fullres/
#   ├── fold_0/checkpoint_final.pth
#   ├── fold_1/checkpoint_final.pth
#   ├── fold_2/checkpoint_final.pth
#   ├── fold_3/checkpoint_final.pth
#   ├── fold_4/checkpoint_final.pth
#   ├── dataset.json
#   └── plans.json

set -e

# Configuration
REPO_OWNER="USERNAME"  # TODO: Replace with your GitHub username
REPO_NAME="LumbarSeg"
RELEASE_TAG="v1.0"
BASE_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/download/${RELEASE_TAG}"

# Output directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${REPO_ROOT}/nnUNet_results/Dataset001_LumbarMuscle/nnUNetTrainerWandb__nnUNetPlans__3d_fullres"

echo "=============================================="
echo "LumbarSeg - Downloading Pre-trained Weights"
echo "=============================================="
echo ""
echo "Repository: ${REPO_OWNER}/${REPO_NAME}"
echo "Release: ${RELEASE_TAG}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Create output directories
for fold in 0 1 2 3 4; do
    mkdir -p "${OUTPUT_DIR}/fold_${fold}"
done

# Download checkpoints for each fold
echo "Downloading model checkpoints..."
for fold in 0 1 2 3 4; do
    CHECKPOINT_URL="${BASE_URL}/fold_${fold}_checkpoint_final.pth"
    CHECKPOINT_PATH="${OUTPUT_DIR}/fold_${fold}/checkpoint_final.pth"

    if [ -f "$CHECKPOINT_PATH" ]; then
        echo "  Fold ${fold}: Already exists, skipping"
    else
        echo "  Fold ${fold}: Downloading..."
        curl -L -o "$CHECKPOINT_PATH" "$CHECKPOINT_URL" || {
            echo "  ERROR: Failed to download fold ${fold}"
            exit 1
        }
    fi
done

# Download configuration files
echo ""
echo "Downloading configuration files..."

for config_file in dataset.json plans.json; do
    CONFIG_URL="${BASE_URL}/${config_file}"
    CONFIG_PATH="${OUTPUT_DIR}/${config_file}"

    if [ -f "$CONFIG_PATH" ]; then
        echo "  ${config_file}: Already exists, skipping"
    else
        echo "  ${config_file}: Downloading..."
        curl -L -o "$CONFIG_PATH" "$CONFIG_URL" || {
            echo "  WARNING: Failed to download ${config_file} (may not be required)"
        }
    fi
done

echo ""
echo "=============================================="
echo "Download Complete!"
echo "=============================================="
echo ""
echo "Model weights are stored in:"
echo "  ${OUTPUT_DIR}"
echo ""
echo "To run inference:"
echo "  export nnUNet_results=${REPO_ROOT}/nnUNet_results"
echo "  export nnUNet_raw=${REPO_ROOT}/nnUNet_raw"
echo "  export nnUNet_preprocessed=${REPO_ROOT}/nnUNet_preprocessed"
echo ""
echo "  nnUNetv2_predict -i INPUT -o OUTPUT -d 001 -c 3d_fullres -tr nnUNetTrainerWandb -f 0 1 2 3 4"
echo ""
