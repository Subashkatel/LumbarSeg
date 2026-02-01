#!/bin/bash
#SBATCH --job-name=nnunet_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00

set -eo pipefail

log(){ printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# Source user configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/../config.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    log "[error] config.sh not found. Copy config.sh.example to config.sh and edit it."
    exit 1
fi

# Configuration
DATASET_ID=${DATASET_ID:-001}
CONFIG=${CONFIG:-3d_fullres}
FOLD=${FOLD:-all}
TRAINER=${TRAINER:-nnUNetTrainerWandb}

# W&B offline mode (no internet on compute nodes)
export WANDB_DIR="${REPO_ROOT}/nnUNet_results"

mkdir -p "$REPO_ROOT/logs"

log "Starting nnU-Net training"
log "  Dataset: $DATASET_ID"
log "  Config: $CONFIG"
log "  Fold: $FOLD"
log "  Trainer: $TRAINER"

module purge
if [ -n "$CLUSTER_MODULES" ]; then
    module load $CLUSTER_MODULES || { log "[error] Failed to load modules."; exit 1; }
fi
[ -n "$CUDA_PATH" ] && export CUDA_PATH
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV" || { log "[error] Failed to activate conda env."; exit 1; }

nvidia-smi || true
log "CUDA devices: ${CUDA_VISIBLE_DEVICES:-not-set}"

log "Running nnUNetv2_train..."

nnUNetv2_train "$DATASET_ID" "$CONFIG" "$FOLD" -tr "$TRAINER"

log "Training complete."
