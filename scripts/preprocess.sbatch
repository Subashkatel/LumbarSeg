#!/bin/bash
#SBATCH --job-name=nnunet_preprocess
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

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

DATASET_ID=${DATASET_ID:-001}

mkdir -p "$REPO_ROOT/logs"

log "Starting nnU-Net preprocessing for Dataset${DATASET_ID}"
log "nnUNet_raw: $nnUNet_raw"
log "nnUNet_preprocessed: $nnUNet_preprocessed"

module purge
if [ -n "$CLUSTER_MODULES" ]; then
    module load $CLUSTER_MODULES || { log "[error] Failed to load modules."; exit 1; }
fi
[ -n "$CUDA_PATH" ] && export CUDA_PATH
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV" || { log "[error] Failed to activate conda env."; exit 1; }

log "Running nnUNetv2_plan_and_preprocess..."

nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity -c 3d_fullres

log "Preprocessing complete."
