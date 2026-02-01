#!/bin/bash
#SBATCH --job-name=nnunet_5fold
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/gpfs/MARTONOSI/sk2415/ml/logs/train_fold%a_%j.out
#SBATCH --error=/scratch/gpfs/MARTONOSI/sk2415/ml/logs/train_fold%a_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=sk2415@princeton.edu

set -eo pipefail

log(){ printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

REPO_ROOT=${REPO_ROOT:-/scratch/gpfs/MARTONOSI/sk2415/ml}
DATASET_ID=${DATASET_ID:-001}
CONFIG=${CONFIG:-3d_fullres}
TRAINER=${TRAINER:-nnUNetTrainerWandb}
FOLD=${SLURM_ARRAY_TASK_ID}

# Set nnU-Net environment variables
export nnUNet_raw="${REPO_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${REPO_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${REPO_ROOT}/nnUNet_results"

# W&B offline mode (no internet on compute nodes)
export WANDB_MODE=offline
export WANDB_PROJECT=${WANDB_PROJECT:-nnunet-lumbar-muscle}
export WANDB_DIR="${REPO_ROOT}/nnUNet_results"

mkdir -p "$REPO_ROOT/logs"

log "Starting nnU-Net training"
log "  Dataset: $DATASET_ID"
log "  Config: $CONFIG"
log "  Fold: $FOLD"
log "  Trainer: $TRAINER"

module purge
module load anaconda3/2024.10 nvhpc/25.5 openmpi/cuda-12.9/nvhpc-25.5/4.1.8 || { log "[error] Failed to load modules."; exit 1; }
export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9
eval "$(conda shell.bash hook)"
conda activate /scratch/gpfs/MARTONOSI/sk2415/ml/env || { log "[error] Failed to activate conda env."; exit 1; }

nvidia-smi || true
log "CUDA devices: ${CUDA_VISIBLE_DEVICES:-not-set}"

log "Running nnUNetv2_train for fold $FOLD..."

nnUNetv2_train "$DATASET_ID" "$CONFIG" "$FOLD" -tr "$TRAINER"

log "Fold $FOLD training complete."
