#!/bin/bash
#SBATCH --job-name=nnunet_preprocess
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/MARTONOSI/sk2415/ml/logs/preprocess_%j.out
#SBATCH --error=/scratch/gpfs/MARTONOSI/sk2415/ml/logs/preprocess_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sk2415@princeton.edu

set -eo pipefail

log(){ printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

REPO_ROOT=${REPO_ROOT:-/scratch/gpfs/MARTONOSI/sk2415/ml}
DATASET_ID=${DATASET_ID:-001}

# Set nnU-Net environment variables
export nnUNet_raw="${REPO_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${REPO_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${REPO_ROOT}/nnUNet_results"

mkdir -p "$REPO_ROOT/logs"

log "Starting nnU-Net preprocessing for Dataset${DATASET_ID}"
log "nnUNet_raw: $nnUNet_raw"
log "nnUNet_preprocessed: $nnUNet_preprocessed"

module purge
module load anaconda3/2024.10 || { log "[error] Failed to load modules."; exit 1; }
eval "$(conda shell.bash hook)"
conda activate /scratch/gpfs/MARTONOSI/sk2415/ml/env || { log "[error] Failed to activate conda env."; exit 1; }

log "Running nnUNetv2_plan_and_preprocess..."

nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity -c 3d_fullres

log "Preprocessing complete."
