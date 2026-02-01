#!/bin/bash
#SBATCH --job-name=nnunet_ensemble
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/MARTONOSI/sk2415/ml/logs/ensemble_%j.out
#SBATCH --error=/scratch/gpfs/MARTONOSI/sk2415/ml/logs/ensemble_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=sk2415@princeton.edu

set -eo pipefail

log(){ printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# Configuration
REPO_ROOT=${REPO_ROOT:-/scratch/gpfs/MARTONOSI/sk2415/ml}
DATASET_ID=${DATASET_ID:-001}
CONFIG=${CONFIG:-3d_fullres}
TRAINER=${TRAINER:-nnUNetTrainerWandb}

# Directories
INPUT_DIR="${REPO_ROOT}/nnUNet_raw/Dataset001_LumbarMuscle/imagesTr"
LABELS_DIR="${REPO_ROOT}/nnUNet_raw/Dataset001_LumbarMuscle/labelsTr"
OUTPUT_DIR="${REPO_ROOT}/predictions_ensemble"
VIZ_DIR="${OUTPUT_DIR}/visualizations"

# nnU-Net environment
export nnUNet_raw="${REPO_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${REPO_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${REPO_ROOT}/nnUNet_results"

# W&B offline mode
export WANDB_MODE=offline
export WANDB_PROJECT=${WANDB_PROJECT:-nnunet-lumbar-muscle}

mkdir -p "$REPO_ROOT/logs" "$OUTPUT_DIR" "$VIZ_DIR"

log "=============================================="
log "nnU-Net Ensemble Evaluation (All 5 Folds + TTA)"
log "=============================================="
log "  Dataset: $DATASET_ID"
log "  Config: $CONFIG"
log "  Trainer: $TRAINER"
log "  Input: $INPUT_DIR"
log "  Output: $OUTPUT_DIR"
log ""

# Load modules and activate environment
module purge
module load anaconda3/2024.10 nvhpc/25.5 openmpi/cuda-12.9/nvhpc-25.5/4.1.8 || { log "[error] Failed to load modules."; exit 1; }
export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9
eval "$(conda shell.bash hook)"
conda activate /scratch/gpfs/MARTONOSI/sk2415/ml/env || { log "[error] Failed to activate conda env."; exit 1; }

nvidia-smi || true

# =============================================================================
# Step 1: Run ensemble inference (all 5 folds + TTA)
# =============================================================================
log ""
log "Step 1: Running ensemble inference (5 folds + TTA)"
log "---------------------------------------------------"
log "This uses all 5 trained models and averages their predictions."
log "TTA (test-time augmentation) is enabled by default."
log ""

nnUNetv2_predict \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -d "$DATASET_ID" \
    -c "$CONFIG" \
    -tr "$TRAINER" \
    -f 0 1 2 3 4

log "Inference complete."

# Count predictions
n_preds=$(ls -1 "$OUTPUT_DIR"/*.nii.gz 2>/dev/null | grep -v "_probs" | wc -l)
log "Generated $n_preds predictions"

# =============================================================================
# Step 2: Compute metrics
# =============================================================================
log ""
log "Step 2: Computing evaluation metrics"
log "-------------------------------------"

python "${REPO_ROOT}/evaluation/compute_metrics.py" \
    --pred-dir "$OUTPUT_DIR" \
    --gt-dir "$LABELS_DIR" \
    --output "${OUTPUT_DIR}/metrics.csv" \
    --workers 8 \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "ensemble_eval_${CONFIG}_${SLURM_JOB_ID}"

# =============================================================================
# Step 3: Generate visualizations
# =============================================================================
log ""
log "Step 3: Generating visualizations"
log "----------------------------------"

python "${REPO_ROOT}/evaluation/generate_visualizations.py" \
    --image-dir "$INPUT_DIR" \
    --gt-dir "$LABELS_DIR" \
    --pred-dir "$OUTPUT_DIR" \
    --output-dir "$VIZ_DIR" \
    --n-slices 3 \
    --montage \
    --workers 8

# =============================================================================
# Summary
# =============================================================================
log ""
log "=============================================="
log "Ensemble Evaluation Complete!"
log "=============================================="
log ""
log "Output files:"
log "  Predictions: $OUTPUT_DIR"
log "  Metrics CSV: ${OUTPUT_DIR}/metrics.csv"
log "  Metrics JSON: ${OUTPUT_DIR}/metrics.json"
log "  Visualizations: $VIZ_DIR"
log ""

# Print comparison with per-fold results
log "Comparison with per-fold cross-validation:"
log "-------------------------------------------"
if [ -f "${REPO_ROOT}/predictions_cv/metrics.json" ]; then
    log "Per-fold CV results:"
    cat "${REPO_ROOT}/predictions_cv/metrics.json" | grep -A4 "dice_macro"
fi
log ""
log "Ensemble results:"
if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
    cat "${OUTPUT_DIR}/metrics.json"
fi

log ""
log "To sync W&B runs: ./scripts/wandb_sync.sh"
