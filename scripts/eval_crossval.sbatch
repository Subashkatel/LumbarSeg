#!/bin/bash
#SBATCH --job-name=nnunet_cv_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

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
TRAINER=${TRAINER:-nnUNetTrainerWandb}

# Directories
IMAGES_DIR="${REPO_ROOT}/nnUNet_raw/Dataset001_LumbarMuscle/imagesTr"
LABELS_DIR="${REPO_ROOT}/nnUNet_raw/Dataset001_LumbarMuscle/labelsTr"
SPLITS_FILE="${REPO_ROOT}/nnUNet_preprocessed/Dataset001_LumbarMuscle/splits_final.json"
OUTPUT_BASE="${REPO_ROOT}/predictions_cv"
VIZ_DIR="${OUTPUT_BASE}/visualizations"

mkdir -p "$REPO_ROOT/logs" "$OUTPUT_BASE" "$VIZ_DIR"

log "=============================================="
log "nnU-Net Cross-Validation Evaluation Pipeline"
log "=============================================="
log "  Dataset: $DATASET_ID"
log "  Config: $CONFIG"
log "  Trainer: $TRAINER"
log "  Output: $OUTPUT_BASE"
log ""

# Load modules and activate environment
module purge
if [ -n "$CLUSTER_MODULES" ]; then
    module load $CLUSTER_MODULES || { log "[error] Failed to load modules."; exit 1; }
fi
[ -n "$CUDA_PATH" ] && export CUDA_PATH
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV" || { log "[error] Failed to activate conda env."; exit 1; }

nvidia-smi || true

# =============================================================================
# Step 1: Run inference for each fold on its validation set
# =============================================================================
log ""
log "Step 1: Running per-fold inference on validation sets"
log "------------------------------------------------------"

COMBINED_DIR="${OUTPUT_BASE}/combined"
mkdir -p "$COMBINED_DIR"

for fold in 0 1 2 3 4; do
    log ""
    log "Processing Fold $fold..."

    # Create temp dir for this fold's validation images
    FOLD_INPUT="${OUTPUT_BASE}/temp_val_fold${fold}"
    FOLD_OUTPUT="${OUTPUT_BASE}/fold_${fold}"

    mkdir -p "$FOLD_INPUT" "$FOLD_OUTPUT"

    # Extract validation cases for this fold
    python "${REPO_ROOT}/scripts/extract_val_cases.py" \
        --splits-file "$SPLITS_FILE" \
        --images-dir "$IMAGES_DIR" \
        --output-dir "$FOLD_INPUT" \
        --fold "$fold"

    # Count validation cases
    n_cases=$(ls -1 "$FOLD_INPUT"/*.nii.gz 2>/dev/null | wc -l)
    log "  Found $n_cases validation cases for fold $fold"

    if [ "$n_cases" -eq 0 ]; then
        log "  [warning] No validation cases found, skipping fold $fold"
        continue
    fi

    # Run inference with TTA (enabled by default)
    log "  Running nnUNetv2_predict with TTA..."
    nnUNetv2_predict \
        -i "$FOLD_INPUT" \
        -o "$FOLD_OUTPUT" \
        -d "$DATASET_ID" \
        -c "$CONFIG" \
        -tr "$TRAINER" \
        -f "$fold"

    # Copy predictions to combined directory (exclude probability maps)
    log "  Copying predictions to combined directory..."
    for pred_file in "$FOLD_OUTPUT"/*.nii.gz; do
        # Skip probability maps
        [[ "$pred_file" == *_probs.nii.gz ]] && continue
        cp "$pred_file" "$COMBINED_DIR"/ 2>/dev/null || true
    done

    # Cleanup temp input directory
    rm -rf "$FOLD_INPUT"

    log "  Fold $fold complete"
done

# Count total predictions
n_total=$(ls -1 "$COMBINED_DIR"/*.nii.gz 2>/dev/null | wc -l)
log ""
log "Combined predictions: $n_total cases"

# =============================================================================
# Step 2: Compute metrics on combined predictions
# =============================================================================
log ""
log "Step 2: Computing evaluation metrics"
log "-------------------------------------"

python "${REPO_ROOT}/evaluation/compute_metrics.py" \
    --pred-dir "$COMBINED_DIR" \
    --gt-dir "$LABELS_DIR" \
    --output "${OUTPUT_BASE}/metrics.csv" \
    --workers 8 \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "cv_eval_${CONFIG}_${SLURM_JOB_ID}"

# =============================================================================
# Step 3: Generate visualizations
# =============================================================================
log ""
log "Step 3: Generating visualizations"
log "----------------------------------"

python "${REPO_ROOT}/evaluation/generate_visualizations.py" \
    --image-dir "$IMAGES_DIR" \
    --gt-dir "$LABELS_DIR" \
    --pred-dir "$COMBINED_DIR" \
    --output-dir "$VIZ_DIR" \
    --n-slices 3 \
    --montage \
    --workers 8

# =============================================================================
# Summary
# =============================================================================
log ""
log "=============================================="
log "Evaluation Complete!"
log "=============================================="
log ""
log "Output files:"
log "  Predictions: $COMBINED_DIR"
log "  Metrics CSV: ${OUTPUT_BASE}/metrics.csv"
log "  Metrics JSON: ${OUTPUT_BASE}/metrics.json"
log "  Visualizations: $VIZ_DIR"
log ""
log "To sync W&B runs:"
log "  ./scripts/wandb_sync.sh"
log ""

# Print summary metrics
if [ -f "${OUTPUT_BASE}/metrics.json" ]; then
    log "Summary metrics:"
    cat "${OUTPUT_BASE}/metrics.json"
fi
