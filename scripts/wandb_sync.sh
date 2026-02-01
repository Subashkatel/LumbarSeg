#!/bin/bash
# Sync offline W&B runs to the cloud.
# Run this from the login node (which has internet access).
#
# Usage:
#   ./scripts/wandb_sync.sh                    # Sync all runs in nnUNet_results
#   ./scripts/wandb_sync.sh /path/to/run_dir   # Sync specific run


REPO_ROOT=${REPO_ROOT:-/scratch/gpfs/MARTONOSI/sk2415/ml}

module load anaconda3/2024.10
conda activate "${REPO_ROOT}/env"

if [[ $# -ge 1 ]]; then
    # Sync specific directory
    RUN_DIR="$1"
    echo "Syncing W&B run: $RUN_DIR"

    if [[ -d "$RUN_DIR/wandb" ]]; then
        wandb sync "$RUN_DIR/wandb/latest-run"
    else
        echo "No wandb directory found in $RUN_DIR"
        exit 1
    fi
else
    # Sync all runs in nnUNet_results
    RESULTS_DIR="${REPO_ROOT}/nnUNet_results"

    echo "Searching for W&B runs in: $RESULTS_DIR"

    find "$RESULTS_DIR" -type d -name "wandb" | while read wandb_dir; do
        latest_run="$wandb_dir/latest-run"
        if [[ -d "$latest_run" ]]; then
            echo "Syncing: $latest_run"
            wandb sync "$latest_run" || echo "Failed to sync: $latest_run"
        fi
    done

    echo "Done syncing all W&B runs"
fi
