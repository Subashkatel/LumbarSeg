#!/bin/bash
# Sync offline W&B runs to the cloud.
# Run this from the login node (which has internet access).
#
# Usage:
#   ./scripts/wandb_sync.sh                    # Sync all runs in nnUNet_results
#   ./scripts/wandb_sync.sh /path/to/run_dir   # Sync specific run

set -eo pipefail

# Source user configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/../config.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    echo "[error] config.sh not found. Copy config.sh.example to config.sh and edit it."
    exit 1
fi

# Load modules if specified
if [ -n "$CLUSTER_MODULES" ]; then
    module load $CLUSTER_MODULES 2>/dev/null || true
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV" || { echo "[error] Failed to activate conda env."; exit 1; }

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
