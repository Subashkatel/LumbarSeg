#!/bin/bash
#SBATCH --job-name=lumbarseg-tests
#SBATCH --output=logs/tests_%j.out
#SBATCH --error=logs/tests_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# LumbarSeg Test Suite
# Runs all tests including integration tests with GPU

set -e

echo "=============================================="
echo "LumbarSeg Test Suite"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules
module load anaconda3/2024.10
module load nvhpc/25.5

# Activate environment
conda activate /scratch/gpfs/MARTONOSI/sk2415/ml/env

# Set paths
export nnUNet_raw=/scratch/gpfs/MARTONOSI/sk2415/ml/nnUNet_raw
export nnUNet_preprocessed=/scratch/gpfs/MARTONOSI/sk2415/ml/nnUNet_preprocessed
export nnUNet_results=/scratch/gpfs/MARTONOSI/sk2415/ml/nnUNet_results

# Show GPU info
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Run unit tests first
echo "=============================================="
echo "Running Unit Tests"
echo "=============================================="
python -m pytest tests/ -v --ignore=tests/test_integration.py

echo ""
echo "=============================================="
echo "Running Integration Tests"
echo "=============================================="
python -m pytest tests/test_integration.py -v

echo ""
echo "=============================================="
echo "All Tests Complete!"
echo "=============================================="
echo "End time: $(date)"
