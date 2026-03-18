#!/bin/bash
# Submit benchmark computation as a SLURM array job.
#
# Before running, update --partition and --gpus-per-node to match
# your cluster.
#
# Usage:
#   sbatch reproducibility/05_benchmark/submit_benchmark.sh

#SBATCH --job-name=pgd_05
#SBATCH --partition=TODO_SET_YOUR_GPU_PARTITION
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=reproducibility/05_benchmark/logs/%A_%a.out
#SBATCH --error=reproducibility/05_benchmark/logs/%A_%a.err
#SBATCH --gpus-per-node=1
#SBATCH --array=0-15

DATASETS=(planar planar planar planar lobster lobster lobster lobster sbm sbm sbm sbm proteins proteins proteins proteins)
MODELS=(AUTOGRAPH DIGRESS GRAN ESGG AUTOGRAPH DIGRESS GRAN ESGG AUTOGRAPH DIGRESS GRAN ESGG AUTOGRAPH DIGRESS GRAN ESGG)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$DATASET model=$MODEL"

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

export TABPFN_ALLOW_CPU_LARGE_DATASET=1

pixi run python reproducibility/05_benchmark/compute.py \
  dataset="$DATASET" model="$MODEL"

echo "Task $SLURM_ARRAY_TASK_ID completed with exit code $?"
