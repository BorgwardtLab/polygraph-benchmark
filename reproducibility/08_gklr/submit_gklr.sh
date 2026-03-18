#!/bin/bash
# Submit GKLR computation as a SLURM array job.
#
# Before running, update --partition to match your cluster.
#
# Usage:
#   sbatch reproducibility/08_gklr/submit_gklr.sh

#SBATCH --job-name=pgd_08
#SBATCH --partition=TODO_SET_YOUR_CPU_PARTITION
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=reproducibility/08_gklr/logs/%A_%a.out
#SBATCH --error=reproducibility/08_gklr/logs/%A_%a.err
#SBATCH --array=0-15%4

DATASETS=(planar planar planar planar lobster lobster lobster lobster sbm sbm sbm sbm proteins proteins proteins proteins)
MODELS=(AUTOGRAPH DIGRESS GRAN ESGG AUTOGRAPH DIGRESS GRAN ESGG AUTOGRAPH DIGRESS GRAN ESGG AUTOGRAPH DIGRESS GRAN ESGG)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$DATASET model=$MODEL"

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

pixi run python reproducibility/08_gklr/run_single.py \
  --dataset "$DATASET" --model "$MODEL"

echo "Task $SLURM_ARRAY_TASK_ID completed with exit code $?"
