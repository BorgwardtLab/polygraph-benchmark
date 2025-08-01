#!/bin/bash

#SBATCH --job-name=denoising-model-quality-jobs
#SBATCH --output=/fs/pool/pool-krimmel/polygraph/.local/logs/job_%A_%a.out
#SBATCH --error=/fs/pool/pool-krimmel/polygraph/.local/logs/job_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH -p p.hpcl91
#SBATCH --gres=gpu:h100_pcie_2g.20gb:1
#SBATCH --array=0-2%3

# Define reference types based on array task ID
metrics=("informedness-adaptive" "jsd" "informedness")
metric=${metrics[$SLURM_ARRAY_TASK_ID]}


python model-quality/evaluate.py \
    --checkpoint-folder "/fs/pool/pool-mlsb/polygraph/model-quality/denoising-iterations/" \
    --metric $metric \
    --reference planar \
    --num-graphs 2048 \
    --filename "${metric}_planar.csv" \
    --skip-assert
