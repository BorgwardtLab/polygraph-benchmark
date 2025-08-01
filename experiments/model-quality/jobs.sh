#!/bin/bash

#SBATCH --job-name=informedness-jobs
#SBATCH --output=/fs/pool/pool-krimmel/polygraph/.local/logs/job_%A_%a.out
#SBATCH --error=/fs/pool/pool-krimmel/polygraph/.local/logs/job_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH -p p.hpcl91
#SBATCH --gres=gpu:h100_pcie_2g.20gb:1
#SBATCH --array=0-8%9

# Define reference types based on array task ID
references=("sbm" "planar" "lobster")
metrics=("informedness-adaptive" "jsd" "informedness")
metric=${metrics[$((SLURM_ARRAY_TASK_ID / 3))]}
reference=${references[$SLURM_ARRAY_TASK_ID % 3]}

# Define checkpoint folders for each reference type
declare -A checkpoint_folders
checkpoint_folders["sbm"]="/fs/pool/pool-mlsb/polygraph/digress-samples/sbm-procedural"
checkpoint_folders["planar"]="/fs/pool/pool-mlsb/polygraph/digress-samples/planar-procedural"
checkpoint_folders["lobster"]="/fs/pool/pool-mlsb/polygraph/digress-samples/lobster-procedural"

python model-quality/evaluate.py \
    --checkpoint-folder "${checkpoint_folders[$reference]}" \
    --metric $metric \
    --reference $reference \
    --num-graphs 2048 \
    --filename "${metric}_${reference}.csv"
