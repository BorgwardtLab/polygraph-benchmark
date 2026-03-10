#!/bin/bash
#SBATCH --job-name=model_quality
#SBATCH --output=logs/model_quality_%A_%a.out
#SBATCH --error=logs/model_quality_%A_%a.err
#SBATCH --cpus-per-task=10
#SBATCH --mem=192G
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH -p p.hpcl91,p.hpcl94g,p.hpcl94c
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-3%4

# 4 jobs: {sbm,lobster} x {jsd,informedness}
datasets=("sbm" "sbm" "lobster" "lobster")
variants=("jsd" "informedness" "jsd" "informedness")

dataset=${datasets[$SLURM_ARRAY_TASK_ID]}
variant=${variants[$SLURM_ARRAY_TASK_ID]}

echo "Running: dataset=$dataset variant=$variant"
cd /fs/pool/pool-hartout/Documents/Git/polygraph-benchmark/reproducibility

export TABPFN_ALLOW_CPU_LARGE_DATASET=1

pixi run python 03_model_quality/compute.py \
    curve_type=training \
    dataset=$dataset \
    variant=$variant \
    num_graphs=2048
