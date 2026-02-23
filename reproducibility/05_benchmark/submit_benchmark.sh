#!/bin/bash
#SBATCH --job-name=pgd_05
#SBATCH --partition=p.hpcl91
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=reproducibility/05_benchmark/logs/%A_%a.out
#SBATCH --error=reproducibility/05_benchmark/logs/%A_%a.err
#SBATCH --gres=gpu:h100_pcie_2g.20gb:1
#SBATCH --array=0-15

# Map array index to (dataset, model) pairs
DATASETS=(planar planar planar planar lobster lobster lobster lobster sbm sbm sbm sbm proteins proteins proteins proteins)
MODELS=(AUTOGRAPH DIGRESS GRAN ESGG AUTOGRAPH DIGRESS GRAN ESGG AUTOGRAPH DIGRESS GRAN ESGG AUTOGRAPH DIGRESS GRAN ESGG)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$DATASET model=$MODEL"
echo "Node: $(hostname)"
free -h | head -2

cd /fs/gpfs41/lv11/fileset01/pool/pool-hartout/Documents/Git/polygraph-benchmark

# Use Python directly from pixi environment
export PATH="/fs/gpfs41/lv11/fileset01/pool/pool-hartout/Documents/Git/polygraph-benchmark/.pixi/envs/default/bin:$PATH"
export PYTHONPATH="/fs/gpfs41/lv11/fileset01/pool/pool-hartout/Documents/Git/polygraph-benchmark:$PYTHONPATH"
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

python reproducibility/05_benchmark/compute.py dataset="$DATASET" model="$MODEL"

echo "Task $SLURM_ARRAY_TASK_ID completed with exit code $?"
