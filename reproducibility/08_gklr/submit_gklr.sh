#!/bin/bash
#SBATCH --job-name=pgd_08
#SBATCH --partition=p.hpcl94c
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=reproducibility/08_gklr/logs/%A_%a.out
#SBATCH --error=reproducibility/08_gklr/logs/%A_%a.err
#SBATCH --array=0-15%4

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

python reproducibility/08_gklr/run_single.py --dataset "$DATASET" --model "$MODEL"

echo "Task $SLURM_ARRAY_TASK_ID completed with exit code $?"
