#!/bin/bash

#SBATCH --job-name=perturbation-experiments
#SBATCH --output=./perturbation/results_full/job_%A_%a.out
#SBATCH --error=./perturbation/results_full/job_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH -p p.hpcl94g
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-24%5

# Define the datasets array
datasets=(planar sbm lobster proteins ego)
perturbations=(edge_rewiring edge_swapping edge_deletion edge_addition mixing)

# Calculate dataset and perturbation indices
dataset_idx=$((SLURM_ARRAY_TASK_ID / 5))
perturbation_idx=$((SLURM_ARRAY_TASK_ID % 5))

# Get the dataset and perturbation
dataset=${datasets[$dataset_idx]}
perturbation=${perturbations[$perturbation_idx]}

# Get the noise level using pandas in a one-liner
max_noise=$(python3 -c "
import pandas as pd
print(float(pd.read_csv('perturbation/all_saturation_points.csv', index_col=0).loc['$dataset', '$perturbation']))
")

export OMP_NUM_THREADS=1
export PYTHONFAULTHANDLER=1

# Run the Python script with the selected dataset and max noise level
srun python -O perturbation/main.py \
    --perturbation-type=$perturbation \
    --dataset=$dataset \
    --max-noise-level=$max_noise \
    --num-workers=1 \
    --dump-dir=./perturbation/results_cropped/
