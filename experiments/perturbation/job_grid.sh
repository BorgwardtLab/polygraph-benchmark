#!/bin/bash

#SBATCH --job-name=perturbation-experiments
#SBATCH --output=./perturbation/results_full/job_%A_%a.out
#SBATCH --error=./perturbation/results_full/job_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-core=2
#SBATCH --mem-per-cpu=10G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --array=0-24

# Define the datasets array
datasets=(planar sbm lobster proteins ego)

perturbations=(edge_rewiring edge_swapping edge_deletion edge_addition mixing)

# Calculate dataset and perturbation indices
dataset_idx=$((SLURM_ARRAY_TASK_ID / 5))
perturbation_idx=$((SLURM_ARRAY_TASK_ID % 5))

# Get the dataset and perturbation
dataset=${datasets[$dataset_idx]}
perturbation=${perturbations[$perturbation_idx]}

# Run the Python script with the selected dataset
srun python -O perturbation/main.py --perturbation-type=$perturbation --dataset=$dataset --dump-dir=./perturbation/results_full/
