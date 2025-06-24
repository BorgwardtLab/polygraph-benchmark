#!/bin/bash -l
#SBATCH -J polygraph
#SBATCH -o logs/slurm_logs/bootstrapping_%A_%a.out
#SBATCH -e logs/slurm_logs/bootstrapping_%A_%a.err
#SBATCH -t 8-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=1500G
#SBATCH --partition=p.hpcl94c

/fs/pool/pool-hartout/.conda/envs/polygraph/bin/python experiments/subsampling/subsampling.py
