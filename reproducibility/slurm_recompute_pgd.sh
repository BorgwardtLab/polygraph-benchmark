#!/bin/bash
#SBATCH --job-name=recompute_pgd
#SBATCH --partition=p.hpcl8
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=logs/recompute_pgd_%j.out
#SBATCH --error=logs/recompute_pgd_%j.err

export TABPFN_ALLOW_CPU_LARGE_DATASET=1

cd /fs/gpfs41/lv11/fileset01/pool/pool-hartout/Documents/Git/polygraph-benchmark/reproducibility

pixi run python recompute_training_pgd.py
