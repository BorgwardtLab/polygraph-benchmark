#!/usr/bin/env bash
set -euo pipefail

# Submit all subsampling experiments to SLURM.
#
# Configure partitions in configs/hydra/launcher/slurm_{cpu,gpu}.yaml
# before running this script.
#
# Extra Hydra overrides are forwarded to all submissions.
#
# Example:
#   ./submit_all.sh dataset=planar model=AUTOGRAPH subset=true

echo "[1/2] Submitting MMD jobs (CPU)..."
pixi run python "reproducibility/01_subsampling/compute_mmd.py" --multirun \
  hydra/launcher=slurm_cpu \
  "$@"

echo "[2/2] Submitting PGD jobs (GPU)..."
pixi run python "reproducibility/01_subsampling/compute_pgd.py" --multirun \
  hydra/launcher=slurm_gpu \
  "$@"

echo "All submissions launched."
