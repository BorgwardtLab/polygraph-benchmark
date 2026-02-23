#!/usr/bin/env bash
set -euo pipefail

# Submit all subsampling experiments in one command:
# - MMD: tiered CPU scheduling (small -> hpcl8, large -> hpcl94c)
# - PGD: GPU scheduling on hpcl8
#
# Extra Hydra overrides are forwarded to all submissions.
# Async result streaming:
#   - If POLYGRAPH_ASYNC_RESULTS_FILE is unset, defaults to
#     reproducibility/results/01_subsampling/async_results.jsonl
# Example:
#   ./submit_all.sh dataset=planar model=AUTOGRAPH subset=true

if [[ -z "${POLYGRAPH_ASYNC_RESULTS_FILE:-}" ]]; then
  export POLYGRAPH_ASYNC_RESULTS_FILE="reproducibility/results/01_subsampling/async_results.jsonl"
fi
mkdir -p "$(dirname "${POLYGRAPH_ASYNC_RESULTS_FILE}")"
echo "Streaming async results to: ${POLYGRAPH_ASYNC_RESULTS_FILE}"

echo "[1/3] Submitting MMD small jobs on hpcl8..."
pixi run python "reproducibility/01_subsampling/compute_mmd.py" --multirun \
  hydra/launcher=slurm_cpu_small \
  subsample_size=8,16,32,64,128 \
  "$@"

echo "[2/3] Submitting MMD large jobs on hpcl94c..."
pixi run python "reproducibility/01_subsampling/compute_mmd.py" --multirun \
  hydra/launcher=slurm_cpu_large \
  subsample_size=256,512,1024,2048,4096 \
  "$@"

echo "[3/3] Submitting PGD jobs on hpcl8 GPU..."
pixi run python "reproducibility/01_subsampling/compute_pgd.py" --multirun \
  hydra/launcher=slurm_gpu \
  "$@"

echo "All submissions launched."
