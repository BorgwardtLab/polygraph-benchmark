#!/bin/bash
# Recompute all PGD-dependent results with TabPFN V2 weights.
#
# Uses Hydra --multirun with the H100 SLURM launcher for GPU experiments,
# and slurm_cpu for experiments that don't benefit as much from GPU.
#
# Usage:
#   bash slurm_recompute_v2.sh           # Submit all jobs
#   bash slurm_recompute_v2.sh --dry-run # Print commands without running

set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN - commands will be printed but not executed ==="
fi

run_cmd() {
    echo ""
    echo ">>> $*"
    if [[ "$DRY_RUN" == "false" ]]; then
        eval "$@"
    fi
}

COMMON="tabpfn_weights_version=v2 results_suffix=_tabpfn_weights_v2"

# ---------------------------------------------------------------------------
# 1. Subsampling PGD (01) — most expensive, many combos
# ---------------------------------------------------------------------------
echo "=== 01_subsampling PGD (V2 weights) ==="
run_cmd "pixi run python 01_subsampling/compute_pgd.py --multirun \
    hydra/launcher=slurm_gpu_h100 \
    $COMMON"

# ---------------------------------------------------------------------------
# 2. Perturbation (02) — 25 (dataset, perturbation) combos
# ---------------------------------------------------------------------------
echo ""
echo "=== 02_perturbation (V2 weights) ==="
run_cmd "pixi run python 02_perturbation/compute.py --multirun \
    hydra/launcher=slurm_gpu_h100 \
    $COMMON"

# ---------------------------------------------------------------------------
# 3. Model quality (03) — training + denoising curves
# ---------------------------------------------------------------------------
echo ""
echo "=== 03_model_quality (V2 weights) ==="
run_cmd "pixi run python 03_model_quality/compute.py --multirun \
    hydra/launcher=slurm_gpu_h100 \
    pgd_only=true \
    $COMMON"

# ---------------------------------------------------------------------------
# 4. Benchmark table (05) — 16 (dataset, model) combos
# ---------------------------------------------------------------------------
echo ""
echo "=== 05_benchmark (V2 weights) ==="
run_cmd "pixi run python 05_benchmark/compute.py --multirun \
    hydra/launcher=slurm_gpu_h100 \
    skip_vun=true \
    $COMMON"

# ---------------------------------------------------------------------------
# 5. Concatenation ablation (07) — 12 (dataset, model) combos
# ---------------------------------------------------------------------------
echo ""
echo "=== 07_concatenation (V2 weights) ==="
run_cmd "pixi run python 07_concatenation/compute.py --multirun \
    hydra/launcher=slurm_gpu_h100 \
    $COMMON"

echo ""
echo "=== All jobs submitted (or printed in dry-run mode) ==="
echo "Results will be saved to *_tabpfn_weights_v2 directories."
echo "Use 'squeue -u \$USER' to monitor job status."
