#!/bin/bash
# Submit ALL compute jobs to SLURM
# Uses the most available partitions (p.hpcl93 has 3384 idle CPUs)
#
# Usage:
#   bash submit_all.sh              # Submit everything
#   bash submit_all.sh --dry-run    # Show what would be submitted

set -euo pipefail
cd "$(dirname "$0")"

CPU_LAUNCHER="slurm_cpu"
GPU_LAUNCHER="slurm_gpu"

echo "=== Submitting all experiments to SLURM ==="
echo ""

# 01: Subsampling (already has submit_all.sh, skip if results exist)
echo "--- 01: Subsampling ---"
bash 01_subsampling/submit_all.sh "$@"

# 02: Perturbation (25 jobs: 5 datasets × 5 perturbations, CPU heavy)
echo ""
echo "--- 02: Perturbation ---"
python 02_perturbation/compute.py --multirun hydra/launcher=${CPU_LAUNCHER}

# 03: Model quality (12 jobs: 2 curve_types × 3 datasets × 2 variants, GPU for GIN)
echo ""
echo "--- 03: Model Quality ---"
python 03_model_quality/compute.py --multirun hydra/launcher=${GPU_LAUNCHER}

# 04: Phase plot (single run, uses existing CSV or AutoGraph logs)
echo ""
echo "--- 04: Phase Plot ---"
python 04_phase_plot/compute.py

# 05: Benchmark (16 jobs: 4 datasets × 4 models, GPU for TabPFN)
echo ""
echo "--- 05: Benchmark ---"
python 05_benchmark/compute.py --multirun hydra/launcher=${GPU_LAUNCHER}

# 06: MMD (16 jobs: 4 datasets × 4 models)
echo ""
echo "--- 06: MMD ---"
python 06_mmd/compute.py --multirun hydra/launcher=${CPU_LAUNCHER}

# 07: Concatenation (16 jobs: 4 datasets × 4 models, GPU for TabPFN)
echo ""
echo "--- 07: Concatenation ---"
python 07_concatenation/compute.py --multirun hydra/launcher=${GPU_LAUNCHER}

# 08: GKLR (16 jobs: 4 datasets × 4 models, CPU heavy)
echo ""
echo "--- 08: GKLR ---"
python 08_gklr/compute.py --multirun hydra/launcher=${CPU_LAUNCHER}

echo ""
echo "=== All jobs submitted ==="
