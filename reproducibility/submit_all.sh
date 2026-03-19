#!/bin/bash
# Submit ALL compute jobs to SLURM.
#
# Configure partitions in configs/hydra/launcher/slurm_{cpu,gpu}.yaml
# before running this script.
#
# Usage:
#   bash submit_all.sh

set -euo pipefail
cd "$(dirname "$0")"

CPU_LAUNCHER="slurm_cpu"
GPU_LAUNCHER="slurm_gpu"

echo "=== Submitting all experiments to SLURM ==="
echo ""

echo "--- 01: Subsampling ---"
bash 01_subsampling/submit_all.sh "$@"

echo ""
echo "--- 02: Perturbation ---"
python 02_perturbation/compute.py --multirun hydra/launcher=${CPU_LAUNCHER}

echo ""
echo "--- 03: Model Quality ---"
python 03_model_quality/compute.py --multirun hydra/launcher=${GPU_LAUNCHER}

echo ""
echo "--- 03b: Model Quality VUN ---"
python 03_model_quality/compute_vun.py --multirun hydra/launcher=${CPU_LAUNCHER}

echo ""
echo "--- 04: Phase Plot ---"
python 04_phase_plot/compute.py

echo ""
echo "--- 05: Benchmark ---"
python 05_benchmark/compute.py --multirun hydra/launcher=${GPU_LAUNCHER}

echo ""
echo "--- 06: MMD ---"
python 06_mmd/compute.py --multirun hydra/launcher=${CPU_LAUNCHER}

echo ""
echo "--- 07: Concatenation ---"
python 07_concatenation/compute.py --multirun hydra/launcher=${GPU_LAUNCHER}

echo ""
echo "--- 08: GKLR ---"
python 08_gklr/compute.py --multirun hydra/launcher=${CPU_LAUNCHER}

echo ""
echo "--- 09: Train-test reference ---"
python 09_train_test_reference/compute.py --multirun hydra/launcher=${GPU_LAUNCHER}

echo ""
echo "=== All jobs submitted ==="
