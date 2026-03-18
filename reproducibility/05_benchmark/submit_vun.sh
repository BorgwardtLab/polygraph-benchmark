#!/bin/bash
# Submit VUN computation as separate SLURM jobs per (dataset, model).
# Skips combos that already have VUN in their result JSONs.
#
# Before running, update --partition to match your cluster.
#
# Usage:
#   bash submit_vun.sh                # submit missing combos
#   bash submit_vun.sh --force        # resubmit all combos

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
SCRIPT="${REPO_ROOT}/reproducibility/05_benchmark/compute_vun.py"
RESULTS_DIR="${REPO_ROOT}/reproducibility/tables/results/benchmark"
PARTITION="TODO_SET_YOUR_CPU_PARTITION"

FORCE_FLAG=""
if [[ "${1:-}" == "--force" ]]; then
    FORCE_FLAG="--force"
fi

DATASETS=("planar" "lobster" "sbm")
MODELS=("AUTOGRAPH" "DIGRESS" "GRAN" "ESGG")

for ds in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        if [[ -z "$FORCE_FLAG" ]]; then
            json_file="${RESULTS_DIR}/${ds}_${model}.json"
            if [[ -f "$json_file" ]]; then
                has_vun=$(python3 -c "import json; d=json.load(open('${json_file}')); print('yes' if 'vun' in d and d['vun'] is not None else 'no')" 2>/dev/null || echo "no")
                if [[ "$has_vun" == "yes" ]]; then
                    echo "SKIP ${ds}/${model} (VUN already present)"
                    continue
                fi
            fi
        fi

        echo "SUBMIT ${ds}/${model}"
        sbatch \
            --job-name="vun_${ds}_${model}" \
            --partition="${PARTITION}" \
            --nodes=1 --ntasks=1 --cpus-per-task=10 \
            --mem=32G --time=02:00:00 \
            --output="${REPO_ROOT}/vun_${ds}_${model}_%j.out" \
            --error="${REPO_ROOT}/vun_${ds}_${model}_%j.err" \
            --wrap="cd ${REPO_ROOT} && pixi run python ${SCRIPT} --dataset ${ds} --model ${model} --n-workers 8 --iso-timeout 10 ${FORCE_FLAG}"
    done
done
