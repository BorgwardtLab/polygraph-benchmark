#!/bin/bash
# Submit VUN computation as separate SLURM jobs per (dataset, model) combo.
# Only submits combos that are missing VUN in the result JSONs.
#
# Usage:
#   bash submit_vun.sh                # submit missing combos
#   bash submit_vun.sh --force        # resubmit all combos

set -euo pipefail

REPO_ROOT="/fs/gpfs41/lv11/fileset01/pool/pool-hartout/Documents/Git/polygraph-benchmark"
SCRIPT="${REPO_ROOT}/reproducibility/05_benchmark/compute_vun.py"

FORCE_FLAG=""
if [[ "${1:-}" == "--force" ]]; then
    FORCE_FLAG="--force"
fi

DATASETS=("planar" "lobster" "sbm")
MODELS=("AUTOGRAPH" "DIGRESS" "GRAN" "ESGG")

for ds in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        # Check if VUN already present (skip if so, unless --force)
        if [[ -z "$FORCE_FLAG" ]]; then
            json_file="${REPO_ROOT}/reproducibility/tables/results/benchmark_tabpfn_v6/${ds}_${model}.json"
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
            --partition=p.hpcl93,p.hpcl94c,p.hpcl8 \
            --nodes=1 --ntasks=1 --cpus-per-task=10 \
            --mem=32G --time=02:00:00 \
            --output="${REPO_ROOT}/vun_${ds}_${model}_%j.out" \
            --error="${REPO_ROOT}/vun_${ds}_${model}_%j.err" \
            --wrap="cd ${REPO_ROOT} && pixi run python ${SCRIPT} --dataset ${ds} --model ${model} --n-workers 8 --iso-timeout 10 ${FORCE_FLAG}"
    done
done
