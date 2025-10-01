#!/bin/bash
set -e

# Parse command line arguments
CONVERT_MOLECULES=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --molecules)
            CONVERT_MOLECULES=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p .local/data
rm -rf .local/data
mkdir -p .local/data/planar
mkdir -p .local/data/sbm

python conversion/dobson_doig.py --destination .local/data/dobson_doig
python conversion/ego.py --destination .local/data/ego
python conversion/ego.py --destination .local/data/ego_small --small
python conversion/spectre.py --destination .local/data/planar --dataset planar
python conversion/spectre.py --destination .local/data/sbm --dataset sbm
python conversion/lobster.py --destination .local/data/lobster
python conversion/point_clouds.py --destination .local/data/point_clouds
python conversion/modelnet10.py --destination .local/data/modelnet10 --dataset modelnet10
python conversion/modelnet10.py --destination .local/data/modelnet40 --dataset modelnet40

# Only convert molecule datasets if flag is set
if [ $CONVERT_MOLECULES -eq 1 ]; then
    python conversion/moses.py --destination .local/data/moses
    python conversion/guacamol.py --destination .local/data/guacamol
    python conversion/qm9.py --destination .local/data/qm9
fi
