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

rm -rf ./data
mkdir -p ./data/
mkdir -p ./data/planar
mkdir -p ./data/sbm

python conversion/dobson_doig.py --destination ./data/dobson_doig
python conversion/ego.py --destination ./data/ego
python conversion/ego.py --destination ./data/ego_small --small
python conversion/spectre.py --destination ./data/planar --dataset planar
python conversion/spectre.py --destination ./data/sbm --dataset sbm
python conversion/lobster.py --destination ./data/lobster
python conversion/point_clouds.py --destination ./data/point_clouds

# Only convert molecule datasets if flag is set
if [ $CONVERT_MOLECULES -eq 1 ]; then
    python conversion/moses.py --destination ./data/moses
    python conversion/guacamol.py --destination ./data/guacamol
    python conversion/qm9.py --destination ./data/qm9
fi
