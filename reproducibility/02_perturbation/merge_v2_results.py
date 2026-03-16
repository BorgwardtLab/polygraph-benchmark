#!/usr/bin/env python3
"""Merge V2 perturbation results from V2.5 (MMD + LR) and V2 TabPFN-only runs.

For each of the 25 (dataset, perturbation) combos:
  - If a full V2 result already exists: skip (already complete).
  - Otherwise: take MMD + LR metrics from V2.5, TabPFN metrics from the V2
    tabpfn-only run, and merge them into a single result file.

Usage:
    python merge_v2_results.py                # dry-run (default)
    python merge_v2_results.py --execute      # actually write merged files
"""

import argparse
import json
import sys

from pyprojroot import here

REPO_ROOT = here()
BASE_DIR = REPO_ROOT / "reproducibility" / "figures" / "02_perturbation"

V25_DIR = BASE_DIR / "results_tabpfn_weights_v2.5"
TABPFN_ONLY_DIR = BASE_DIR / "results_tabpfn_weights_v2_tabpfn_only"
FULL_V2_DIR = BASE_DIR / "results_tabpfn_weights_v2"

DATASETS = ["sbm", "planar", "lobster", "proteins", "ego"]
PERTURBATIONS = [
    "edge_rewiring",
    "edge_swapping",
    "mixing",
    "edge_deletion",
    "edge_addition",
]

# 12 TabPFN classifier metrics (weight-dependent)
TABPFN_KEYS = {
    "orbit_tabpfn_informedness",
    "orbit_tabpfn_jsd",
    "orbit5_tabpfn_informedness",
    "orbit5_tabpfn_jsd",
    "degree_tabpfn_informedness",
    "degree_tabpfn_jsd",
    "spectral_tabpfn_informedness",
    "spectral_tabpfn_jsd",
    "clustering_tabpfn_informedness",
    "clustering_tabpfn_jsd",
    "gin_tabpfn_informedness",
    "gin_tabpfn_jsd",
}

# 22 weight-independent metrics: 10 MMD + 12 LR classifier
NON_TABPFN_KEYS = {
    "orbit_tv",
    "degree_tv",
    "spectral_tv",
    "clustering_tv",
    "orbit_rbf",
    "orbit5_rbf",
    "degree_rbf",
    "spectral_rbf",
    "clustering_rbf",
    "gin_rbf",
    "orbit_lr_informedness",
    "orbit_lr_jsd",
    "orbit5_lr_informedness",
    "orbit5_lr_jsd",
    "degree_lr_informedness",
    "degree_lr_jsd",
    "spectral_lr_informedness",
    "spectral_lr_jsd",
    "clustering_lr_informedness",
    "clustering_lr_jsd",
    "gin_lr_informedness",
    "gin_lr_jsd",
}

ALL_METRIC_KEYS = TABPFN_KEYS | NON_TABPFN_KEYS


def filename(dataset: str, perturbation: str) -> str:
    return f"perturbation_{dataset}_{perturbation}.json"


def merge_one(
    dataset: str,
    perturbation: str,
    execute: bool,
) -> str:
    """Merge one (dataset, perturbation) combo. Returns status string."""
    fname = filename(dataset, perturbation)
    output_path = FULL_V2_DIR / fname

    # Already complete?
    if output_path.exists():
        data = json.loads(output_path.read_text())
        result_keys = set(data["results"][0].keys()) - {"noise_level"}
        if ALL_METRIC_KEYS.issubset(result_keys):
            return "skip (full V2 exists)"

    # Source files
    v25_path = V25_DIR / fname
    tabpfn_path = TABPFN_ONLY_DIR / fname

    if not v25_path.exists():
        return f"ERROR: V2.5 source missing ({v25_path})"
    if not tabpfn_path.exists():
        return f"WAITING: TabPFN-only source missing ({tabpfn_path})"

    v25_data = json.loads(v25_path.read_text())
    tabpfn_data = json.loads(tabpfn_path.read_text())

    v25_results = v25_data["results"]
    tabpfn_results = tabpfn_data["results"]

    # Validate matching noise levels
    if len(v25_results) != len(tabpfn_results):
        return (
            f"ERROR: result count mismatch "
            f"(V2.5={len(v25_results)}, TabPFN={len(tabpfn_results)})"
        )

    merged_results = []
    for i, (v25_row, tabpfn_row) in enumerate(zip(v25_results, tabpfn_results)):
        v25_nl = v25_row["noise_level"]
        tabpfn_nl = tabpfn_row["noise_level"]
        if abs(v25_nl - tabpfn_nl) > 1e-9:
            return (
                f"ERROR: noise_level mismatch at step {i} "
                f"(V2.5={v25_nl}, TabPFN={tabpfn_nl})"
            )

        merged_row = {"noise_level": v25_nl}

        # Non-TabPFN metrics from V2.5 (identical across weight versions)
        for key in NON_TABPFN_KEYS:
            if key not in v25_row:
                return f"ERROR: missing key '{key}' in V2.5 at step {i}"
            merged_row[key] = v25_row[key]

        # TabPFN metrics from V2 tabpfn-only run
        for key in TABPFN_KEYS:
            if key not in tabpfn_row:
                return f"ERROR: missing key '{key}' in TabPFN-only at step {i}"
            merged_row[key] = tabpfn_row[key]

        merged_results.append(merged_row)

    # Build merged output metadata
    merged_output = {
        "dataset": dataset,
        "perturbation": perturbation,
        "num_graphs": v25_data["num_graphs"],
        "seed": v25_data["seed"],
        "num_steps": v25_data["num_steps"],
        "max_noise_level": v25_data["max_noise_level"],
        "classifiers": ["tabpfn", "lr"],
        "variants": ["informedness", "jsd"],
        "results": merged_results,
        "tabpfn_package_version": tabpfn_data.get(
            "tabpfn_package_version", v25_data.get("tabpfn_package_version")
        ),
        "tabpfn_weights_version": "v2",
        "merge_source_v25": str(v25_path),
        "merge_source_tabpfn_only": str(tabpfn_path),
    }

    if execute:
        FULL_V2_DIR.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(merged_output, indent=2))
        return f"MERGED -> {output_path}"
    else:
        return f"WOULD MERGE -> {output_path}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write merged files (default: dry-run)",
    )
    args = parser.parse_args()

    print(f"V2.5 source dir:       {V25_DIR}")
    print(f"TabPFN-only source dir: {TABPFN_ONLY_DIR}")
    print(f"Output dir:            {FULL_V2_DIR}")
    print(f"Mode:                  {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print()

    statuses = {"skip": 0, "merged": 0, "waiting": 0, "error": 0}

    for dataset in DATASETS:
        for perturbation in PERTURBATIONS:
            status = merge_one(dataset, perturbation, args.execute)
            combo = f"{dataset}/{perturbation}"
            print(f"  {combo:30s} {status}")

            if status.startswith("skip"):
                statuses["skip"] += 1
            elif "MERGE" in status:
                statuses["merged"] += 1
            elif status.startswith("WAITING"):
                statuses["waiting"] += 1
            else:
                statuses["error"] += 1

    print()
    print(
        f"Summary: {statuses['skip']} skipped, {statuses['merged']} merged, "
        f"{statuses['waiting']} waiting, {statuses['error']} errors"
    )

    if statuses["error"] > 0:
        sys.exit(1)
    if statuses["waiting"] > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
