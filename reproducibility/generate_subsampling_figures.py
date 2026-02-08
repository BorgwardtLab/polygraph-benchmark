#!/usr/bin/env python3
"""Generate subsampling figures showing bias/variance tradeoff.

This script evaluates how PGD metrics behave with different subsample sizes,
demonstrating the bias-variance tradeoff in metric estimation.

Usage:
    python generate_subsampling_figures.py
    python generate_subsampling_figures.py --subset  # Use smaller sample for testing
"""

import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

app = typer.Typer()

# Paths
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data" / "polygraph_graphs"
OUTPUT_DIR = Path(__file__).parent / "figures" / "subsampling"

# Styling
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def get_reference_dataset(dataset: str = "planar", split: str = "test", num_graphs: int = 4096):
    """Get reference dataset from polygraph library."""
    from polygraph.datasets.lobster import ProceduralLobsterGraphDataset
    from polygraph.datasets.planar import ProceduralPlanarGraphDataset
    from polygraph.datasets.sbm import ProceduralSBMGraphDataset

    if dataset == "planar":
        return list(ProceduralPlanarGraphDataset(split=split, num_graphs=num_graphs).to_nx())
    elif dataset == "lobster":
        return list(ProceduralLobsterGraphDataset(split=split, num_graphs=num_graphs).to_nx())
    elif dataset == "sbm":
        return list(ProceduralSBMGraphDataset(split=split, num_graphs=num_graphs).to_nx())
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def compute_pgs_for_subsample_size(
    reference_graphs: List,
    generated_graphs: List,
    subsample_size: int,
    num_bootstrap: int = 10,
) -> Dict:
    """Compute PGD metrics with a specific subsample size."""
    from polygraph.metrics import StandardPGDInterval

    # Use StandardPGDInterval with specified subsample_size
    metric = StandardPGDInterval(
        reference_graphs,
        subsample_size=subsample_size,
        num_samples=num_bootstrap,
    )

    result = metric.compute(generated_graphs)
    return {
        "mean": result["pgd"].mean,
        "std": result["pgd"].std,
    }


def plot_subsampling_curve(results_df: pd.DataFrame, output_path: Path, dataset: str):
    """Plot PGD vs subsample size."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sizes = results_df["subsample_size"].values
    means = results_df["pgs_mean"].values * 100
    stds = results_df["pgs_std"].values * 100

    ax.plot(sizes, means, "o-", color="#2E86AB", linewidth=2, markersize=8)
    ax.fill_between(sizes, means - stds, means + stds, alpha=0.2, color="#2E86AB")

    ax.set_xlabel("Subsample Size")
    ax.set_ylabel("PGD (× 100)")
    ax.set_title(f"PGD Variance vs Subsample Size ({dataset.title()})")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


@app.command()
def main(
    subset: bool = typer.Option(False, "--subset", help="Use smaller sample for quick testing"),
    dataset: str = typer.Option("planar", "--dataset", "-d", help="Dataset to use"),
):
    """Generate subsampling figures showing bias/variance tradeoff."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define subsample sizes to test
    if subset:
        subsample_sizes = [25, 50, 100]
        num_bootstrap = 3
    else:
        subsample_sizes = [50, 100, 200, 500, 1000, 2000]
        num_bootstrap = 10

    # Load data
    print(f"Loading {dataset} dataset...")
    try:
        reference_graphs = get_reference_dataset(dataset, split="test")
        # Use train split as "generated" for this experiment
        # (comparing two similar distributions to show variance)
        train_graphs = get_reference_dataset(dataset, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Compute metrics for each subsample size
    print("\nComputing PGD for different subsample sizes...")
    results = []

    for size in tqdm(subsample_sizes, desc="Subsample sizes"):
        print(f"  Subsample size: {size}")

        if size > min(len(reference_graphs), len(train_graphs)):
            print(f"    Skipping (size exceeds available graphs)")
            continue

        try:
            result = compute_pgs_for_subsample_size(
                reference_graphs,
                train_graphs,
                subsample_size=size,
                num_bootstrap=num_bootstrap,
            )
            results.append({
                "subsample_size": size,
                "pgs_mean": result["mean"],
                "pgs_std": result["std"],
            })
        except Exception as e:
            print(f"    Error: {e}")

    if results:
        results_df = pd.DataFrame(results)
        plot_subsampling_curve(results_df, OUTPUT_DIR / f"subsampling_{dataset}.pdf", dataset)

        # Save CSV for reference
        results_df.to_csv(OUTPUT_DIR / f"subsampling_{dataset}.csv", index=False)
        print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    app()
