#!/usr/bin/env python3
"""Generate perturbation figures showing metric sensitivity.

This script applies various perturbations to graphs and measures how
PGD metrics respond, demonstrating metric sensitivity to distributional shift.

Usage:
    python generate_perturbation_figures.py
    python generate_perturbation_figures.py --subset  # Use smaller sample for testing
"""

import random
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

app = typer.Typer()

# Paths
REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / "figures" / "perturbation"

# Styling
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def get_reference_dataset(dataset: str = "planar", split: str = "test", num_graphs: int = 2048):
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


def edge_rewiring_perturbation(graph: nx.Graph, noise_level: float, rng: np.random.Generator) -> nx.Graph:
    """Rewire edges randomly while preserving edge count."""
    g = graph.copy()
    edges = list(g.edges())
    n_rewire = int(len(edges) * noise_level)

    for _ in range(n_rewire):
        if not edges:
            break
        edge = edges[rng.integers(len(edges))]
        g.remove_edge(*edge)
        edges.remove(edge)

        # Add random edge
        nodes = list(g.nodes())
        for _ in range(100):  # Max attempts
            u, v = rng.choice(nodes, size=2, replace=False)
            if not g.has_edge(u, v):
                g.add_edge(u, v)
                break

    return g


def edge_deletion_perturbation(graph: nx.Graph, noise_level: float, rng: np.random.Generator) -> nx.Graph:
    """Delete edges randomly."""
    g = graph.copy()
    edges = list(g.edges())
    n_delete = int(len(edges) * noise_level)

    if n_delete > 0:
        edges_to_delete = rng.choice(len(edges), size=min(n_delete, len(edges)), replace=False)
        for i in edges_to_delete:
            g.remove_edge(*edges[i])

    return g


def edge_addition_perturbation(graph: nx.Graph, noise_level: float, rng: np.random.Generator) -> nx.Graph:
    """Add random edges."""
    g = graph.copy()
    n_edges = g.number_of_edges()
    n_add = int(n_edges * noise_level)

    nodes = list(g.nodes())
    added = 0
    for _ in range(n_add * 10):  # Max attempts
        if added >= n_add:
            break
        u, v = rng.choice(nodes, size=2, replace=False)
        if not g.has_edge(u, v):
            g.add_edge(u, v)
            added += 1

    return g


def apply_perturbation(
    graphs: List[nx.Graph],
    perturbation_fn: Callable,
    noise_level: float,
    seed: int = 42,
) -> List[nx.Graph]:
    """Apply perturbation to all graphs."""
    rng = np.random.default_rng(seed)
    return [perturbation_fn(g, noise_level, rng) for g in graphs]


def compute_pgs_metrics(reference_graphs: List, generated_graphs: List, subset: bool = False) -> Dict:
    """Compute PGD metrics."""
    from polygraph.metrics import StandardPGDInterval

    if subset:
        subsample_size = 15
        num_samples = 3
    else:
        subsample_size = min(len(reference_graphs), len(generated_graphs)) // 4
        num_samples = 10

    metric = StandardPGDInterval(
        reference_graphs,
        subsample_size=subsample_size,
        num_samples=num_samples,
    )

    result = metric.compute(generated_graphs)
    return {
        "polyscore_mean": result["pgd"].mean,
        "polyscore_std": result["pgd"].std,
    }


def plot_perturbation_curve(
    results_df: pd.DataFrame,
    output_path: Path,
    perturbation_name: str,
    dataset: str,
):
    """Plot PGD vs noise level."""
    fig, ax = plt.subplots(figsize=(8, 5))

    noise_levels = results_df["noise_level"].values
    means = results_df["pgs_mean"].values * 100
    stds = results_df["pgs_std"].values * 100

    ax.plot(noise_levels, means, "o-", color="#2E86AB", linewidth=2, markersize=8)
    ax.fill_between(noise_levels, means - stds, means + stds, alpha=0.2, color="#2E86AB")

    ax.set_xlabel("Noise Level (fraction of edges)")
    ax.set_ylabel("PGD (× 100)")
    ax.set_title(f"PGD vs {perturbation_name.replace('_', ' ').title()} ({dataset.title()})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_perturbations(
    all_results: Dict[str, pd.DataFrame],
    output_path: Path,
    dataset: str,
):
    """Plot all perturbation types on one figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"edge_rewiring": "#2E86AB", "edge_deletion": "#A23B72", "edge_addition": "#F18F01"}
    markers = {"edge_rewiring": "o", "edge_deletion": "s", "edge_addition": "^"}

    for name, df in all_results.items():
        noise_levels = df["noise_level"].values
        means = df["pgs_mean"].values * 100
        stds = df["pgs_std"].values * 100

        label = name.replace("_", " ").title()
        ax.plot(noise_levels, means, f"{markers[name]}-", color=colors[name],
                linewidth=2, markersize=8, label=label)
        ax.fill_between(noise_levels, means - stds, means + stds, alpha=0.15, color=colors[name])

    ax.set_xlabel("Noise Level (fraction of edges)")
    ax.set_ylabel("PGD (× 100)")
    ax.set_title(f"PGD Sensitivity to Perturbations ({dataset.title()})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


@app.command()
def main(
    subset: bool = typer.Option(False, "--subset", help="Use smaller sample for quick testing"),
    dataset: str = typer.Option("sbm", "--dataset", "-d", help="Dataset to use"),
):
    """Generate perturbation figures showing metric sensitivity."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define noise levels to test
    if subset:
        noise_levels = [0.0, 0.1, 0.3, 0.5]
        num_graphs = 100
    else:
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        num_graphs = 1000

    # Load data
    print(f"Loading {dataset} dataset ({num_graphs} graphs)...")
    try:
        reference_graphs = get_reference_dataset(dataset, split="test", num_graphs=num_graphs)
        # Use a copy of test set as base for perturbations
        base_graphs = get_reference_dataset(dataset, split="train", num_graphs=num_graphs)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    perturbations = {
        "edge_rewiring": edge_rewiring_perturbation,
        "edge_deletion": edge_deletion_perturbation,
        "edge_addition": edge_addition_perturbation,
    }

    all_results = {}

    for pert_name, pert_fn in perturbations.items():
        print(f"\nProcessing {pert_name}...")
        results = []

        for noise_level in tqdm(noise_levels, desc=f"Noise levels ({pert_name})"):
            print(f"  Noise level: {noise_level}")

            # Apply perturbation
            perturbed_graphs = apply_perturbation(base_graphs, pert_fn, noise_level)

            try:
                result = compute_pgs_metrics(reference_graphs, perturbed_graphs, subset=subset)
                results.append({
                    "noise_level": noise_level,
                    "pgs_mean": result.get("polyscore_mean", np.nan),
                    "pgs_std": result.get("polyscore_std", np.nan),
                })
            except Exception as e:
                print(f"    Error: {e}")

        if results:
            results_df = pd.DataFrame(results)
            all_results[pert_name] = results_df
            plot_perturbation_curve(
                results_df,
                OUTPUT_DIR / f"perturbation_{pert_name}_{dataset}.pdf",
                pert_name,
                dataset,
            )

    # Combined plot
    if all_results:
        plot_all_perturbations(all_results, OUTPUT_DIR / f"perturbation_all_{dataset}.pdf", dataset)

    print(f"\nFigures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    app()
