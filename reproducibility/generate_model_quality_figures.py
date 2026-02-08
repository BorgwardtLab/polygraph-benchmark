#!/usr/bin/env python3
"""Generate model quality figures (training/denoising iterations).

This script computes PGD metrics as a function of training/denoising steps
for DiGress, showing how metric values change as model quality improves.

Usage:
    python generate_model_quality_figures.py
    python generate_model_quality_figures.py --subset  # Use smaller sample for testing
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
OUTPUT_DIR = Path(__file__).parent / "figures" / "model_quality"

# Styling
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def load_graphs(path: Path) -> List:
    """Load graphs from pickle file."""
    if not path.exists():
        print(f"Warning: {path} not found")
        return []
    with open(path, "rb") as f:
        graphs = pickle.load(f)
    return graphs


def get_reference_dataset(dataset: str = "planar", split: str = "test"):
    """Get reference dataset from polygraph library."""
    from polygraph.datasets.planar import ProceduralPlanarGraphDataset

    return list(ProceduralPlanarGraphDataset(split=split, num_graphs=4096).to_nx())


def compute_pgs_metrics(reference_graphs: List, generated_graphs: List, subset: bool = False) -> Dict:
    """Compute PGD metrics."""
    from polygraph.metrics import StandardPGDInterval

    if subset:
        reference_graphs = reference_graphs[:30]
        generated_graphs = generated_graphs[:30]
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


def plot_training_curve(results_df: pd.DataFrame, output_path: Path):
    """Plot PGD vs training steps."""
    fig, ax = plt.subplots(figsize=(8, 5))

    steps = results_df["steps"].values
    means = results_df["pgs_mean"].values * 100
    stds = results_df["pgs_std"].values * 100

    ax.plot(steps, means, "o-", color="#2E86AB", linewidth=2, markersize=8)
    ax.fill_between(steps, means - stds, means + stds, alpha=0.2, color="#2E86AB")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("PGD (× 100)")
    ax.set_title("DiGress: PGD vs Training Steps (Planar)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_denoising_curve(results_df: pd.DataFrame, output_path: Path):
    """Plot PGD vs denoising steps."""
    fig, ax = plt.subplots(figsize=(8, 5))

    steps = results_df["steps"].values
    means = results_df["pgs_mean"].values * 100
    stds = results_df["pgs_std"].values * 100

    ax.plot(steps, means, "o-", color="#A23B72", linewidth=2, markersize=8)
    ax.fill_between(steps, means - stds, means + stds, alpha=0.2, color="#A23B72")

    ax.set_xlabel("Denoising Steps")
    ax.set_ylabel("PGD (× 100)")
    ax.set_title("DiGress: PGD vs Denoising Steps (Planar)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def save_latex_table(results_df: pd.DataFrame, output_path: Path, title: str):
    """Save results as LaTeX table."""
    lines = []
    lines.append("\\begin{table}")
    lines.append("\\centering")
    lines.append(f"\\caption{{{title}}}")
    lines.append("\\begin{tabular}{rc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Steps} & \\textbf{PGD (× 100)} \\\\")
    lines.append("\\midrule")

    for _, row in results_df.iterrows():
        mean_scaled = row["pgs_mean"] * 100
        std_scaled = row["pgs_std"] * 100
        lines.append(f"{int(row['steps'])} & {mean_scaled:.1f} $\\pm$ {std_scaled:.1f} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")


@app.command()
def main(
    subset: bool = typer.Option(False, "--subset", help="Use smaller sample for quick testing"),
):
    """Generate model quality figures from pre-generated graphs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables_dir = Path(__file__).parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load reference dataset (Planar)
    print("Loading reference dataset...")
    try:
        reference_graphs = get_reference_dataset("planar", split="test")
    except Exception as e:
        print(f"Error loading reference dataset: {e}")
        return

    # Process training iterations
    print("\nProcessing training iterations...")
    training_dir = DATA_DIR / "DIGRESS" / "training-iterations"
    training_results = []

    if training_dir.exists():
        pkl_files = sorted(training_dir.glob("*.pkl"), key=lambda p: int(p.stem.split("_")[0]))

        for pkl_path in tqdm(pkl_files, desc="Training"):
            steps = int(pkl_path.stem.split("_")[0])
            print(f"  {steps} steps...")

            graphs = load_graphs(pkl_path)
            if not graphs:
                continue

            graphs = graphs[:len(reference_graphs)]

            try:
                results = compute_pgs_metrics(reference_graphs, graphs, subset=subset)
                training_results.append({
                    "steps": steps,
                    "pgs_mean": results.get("polyscore_mean", np.nan),
                    "pgs_std": results.get("polyscore_std", np.nan),
                })
            except Exception as e:
                print(f"    Error: {e}")

    if training_results:
        training_df = pd.DataFrame(training_results)
        plot_training_curve(training_df, OUTPUT_DIR / "training_curve.pdf")
        save_latex_table(training_df, tables_dir / "training_quality.tex", "PGD vs Training Steps (DiGress on Planar)")

    # Process denoising iterations
    print("\nProcessing denoising iterations...")
    denoising_dir = DATA_DIR / "DIGRESS" / "denoising-iterations"
    denoising_results = []

    if denoising_dir.exists():
        pkl_files = sorted(denoising_dir.glob("*.pkl"), key=lambda p: int(p.stem.split("_")[0]))

        for pkl_path in tqdm(pkl_files, desc="Denoising"):
            steps = int(pkl_path.stem.split("_")[0])
            print(f"  {steps} steps...")

            graphs = load_graphs(pkl_path)
            if not graphs:
                continue

            graphs = graphs[:len(reference_graphs)]

            try:
                results = compute_pgs_metrics(reference_graphs, graphs, subset=subset)
                denoising_results.append({
                    "steps": steps,
                    "pgs_mean": results.get("polyscore_mean", np.nan),
                    "pgs_std": results.get("polyscore_std", np.nan),
                })
            except Exception as e:
                print(f"    Error: {e}")

    if denoising_results:
        denoising_df = pd.DataFrame(denoising_results)
        plot_denoising_curve(denoising_df, OUTPUT_DIR / "denoising_curve.pdf")
        save_latex_table(denoising_df, tables_dir / "denoising_quality.tex", "PGD vs Denoising Steps (DiGress on Planar)")

    print(f"\nFigures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    app()
