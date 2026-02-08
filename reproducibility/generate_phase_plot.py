#!/usr/bin/env python3
"""Generate phase plot showing training dynamics.

This script generates a phase plot showing the relationship between
validation loss and metric values during training.

Note: This requires training logs with validation loss values.
If logs are not available, this script will generate a placeholder figure.

Usage:
    python generate_phase_plot.py
    python generate_phase_plot.py --logs-dir /path/to/logs
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.collections import LineCollection
from tqdm import tqdm

app = typer.Typer()

# Paths
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data" / "polygraph_graphs"
OUTPUT_DIR = Path(__file__).parent / "figures" / "phase_plot"

# Styling
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def load_training_logs(logs_dir: Path) -> Optional[pd.DataFrame]:
    """Load training logs if available."""
    if not logs_dir.exists():
        return None

    # Look for CSV files with metrics
    csv_files = list(logs_dir.glob("*.csv"))
    if not csv_files:
        return None

    # Try to load and combine
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if "val/loss" in df.columns or "val/loss_epoch" in df.columns:
                dfs.append(df)
        except Exception:
            continue

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None


def get_reference_dataset(dataset: str = "planar", split: str = "test"):
    """Get reference dataset from polygraph library."""
    from polygraph.datasets.planar import ProceduralPlanarGraphDataset

    return list(ProceduralPlanarGraphDataset(split=split, num_graphs=4096).to_nx())


def compute_metrics_for_training_steps(
    reference_graphs: List,
    training_dir: Path,
    subset: bool = False,
) -> pd.DataFrame:
    """Compute PGD and VUN metrics for each training checkpoint."""
    from polygraph.metrics import StandardPGDInterval, VUN
    from polygraph.datasets.planar import is_planar_graph

    if not training_dir.exists():
        return pd.DataFrame()

    # Get training checkpoint files
    pkl_files = sorted(training_dir.glob("*.pkl"), key=lambda p: int(p.stem.split("_")[0]))
    if not pkl_files:
        return pd.DataFrame()

    # Limit samples for subset mode
    if subset:
        reference_graphs = reference_graphs[:30]
        subsample_size = 15
        num_samples = 3
    else:
        subsample_size = min(len(reference_graphs), 1000) // 4
        num_samples = 10

    results = []

    for pkl_path in tqdm(pkl_files, desc="Computing metrics"):
        steps = int(pkl_path.stem.split("_")[0])

        try:
            with open(pkl_path, "rb") as f:
                graphs = pickle.load(f)

            if subset:
                graphs = graphs[:30]
            else:
                graphs = graphs[:len(reference_graphs)]

            # Compute PGD
            pgs_metric = StandardPGDInterval(
                reference_graphs,
                subsample_size=subsample_size,
                num_samples=num_samples,
            )
            pgs_result = pgs_metric.compute(graphs)

            # Compute VUN (using planar validity)
            vun_metric = VUN(
                train_graphs=reference_graphs,
                validity_fn=is_planar_graph,
            )
            vun_result = vun_metric.compute(graphs)

            results.append({
                "steps": steps,
                "pgs_mean": pgs_result["pgd"].mean,
                "pgs_std": pgs_result["pgd"].std,
                "vun": vun_result.get("valid_unique_novel_mle", np.nan),
            })

        except Exception as e:
            print(f"Error at step {steps}: {e}")

    return pd.DataFrame(results)


def plot_phase_diagram(df: pd.DataFrame, output_path: Path):
    """Create a phase plot showing PGD vs VUN during training."""
    # Check if VUN has valid values
    vun_values = df["vun"].values
    if np.all(np.isnan(vun_values)):
        print("Warning: No valid VUN values, skipping phase diagram")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract data
    pgs = df["pgs_mean"].values * 100
    vun = vun_values * 100
    steps = df["steps"].values

    # Create colored line based on training progress
    points = np.array([vun, pgs]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color by training step (normalized)
    norm = plt.Normalize(steps.min(), steps.max())
    lc = LineCollection(segments, cmap="viridis", norm=norm)
    lc.set_array(steps[:-1])
    lc.set_linewidth(2)
    line = ax.add_collection(lc)

    # Add start/end markers
    ax.scatter(vun[0], pgs[0], color="green", s=100, zorder=5, label="Start", marker="o")
    ax.scatter(vun[-1], pgs[-1], color="red", s=100, zorder=5, label="End", marker="*")

    # Colorbar
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label("Training Steps")

    ax.set_xlabel("VUN (× 100)")
    ax.set_ylabel("PGD (× 100)")
    ax.set_title("Training Phase Diagram: PGD vs VUN")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set axis limits with some padding
    ax.set_xlim(vun.min() - 5, vun.max() + 5)
    ax.set_ylim(pgs.min() - 2, pgs.max() + 2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_metrics_vs_steps(df: pd.DataFrame, output_path: Path):
    """Plot PGD and VUN vs training steps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    steps = df["steps"].values
    pgs = df["pgs_mean"].values * 100
    pgs_std = df["pgs_std"].values * 100
    vun = df["vun"].values * 100

    # PGD vs steps
    ax1.plot(steps, pgs, "o-", color="#2E86AB", linewidth=2, markersize=6)
    ax1.fill_between(steps, pgs - pgs_std, pgs + pgs_std, alpha=0.2, color="#2E86AB")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("PGD (× 100)")
    ax1.set_title("PGD vs Training Steps")
    ax1.grid(True, alpha=0.3)

    # VUN vs steps
    ax2.plot(steps, vun, "o-", color="#A23B72", linewidth=2, markersize=6)
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("VUN (× 100)")
    ax2.set_title("VUN vs Training Steps")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


@app.command()
def main(
    subset: bool = typer.Option(False, "--subset", help="Use smaller sample for quick testing"),
    logs_dir: Optional[Path] = typer.Option(None, "--logs-dir", help="Directory with training logs"),
):
    """Generate phase plot showing training dynamics."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for training iteration data
    training_dir = DATA_DIR / "DIGRESS" / "training-iterations"

    if not training_dir.exists():
        print(f"Warning: Training data not found at {training_dir}")
        print("Please download the full dataset to generate phase plots.")
        return

    # Load reference dataset
    print("Loading reference dataset...")
    try:
        reference_graphs = get_reference_dataset("planar", split="test")
        train_graphs = get_reference_dataset("planar", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Compute metrics for each training checkpoint
    print("\nComputing metrics for training checkpoints...")
    results_df = compute_metrics_for_training_steps(
        train_graphs, training_dir, subset=subset
    )

    if results_df.empty:
        print("No results computed. Please check your data.")
        return

    # Generate plots
    print("\nGenerating plots...")
    plot_phase_diagram(results_df, OUTPUT_DIR / "phase_diagram.pdf")
    plot_metrics_vs_steps(results_df, OUTPUT_DIR / "metrics_vs_steps.pdf")

    # Save CSV for reference
    results_df.to_csv(OUTPUT_DIR / "phase_plot_data.csv", index=False)

    print(f"\nFigures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    app()
