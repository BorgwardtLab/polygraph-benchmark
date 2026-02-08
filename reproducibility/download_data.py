#!/usr/bin/env python3
"""Download graph data from Proton Drive for reproducibility.

Usage:
    python download_data.py           # Download full dataset
    python download_data.py --subset  # Download small subset for CI
"""

import shutil
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer()

# Proton Drive URL for the full dataset
DATA_URL = "https://drive.proton.me/urls/VM4NWYBQD0#3sqmZtmSgWTB"

# Target directory (relative to repo root)
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data" / "polygraph_graphs"


def check_data_exists() -> bool:
    """Check if data directory exists and has expected structure."""
    expected_dirs = ["AUTOGRAPH", "DIGRESS", "ESGG", "GRAN"]
    if not DATA_DIR.exists():
        return False
    for d in expected_dirs:
        if not (DATA_DIR / d).exists():
            return False
    return True


@app.command()
def download(
    subset: bool = typer.Option(False, "--subset", help="Download only a small subset for CI testing"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download even if data exists"),
):
    """Download graph data from Proton Drive."""

    if check_data_exists() and not force:
        print(f"Data already exists at {DATA_DIR}")
        print("Use --force to re-download")
        return

    if subset:
        print("Subset mode: Creating minimal test data...")
        create_subset_data()
        return

    print("=" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print()
    print("Due to Proton Drive's authentication requirements,")
    print("please download the data manually:")
    print()
    print(f"  1. Visit: {DATA_URL}")
    print(f"  2. Download the archive")
    print(f"  3. Extract to: {DATA_DIR}")
    print()
    print("Expected structure after extraction:")
    print("  data/polygraph_graphs/")
    print("  ├── AUTOGRAPH/")
    print("  ├── DIGRESS/")
    print("  ├── ESGG/")
    print("  ├── GRAN/")
    print("  └── molecule_eval/")
    print()
    print("=" * 60)


def create_subset_data():
    """Create a minimal subset of data for CI testing.

    This creates small pickle files with ~50 graphs per model
    for quick validation that scripts run without errors.
    """
    import pickle
    import networkx as nx

    # Create directory structure
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    models = ["AUTOGRAPH", "DIGRESS", "ESGG", "GRAN"]
    datasets = ["planar", "lobster", "sbm", "proteins"]

    for model in models:
        model_dir = DATA_DIR / model
        model_dir.mkdir(exist_ok=True)

        for dataset in datasets:
            # Generate 50 small random graphs as placeholder
            graphs = [nx.erdos_renyi_graph(20, 0.3) for _ in range(50)]

            pkl_path = model_dir / f"{dataset}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(graphs, f)

            print(f"Created: {pkl_path}")

    # Create DIGRESS subdirectories
    digress_denoising = DATA_DIR / "DIGRESS" / "denoising-iterations"
    digress_training = DATA_DIR / "DIGRESS" / "training-iterations"
    digress_denoising.mkdir(exist_ok=True)
    digress_training.mkdir(exist_ok=True)

    # Denoising iterations
    for steps in [15, 30, 45, 60, 75, 90]:
        graphs = [nx.erdos_renyi_graph(20, 0.3) for _ in range(50)]
        pkl_path = digress_denoising / f"{steps}_steps.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(graphs, f)
        print(f"Created: {pkl_path}")

    # Training iterations
    for steps in [119, 209, 299, 419, 509, 1019, 1499, 2009, 2519, 2999, 3479]:
        graphs = [nx.erdos_renyi_graph(20, 0.3) for _ in range(50)]
        pkl_path = digress_training / f"{steps}_steps.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(graphs, f)
        print(f"Created: {pkl_path}")

    # Create molecule_eval directory with placeholder files
    mol_dir = DATA_DIR / "molecule_eval"
    mol_dir.mkdir(exist_ok=True)

    # Create empty placeholder files for molecule data
    mol_files = [
        "guacamol_autograph.smiles",
        "guacamol_digress.smiles",
        "moses_autograph.smiles",
        "moses_digress.smiles",
    ]
    for fname in mol_files:
        (mol_dir / fname).write_text("# Placeholder for CI testing\nC\nCC\nCCC\n")
        print(f"Created: {mol_dir / fname}")

    print()
    print("Subset data created successfully!")
    print("Note: This is placeholder data for CI testing only.")
    print("For actual reproducibility, download the full dataset.")


if __name__ == "__main__":
    app()
