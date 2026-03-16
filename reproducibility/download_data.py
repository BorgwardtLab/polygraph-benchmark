#!/usr/bin/env python3
"""Download graph data from MPCDF DataShare for reproducibility.

Usage:
    python download_data.py           # Download full dataset
    python download_data.py --subset  # Download small subset for CI
"""

import shutil
import subprocess

import typer
from loguru import logger
from pyprojroot import here

app = typer.Typer()

# MPCDF DataShare URL for the full dataset
DATA_URL = "https://datashare.mpcdf.mpg.de/s/sFJBqY5DKLY4pB6"
DOWNLOAD_URL = f"{DATA_URL}/download"

# Target directory (relative to repo root)
REPO_ROOT = here()
DATA_DIR = REPO_ROOT / "data"

EXPECTED_DIRS = ["AUTOGRAPH", "DIGRESS", "ESGG", "GRAN"]


def check_data_exists() -> bool:
    """Check if data directory exists and has expected structure."""
    if not DATA_DIR.exists():
        return False
    return all((DATA_DIR / d).exists() for d in EXPECTED_DIRS)


def _download_and_extract() -> None:
    """Download the archive from MPCDF DataShare and extract it."""
    archive_path = REPO_ROOT / "data_archive.zip"

    logger.info("Downloading dataset from MPCDF DataShare...")
    try:
        subprocess.run(
            ["wget", "-O", str(archive_path), DOWNLOAD_URL],
            check=True,
        )
    except FileNotFoundError:
        subprocess.run(
            ["curl", "-L", "-o", str(archive_path), DOWNLOAD_URL],
            check=True,
        )

    logger.info("Extracting archive to {}", DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(archive_path, DATA_DIR)
    archive_path.unlink()
    logger.success("Dataset downloaded and extracted to {}", DATA_DIR)


@app.command()
def download(
    subset: bool = typer.Option(
        False, "--subset", help="Download only a small subset for CI testing"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-download even if data exists"
    ),
):
    """Download graph data from MPCDF DataShare."""
    if check_data_exists() and not force:
        logger.info(
            "Data already exists at {} — use --force to re-download", DATA_DIR
        )
        return

    if subset:
        logger.info("Subset mode: creating minimal test data...")
        create_subset_data()
        return

    _download_and_extract()


def create_subset_data():
    """Create a minimal subset of data for CI testing.

    This creates small pickle files with ~50 graphs per model
    for quick validation that scripts run without errors.
    """
    import pickle

    import networkx as nx

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    models = ["AUTOGRAPH", "DIGRESS", "ESGG", "GRAN"]
    datasets = ["planar", "lobster", "sbm", "proteins"]

    for model in models:
        model_dir = DATA_DIR / model
        model_dir.mkdir(exist_ok=True)
        for dataset in datasets:
            graphs = [nx.erdos_renyi_graph(20, 0.3) for _ in range(50)]
            pkl_path = model_dir / f"{dataset}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(graphs, f)
            logger.debug("Created {}", pkl_path)

    # DIGRESS iteration subdirectories
    digress_denoising = DATA_DIR / "DIGRESS" / "denoising-iterations"
    digress_training = DATA_DIR / "DIGRESS" / "training-iterations"
    digress_denoising.mkdir(exist_ok=True)
    digress_training.mkdir(exist_ok=True)

    for steps in [15, 30, 45, 60, 75, 90]:
        graphs = [nx.erdos_renyi_graph(20, 0.3) for _ in range(50)]
        pkl_path = digress_denoising / f"{steps}_steps.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(graphs, f)
        logger.debug("Created {}", pkl_path)

    for steps in [119, 209, 299, 419, 509, 1019, 1499, 2009, 2519, 2999, 3479]:
        graphs = [nx.erdos_renyi_graph(20, 0.3) for _ in range(50)]
        pkl_path = digress_training / f"{steps}_steps.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(graphs, f)
        logger.debug("Created {}", pkl_path)

    # Molecule eval placeholder files
    mol_dir = DATA_DIR / "molecule_eval"
    mol_dir.mkdir(exist_ok=True)
    for fname in [
        "guacamol_autograph.smiles",
        "guacamol_digress.smiles",
        "moses_autograph.smiles",
        "moses_digress.smiles",
    ]:
        (mol_dir / fname).write_text(
            "# Placeholder for CI testing\nC\nCC\nCCC\n"
        )
        logger.debug("Created {}", mol_dir / fname)

    logger.success(
        "Subset data created (placeholder only — download full dataset for reproducibility)"
    )


if __name__ == "__main__":
    app()
