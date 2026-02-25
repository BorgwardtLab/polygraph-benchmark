#!/usr/bin/env python3
"""Generate all model quality figures for the paper.

Reads pre-computed JSON results from compute.py and produces:

  Denoising figures (2):
    model-quality_denoising-iterations_validity_polyscore_all_mmd_jsd.pdf
    model-quality_denoising-iterations_validity_polyscore_all_mmd_informedness-adaptive.pdf

  Training figures (4):
    all_training_epochs_jsd.pdf                      (all datasets, appendix)
    all_training_epochs_jsd_sbm.pdf                  (SBM only, main text)
    all_training_epochs_informedness-adaptive.pdf     (all datasets, appendix)
    all_training_epochs_informedness-adaptive_sbm.pdf (SBM only)

Usage:
    python plot.py
    python plot.py --paper-dir /path/to/paper/figures/model_quality/
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from pyprojroot import here

app = typer.Typer()

REPO_ROOT = here()
RESULTS_DIR = REPO_ROOT / "reproducibility" / "figures" / "03_model_quality" / "results"
OUTPUT_DIR = REPO_ROOT / "reproducibility" / "figures" / "03_model_quality"
STYLE_FILE = Path(__file__).resolve().parent.parent / "polygraph.mplstyle"

DATASETS = ["planar", "sbm", "lobster"]
DATASET_DISPLAY = {"planar": "Planar-L", "sbm": "SBM-L", "lobster": "Lobster-L"}

MMD_KEYS = ["orbit_mmd", "orbit5_mmd", "degree_mmd", "spectral_mmd", "clustering_mmd", "gin_mmd"]
MMD_DISPLAY = {
    "orbit_mmd": "Orbit MMD",
    "orbit5_mmd": "Orbit5 MMD",
    "degree_mmd": "Degree MMD",
    "spectral_mmd": "Spectral MMD",
    "clustering_mmd": "Clustering MMD",
    "gin_mmd": "GIN MMD",
}


def setup_plotting():
    if STYLE_FILE.exists():
        plt.style.use(str(STYLE_FILE))
    sns.set_style("ticks")
    sns.set_palette("colorblind")


def load_results(curve_type: str, dataset: str, variant: str) -> Optional[pd.DataFrame]:
    """Load JSON results for a specific (curve_type, dataset, variant)."""
    path = RESULTS_DIR / f"{curve_type}_{dataset}_{variant}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if not data.get("results"):
        return None
    return pd.DataFrame(data["results"])


def _metric_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of metric columns present in the data."""
    cols = []
    if "validity" in df.columns:
        cols.append("validity")
    if "polyscore" in df.columns:
        cols.append("polyscore")
    for k in MMD_KEYS:
        if k in df.columns:
            cols.append(k)
    return cols


def _metric_label(col: str) -> str:
    if col == "validity":
        return "Validity"
    if col == "polyscore":
        return "PGS"
    return MMD_DISPLAY.get(col, col)


def _plot_multi_panel(
    data_dict: Dict[str, pd.DataFrame],
    x_label: str,
    output_path: Path,
    figwidth_per_col: float = 2.2,
    figheight_per_row: float = 1.8,
) -> None:
    """Create a multi-panel figure: rows=datasets, cols=metrics."""
    all_cols = set()
    for df in data_dict.values():
        all_cols.update(_metric_columns(df))

    col_order = []
    for c in ["validity", "polyscore"] + MMD_KEYS:
        if c in all_cols:
            col_order.append(c)

    datasets = list(data_dict.keys())
    n_rows = len(datasets)
    n_cols = len(col_order)

    if n_rows == 0 or n_cols == 0:
        return

    palette = sns.color_palette("colorblind")
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figwidth_per_col * n_cols, figheight_per_row * n_rows),
        squeeze=False,
    )

    for i, ds in enumerate(datasets):
        df = data_dict[ds].sort_values("steps")
        steps = df["steps"].values

        for j, metric in enumerate(col_order):
            ax = axes[i, j]
            if metric not in df.columns:
                ax.set_visible(False)
                continue

            vals = df[metric].values
            color = palette[j % len(palette)]
            ax.plot(steps, vals, marker="o", markersize=2, linewidth=1.2, color=color)

            if i == 0:
                ax.set_title(_metric_label(metric), fontsize=9)
            if j == 0:
                ax.set_ylabel(DATASET_DISPLAY.get(ds, ds), fontsize=9, fontweight="bold")
            if i == n_rows - 1:
                ax.set_xlabel(x_label, fontsize=8)

            ax.tick_params(labelsize=7)
            sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    logger.success("Saved: {}", output_path)


@app.command()
def main(
    paper_dir: Optional[Path] = typer.Option(
        None, "--paper-dir",
        help="Copy outputs into paper figures/model_quality/ directory",
    ),
    results_suffix: str = typer.Option("", "--results-suffix", help="Suffix for results dir and output files (e.g. _tabpfn_v6)"),
):
    """Generate all model quality figures for the paper."""
    setup_plotting()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_dir = REPO_ROOT / "reproducibility" / "figures" / "03_model_quality" / f"results{results_suffix}"

    def _load(curve_type: str, dataset: str, variant: str) -> Optional[pd.DataFrame]:
        path = results_dir / f"{curve_type}_{dataset}_{variant}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        if not data.get("results"):
            return None
        return pd.DataFrame(data["results"])

    import tempfile, shutil
    use_tmp = bool(results_suffix)
    tmp_dir = Path(tempfile.mkdtemp()) if use_tmp else None
    output_dir = tmp_dir if use_tmp else OUTPUT_DIR
    if use_tmp:
        output_dir.mkdir(parents=True, exist_ok=True)

    for variant in ["jsd", "informedness"]:
        variant_suffix = "jsd" if variant == "jsd" else "informedness-adaptive"

        # Denoising figure (planar only)
        df_den = _load("denoising", "planar", variant)
        if df_den is not None:
            fname = f"model-quality_denoising-iterations_validity_polyscore_all_mmd_{variant_suffix}.pdf"
            _plot_multi_panel(
                {"planar": df_den},
                x_label="Denoising Steps",
                output_path=output_dir / fname,
            )
        else:
            logger.warning("No denoising results for planar/{}", variant)

        # Training figure - all datasets
        training_data = {}
        for ds in DATASETS:
            df_train = _load("training", ds, variant)
            if df_train is not None:
                training_data[ds] = df_train

        if training_data:
            fname = f"all_training_epochs_{variant_suffix}.pdf"
            _plot_multi_panel(
                training_data,
                x_label="Training Epochs",
                output_path=output_dir / fname,
            )

            if "sbm" in training_data:
                fname_sbm = f"all_training_epochs_{variant_suffix}_sbm.pdf"
                _plot_multi_panel(
                    {"sbm": training_data["sbm"]},
                    x_label="Training Epochs",
                    output_path=output_dir / fname_sbm,
                )
        else:
            logger.warning("No training results for {}", variant)

    # Copy from temp dir with suffixed filenames
    if use_tmp and tmp_dir:
        for pdf in tmp_dir.glob("*.pdf"):
            dest = OUTPUT_DIR / (pdf.stem + results_suffix + pdf.suffix)
            shutil.copy2(pdf, dest)
            logger.info("Saved: {}", dest)
        shutil.rmtree(tmp_dir)

    if paper_dir is not None:
        paper_dir = Path(paper_dir)
        paper_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for pdf in OUTPUT_DIR.glob("*.pdf"):
            shutil.copy2(pdf, paper_dir / pdf.name)
            count += 1
        logger.success("Copied {} PDFs to {}", count, paper_dir)

    logger.success("Done.")


if __name__ == "__main__":
    app()
