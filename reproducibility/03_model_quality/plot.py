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
"""

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from pyprojroot import here

app = typer.Typer()

REPO_ROOT = here()
RESULTS_DIR = (
    REPO_ROOT / "reproducibility" / "figures" / "03_model_quality" / "results"
)
OUTPUT_DIR = REPO_ROOT / "reproducibility" / "figures" / "03_model_quality"
STYLE_FILE = Path(__file__).resolve().parent.parent / "polygraph.mplstyle"

DATASETS = ["planar", "sbm", "lobster"]
DATASET_DISPLAY = {"planar": "Planar-L", "sbm": "SBM-L", "lobster": "Lobster-L"}

MMD_KEYS = [
    "orbit_mmd",
    "orbit5_mmd",
    "degree_mmd",
    "spectral_mmd",
    "clustering_mmd",
    "gin_mmd",
]
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
    # Override font sizes to match original notebook
    plt.rcParams.update(
        {
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def load_results(
    curve_type: str, dataset: str, variant: str
) -> Optional[pd.DataFrame]:
    """Load JSON results for a specific (curve_type, dataset, variant)."""
    path = RESULTS_DIR / f"{curve_type}_{dataset}_{variant}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if not data.get("results"):
        return None
    return pd.DataFrame(data["results"])


def _plot_multi_panel(
    data_dict: Dict[str, pd.DataFrame],
    x_label: str,
    output_path: Path,
    pgd_label: str = "PGD",
) -> None:
    """Create a multi-panel figure matching the paper: columns for Validity, PGD (if available), MMD twinx."""
    datasets = list(data_dict.keys())
    n_datasets = len(datasets)
    if n_datasets == 0:
        return

    single = n_datasets == 1
    palette = sns.color_palette("colorblind")
    descriptor_colors = {
        "orbit": palette[0],
        "degree": palette[1],
        "spectral": palette[2],
        "clustering": palette[3],
        "gin": palette[4],
    }
    mmd_metrics = [
        "orbit_mmd",
        "degree_mmd",
        "spectral_mmd",
        "clustering_mmd",
        "gin_mmd",
    ]

    # Determine if polyscore is available in any dataset
    has_polyscore = any("polyscore" in df.columns for df in data_dict.values())
    n_cols = 3 if has_polyscore else 2

    if single:
        fig, axes = plt.subplots(
            1, n_cols, figsize=(8 if has_polyscore else 5.5, 1.6), squeeze=False
        )
    else:
        fig, axes = plt.subplots(
            n_datasets,
            n_cols,
            figsize=(8 if has_polyscore else 5.5, n_datasets * 1.5),
            squeeze=False,
        )

    # Build legend elements
    legend_elements = [
        Line2D([0], [0], color="#7e9ef7", lw=2, label="Validity"),
    ]
    if has_polyscore:
        legend_elements.append(
            Line2D([0], [0], color="black", lw=2, label=pgd_label)
        )
    for metric in mmd_metrics:
        desc = metric.replace("_mmd", "")
        color = descriptor_colors.get(desc, "black")
        legend_elements.append(
            Line2D([0], [0], color=color, lw=2, label=f"{desc.title()} RBF")
        )

    for i, ds in enumerate(datasets):
        df = data_dict[ds].sort_values("steps")
        steps = df["steps"].values
        label = DATASET_DISPLAY.get(ds, ds)

        # Dataset label on left margin (multi-dataset only)
        if not single and label:
            axes[i, 0].annotate(
                label,
                xy=(0, 0.5),
                xycoords="axes fraction",
                xytext=(-60, 0),
                textcoords="offset points",
                rotation=90,
                va="center",
                fontsize=12,
                fontweight="bold",
                annotation_clip=False,
            )

        col_idx = 0

        # Column: Validity
        ax_val = axes[i, col_idx]
        ax_val.plot(steps, df["validity"].values, color="#7e9ef7")
        ax_val.set_ylabel("Validity")
        ax_val.set_ylim([0, 1])
        ax_val.yaxis.set_major_locator(MaxNLocator(6))
        ax_val.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        if i == 0:
            ax_val.set_title("Validity")
        if i == n_datasets - 1:
            ax_val.set_xlabel(x_label)
        col_idx += 1

        # Column: PGD (only if data has it)
        if has_polyscore:
            ax_pgs = axes[i, col_idx]
            if "polyscore" in df.columns:
                ax_pgs.plot(steps, df["polyscore"].values, color="black")
            ax_pgs.set_ylabel("PGD")
            ax_pgs.set_ylim([0, 1])
            ax_pgs.yaxis.set_major_locator(MaxNLocator(6))
            ax_pgs.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
            if i == 0:
                ax_pgs.set_title(pgd_label)
            if i == n_datasets - 1:
                ax_pgs.set_xlabel(x_label)
            col_idx += 1

        # Column: Combined MMD with twinx per metric
        ax_mmd = axes[i, col_idx]
        ax_mmd.set_yticks([])
        ax_mmd.set_ylabel("MMD\u00b2")
        ax_mmd.spines["left"].set_visible(False)
        ax_mmd.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        if i == 0:
            ax_mmd.set_title("MMD")
        if i == n_datasets - 1:
            ax_mmd.set_xlabel(x_label)

        present_metrics = [m for m in mmd_metrics if m in df.columns]
        for j, metric in enumerate(present_metrics):
            desc = metric.replace("_mmd", "")
            color = descriptor_colors.get(desc, "black")
            metric_data = df[metric].values

            ax_twin = ax_mmd.twinx()
            ax_twin.spines["right"].set_position(("outward", 35 * j))

            max_val = np.nanmax(metric_data)  # type: ignore[arg-type]
            if max_val > 0:
                power = int(np.floor(np.log10(max_val)))
                scale_factor = 10**power
                scaled = metric_data / scale_factor
                ax_twin.plot(steps, scaled, color=color)
                ax_twin.tick_params(axis="y", labelcolor=color)
                ax_twin.yaxis.set_major_locator(MaxNLocator(6))
                ax_twin.annotate(
                    f"$\\times 10^{{{power}}}$",
                    xy=(1.0, 1.0),
                    xycoords="axes fraction",
                    xytext=(35 * j, 4),
                    textcoords="offset points",
                    color=color,
                    fontsize=10,
                    ha="center",
                    va="bottom",
                    annotation_clip=False,
                )
            else:
                ax_twin.plot(steps, metric_data, color=color)
                ax_twin.tick_params(axis="y", labelcolor=color)
                ax_twin.yaxis.set_major_locator(MaxNLocator(6))

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
    )
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    logger.success("Saved: {}", output_path)


@app.command()
def main(
    results_suffix: str = typer.Option(
        "",
        "--results-suffix",
        help="Suffix for results dir and output files (e.g. _tabpfn_v6)",
    ),
):
    """Generate all model quality figures for the paper."""
    setup_plotting()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_dir = (
        REPO_ROOT
        / "reproducibility"
        / "figures"
        / "03_model_quality"
        / f"results{results_suffix}"
    )

    def _load(
        curve_type: str, dataset: str, variant: str
    ) -> Optional[pd.DataFrame]:
        path = results_dir / f"{curve_type}_{dataset}_{variant}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        if not data.get("results"):
            return None
        return pd.DataFrame(data["results"])

    import tempfile
    import shutil

    use_tmp = bool(results_suffix)
    tmp_dir = Path(tempfile.mkdtemp()) if use_tmp else None
    output_dir: Path = tmp_dir if tmp_dir is not None else OUTPUT_DIR
    if use_tmp:
        output_dir.mkdir(parents=True, exist_ok=True)

    for variant in ["jsd", "informedness"]:
        variant_suffix = "jsd" if variant == "jsd" else "informedness-adaptive"
        pgd_label = "PolyGraph Discrepancy"

        # Denoising figure (planar only)
        df_den = _load("denoising", "planar", variant)
        if df_den is not None:
            fname = f"model-quality_denoising-iterations_validity_polyscore_all_mmd_{variant_suffix}.pdf"
            _plot_multi_panel(
                {"planar": df_den},
                x_label="# Denoising Steps",
                output_path=output_dir / fname,
                pgd_label=pgd_label,
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
                x_label="# Epochs",
                output_path=output_dir / fname,
                pgd_label=pgd_label,
            )

            if "sbm" in training_data:
                fname_sbm = f"all_training_epochs_{variant_suffix}_sbm.pdf"
                _plot_multi_panel(
                    {"sbm": training_data["sbm"]},
                    x_label="# Epochs",
                    output_path=output_dir / fname_sbm,
                    pgd_label=pgd_label,
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

    logger.success("Done.")


if __name__ == "__main__":
    app()
