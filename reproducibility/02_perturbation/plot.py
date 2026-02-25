#!/usr/bin/env python3
"""Generate all perturbation figures for the paper.

Reads pre-computed JSON results from compute.py and produces the exact PDF
figures referenced in the LaTeX source:

  1. correlation_plots_jsd_tabpfn_cropped.pdf
  2. correlation_plots_informedness_tabpfn_cropped.pdf
  3. metrics_vs_noise_level_jsd_tabpfn_full.pdf
  4. metrics_vs_noise_level_jsd_tabpfn_cropped.pdf
  5. metrics_vs_noise_level_informedness_tabpfn_full.pdf
  6. metrics_vs_noise_level_informedness_tabpfn_cropped.pdf
  7. lr_vs_tabpfn_cropped_jsd.pdf
  8. lr_vs_tabpfn_cropped_informedness.pdf

Usage:
    python plot.py
    python plot.py --paper-dir /path/to/paper/figures/perturbation_experiments/
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from pyprojroot import here
from scipy import stats

app = typer.Typer()

REPO_ROOT = here()
RESULTS_DIR = REPO_ROOT / "reproducibility" / "figures" / "02_perturbation" / "results"
OUTPUT_DIR = REPO_ROOT / "reproducibility" / "figures" / "02_perturbation"
STYLE_FILE = Path(__file__).resolve().parent.parent / "polygraph.mplstyle"

DATASETS = ["sbm", "planar", "lobster", "proteins", "ego"]
PERTURBATIONS = ["edge_rewiring", "edge_swapping", "mixing", "edge_deletion", "edge_addition"]

DATASET_DISPLAY = {
    "sbm": "SBM",
    "planar": "Planar",
    "lobster": "Lobster",
    "proteins": "Proteins",
    "ego": "Ego",
}

PERTURBATION_DISPLAY = {
    "edge_rewiring": "Edge Rewiring",
    "edge_swapping": "Edge Swapping",
    "mixing": "Mixing",
    "edge_deletion": "Edge Deletion",
    "edge_addition": "Edge Addition",
}

DESCRIPTORS = ["orbit", "orbit5", "degree", "spectral", "clustering", "gin"]

DESCRIPTOR_DISPLAY = {
    "orbit": "Orbit",
    "orbit5": "Orbit 5",
    "degree": "Degree",
    "spectral": "Spectral",
    "clustering": "Clustering",
    "gin": "GIN",
}

MMD_METRICS = [
    "orbit_tv", "degree_tv", "spectral_tv", "clustering_tv",
    "orbit_rbf", "orbit5_rbf", "degree_rbf", "spectral_rbf",
    "clustering_rbf", "gin_rbf",
]

MMD_DISPLAY = {
    "orbit_tv": "Orb. GTV",
    "degree_tv": "Deg. GTV",
    "spectral_tv": "Eig. GTV",
    "clustering_tv": "Clust. GTV",
    "orbit_rbf": "Orb. RBF",
    "orbit5_rbf": "Orb5. RBF",
    "degree_rbf": "Deg. RBF",
    "spectral_rbf": "Eig. RBF",
    "clustering_rbf": "Clust. RBF",
    "gin_rbf": "GIN RBF",
}


def setup_plotting():
    if STYLE_FILE.exists():
        plt.style.use(str(STYLE_FILE))
    sns.set_style("ticks")
    sns.set_palette("colorblind")


def load_all_results() -> Dict[Tuple[str, str], dict]:
    """Load all perturbation JSON results keyed by (dataset, perturbation)."""
    data = {}
    if not RESULTS_DIR.exists():
        return data
    for f in sorted(RESULTS_DIR.glob("perturbation_*.json")):
        try:
            d = json.loads(f.read_text())
            key = (d["dataset"], d["perturbation"])
            data[key] = d
        except Exception as e:
            logger.warning("Skipping {}: {}", f.name, e)
    return data


def _compute_aggregate_pgs(row: dict, classifier: str, variant: str) -> float:
    """Compute aggregate PGS = max over descriptors for a given classifier and variant."""
    values = []
    for desc in DESCRIPTORS:
        key = f"{desc}_{classifier}_{variant}"
        if key in row and not np.isnan(row[key]):
            values.append(row[key])
    return max(values) if values else np.nan


def _build_long_df(
    all_data: Dict[Tuple[str, str], dict],
    classifier: str,
    variant: str,
) -> pd.DataFrame:
    """Build a long-form DataFrame with columns for plotting."""
    rows = []
    for (dataset, perturbation), d in all_data.items():
        for result_row in d["results"]:
            noise = result_row["noise_level"]
            agg_pgs = _compute_aggregate_pgs(result_row, classifier, variant)

            base = {
                "dataset": dataset,
                "Dataset": DATASET_DISPLAY.get(dataset, dataset),
                "perturbation": perturbation,
                "Perturbation": PERTURBATION_DISPLAY.get(perturbation, perturbation),
                "noise_level": noise,
                "PGS": agg_pgs,
            }
            for desc in DESCRIPTORS:
                key = f"{desc}_{classifier}_{variant}"
                base[f"PGS_{desc}"] = result_row.get(key, np.nan)

            for mmd_key in MMD_METRICS:
                base[mmd_key] = result_row.get(mmd_key, np.nan)

            rows.append(base)
    return pd.DataFrame(rows)


def _find_saturation_threshold(df_subset: pd.DataFrame) -> float:
    """Find noise level where aggregate PGS first exceeds 0.95."""
    sorted_df = df_subset.sort_values("noise_level")
    mask = sorted_df["PGS"] > 0.95
    if mask.any():
        return sorted_df.loc[mask.idxmax(), "noise_level"]
    return sorted_df["noise_level"].max()


def _crop_df(df: pd.DataFrame) -> pd.DataFrame:
    """Crop DataFrame to non-saturating PGS range per (dataset, perturbation)."""
    frames = []
    for (ds, pert), group in df.groupby(["dataset", "perturbation"]):
        threshold = _find_saturation_threshold(group)
        frames.append(group[group["noise_level"] <= threshold])
    if not frames:
        return df
    return pd.concat(frames, ignore_index=True)


def _compute_spearman(series: pd.Series, noise: pd.Series) -> float:
    """Compute Spearman correlation of a metric with noise_level."""
    valid = series.notna() & noise.notna()
    if valid.sum() < 3:
        return np.nan
    rho, _ = stats.spearmanr(noise[valid], series[valid])
    return rho


# ---------------------------------------------------------------------------
# Figure 1 & 2: Correlation bar plots
# ---------------------------------------------------------------------------

def plot_correlation_bars(
    all_data: Dict, classifier: str, variant: str, output_dir: Path,
) -> None:
    """Generate correlation bar chart (one per variant)."""
    df = _build_long_df(all_data, classifier, variant)
    if df.empty:
        logger.warning("No data for correlation plot: clf={}, var={}", classifier, variant)
        return

    df_cropped = _crop_df(df)

    metric_cols = [f"PGS_{d}" for d in DESCRIPTORS] + ["PGS"] + MMD_METRICS
    metric_labels = (
        [DESCRIPTOR_DISPLAY[d] for d in DESCRIPTORS]
        + ["PGS (agg.)"]
        + [MMD_DISPLAY[m] for m in MMD_METRICS]
    )

    correlations = []
    for (ds, pert), group in df_cropped.groupby(["dataset", "perturbation"]):
        noise = group["noise_level"]
        for col, label in zip(metric_cols, metric_labels):
            rho = _compute_spearman(group[col], noise)
            correlations.append({
                "Metric": label,
                "Dataset": DATASET_DISPLAY.get(ds, ds),
                "Perturbation": PERTURBATION_DISPLAY.get(pert, pert),
                "Spearman ρ": rho,
            })

    corr_df = pd.DataFrame(correlations).dropna(subset=["Spearman ρ"])
    if corr_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 3.5))
    order = metric_labels
    present = [m for m in order if m in corr_df["Metric"].unique()]
    sns.boxplot(
        data=corr_df, x="Metric", y="Spearman ρ", order=present,
        ax=ax, palette="colorblind", linewidth=0.8, fliersize=2,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Spearman ρ")
    ax.set_xlabel("")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    sns.despine()
    plt.tight_layout()

    variant_label = "jsd" if variant == "jsd" else "informedness"
    fname = f"correlation_plots_{variant_label}_{classifier}_cropped.pdf"
    out = output_dir / fname
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    logger.success("Saved: {}", out)


# ---------------------------------------------------------------------------
# Figures 3-6: Metrics vs noise level (faceted grid)
# ---------------------------------------------------------------------------

def plot_metrics_vs_noise(
    all_data: Dict,
    classifier: str,
    variant: str,
    cropped: bool,
    output_dir: Path,
) -> None:
    """Generate faceted grid: rows=perturbation, cols=dataset, lines=descriptor PGS."""
    df = _build_long_df(all_data, classifier, variant)
    if df.empty:
        logger.warning("No data for metrics_vs_noise: clf={}, var={}", classifier, variant)
        return

    if cropped:
        df = _crop_df(df)

    palette = sns.color_palette("colorblind", n_colors=len(DESCRIPTORS) + 1)
    color_map = {DESCRIPTOR_DISPLAY[d]: palette[i] for i, d in enumerate(DESCRIPTORS)}
    color_map["PGS (agg.)"] = palette[len(DESCRIPTORS)]

    present_perts = [p for p in PERTURBATIONS if PERTURBATION_DISPLAY[p] in df["Perturbation"].unique()]
    present_datasets = [d for d in DATASETS if DATASET_DISPLAY[d] in df["Dataset"].unique()]

    n_rows = len(present_perts)
    n_cols = len(present_datasets)
    if n_rows == 0 or n_cols == 0:
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2 * n_rows), squeeze=False)

    for i, pert in enumerate(present_perts):
        for j, ds in enumerate(present_datasets):
            ax = axes[i, j]
            subset = df[
                (df["perturbation"] == pert) & (df["dataset"] == ds)
            ].sort_values("noise_level")

            if subset.empty:
                ax.set_visible(False)
                continue

            noise = subset["noise_level"].values

            for desc in DESCRIPTORS:
                col = f"PGS_{desc}"
                if col in subset.columns:
                    vals = subset[col].values
                    ax.plot(noise, vals, linewidth=1, alpha=0.7,
                            color=color_map[DESCRIPTOR_DISPLAY[desc]],
                            label=DESCRIPTOR_DISPLAY[desc])

            agg = subset["PGS"].values
            ax.plot(noise, agg, linewidth=2, color=color_map["PGS (agg.)"],
                    label="PGS (agg.)")

            if cropped:
                rho = _compute_spearman(pd.Series(agg), pd.Series(noise))
                if not np.isnan(rho):
                    ax.text(0.95, 0.05, f"ρ={rho:.2f}", transform=ax.transAxes,
                            ha="right", va="bottom", fontsize=7,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

            ax.set_ylim(-0.05, 1.05)

            if i == 0:
                ax.set_title(DATASET_DISPLAY[ds], fontsize=9)
            if j == 0:
                ax.set_ylabel(PERTURBATION_DISPLAY[pert], fontsize=8)
            if i == n_rows - 1:
                ax.set_xlabel("Noise Level", fontsize=8)

            ax.tick_params(labelsize=7)
            sns.despine(ax=ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(7, len(handles)),
                   fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.08)

    variant_label = "jsd" if variant == "jsd" else "informedness"
    crop_label = "cropped" if cropped else "full"
    fname = f"metrics_vs_noise_level_{variant_label}_{classifier}_{crop_label}.pdf"
    out = output_dir / fname
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    logger.success("Saved: {}", out)


# ---------------------------------------------------------------------------
# Figure 7: LR vs TabPFN comparison
# ---------------------------------------------------------------------------

def plot_lr_vs_tabpfn(
    all_data: Dict, variant: str, output_dir: Path,
) -> None:
    """Generate LR vs TabPFN comparison on cropped range."""
    df_tabpfn = _build_long_df(all_data, "tabpfn", variant)
    df_lr = _build_long_df(all_data, "lr", variant)

    if df_tabpfn.empty or df_lr.empty:
        logger.warning("Missing data for LR vs TabPFN comparison")
        return

    df_tabpfn = _crop_df(df_tabpfn)
    df_lr_cropped = df_lr[
        df_lr.set_index(["dataset", "perturbation", "noise_level"]).index.isin(
            df_tabpfn.set_index(["dataset", "perturbation", "noise_level"]).index
        )
    ]

    palette = sns.color_palette("colorblind")

    present_perts = [p for p in PERTURBATIONS if PERTURBATION_DISPLAY[p] in df_tabpfn["Perturbation"].unique()]
    present_datasets = [d for d in DATASETS if DATASET_DISPLAY[d] in df_tabpfn["Dataset"].unique()]

    n_rows = len(present_perts)
    n_cols = len(present_datasets)
    if n_rows == 0 or n_cols == 0:
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2 * n_rows), squeeze=False)

    for i, pert in enumerate(present_perts):
        for j, ds in enumerate(present_datasets):
            ax = axes[i, j]
            sub_t = df_tabpfn[
                (df_tabpfn["perturbation"] == pert) & (df_tabpfn["dataset"] == ds)
            ].sort_values("noise_level")
            sub_l = df_lr_cropped[
                (df_lr_cropped["perturbation"] == pert) & (df_lr_cropped["dataset"] == ds)
            ].sort_values("noise_level")

            if sub_t.empty:
                ax.set_visible(False)
                continue

            ax.plot(sub_t["noise_level"].values, sub_t["PGS"].values,
                    linewidth=1.5, color=palette[0], label="PGD (TabPFN)")
            if not sub_l.empty:
                ax.plot(sub_l["noise_level"].values, sub_l["PGS"].values,
                        linewidth=1.5, color=palette[1], linestyle="--", label="LR PGD")

            ax.set_ylim(-0.05, 1.05)

            if i == 0:
                ax.set_title(DATASET_DISPLAY[ds], fontsize=9)
            if j == 0:
                ax.set_ylabel(PERTURBATION_DISPLAY[pert], fontsize=8)
            if i == n_rows - 1:
                ax.set_xlabel("Noise Level", fontsize=8)

            ax.tick_params(labelsize=7)
            sns.despine(ax=ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2,
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.08)

    variant_label = "jsd" if variant == "jsd" else "informedness"
    fname = f"lr_vs_tabpfn_cropped_{variant_label}.pdf"
    out = output_dir / fname
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    logger.success("Saved: {}", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    paper_dir: Optional[Path] = typer.Option(
        None, "--paper-dir",
        help="Also copy outputs into this directory (e.g. paper figures/perturbation_experiments/)",
    ),
    results_suffix: str = typer.Option("", "--results-suffix", help="Suffix for results dir and output files (e.g. _tabpfn_v6)"),
):
    """Generate all perturbation figures for the paper."""
    setup_plotting()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_dir = REPO_ROOT / "reproducibility" / "figures" / "02_perturbation" / f"results{results_suffix}"

    import tempfile, shutil
    use_tmp = bool(results_suffix)
    tmp_dir = Path(tempfile.mkdtemp()) if use_tmp else None
    output_dir = tmp_dir if use_tmp else OUTPUT_DIR
    if use_tmp:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load from the (possibly suffixed) results dir
    data = {}
    if not results_dir.exists():
        logger.error("No perturbation results found in {}. Run compute.py first.", results_dir)
        return
    for f in sorted(results_dir.glob("perturbation_*.json")):
        try:
            d = json.loads(f.read_text())
            key = (d["dataset"], d["perturbation"])
            data[key] = d
        except Exception as e:
            logger.warning("Skipping {}: {}", f.name, e)
    if not data:
        logger.error("No perturbation results found in {}", results_dir)
        return

    logger.info("Loaded {} result files from {}", len(data), results_dir)

    for variant in ["jsd", "informedness"]:
        plot_correlation_bars(data, "tabpfn", variant, output_dir)
        plot_metrics_vs_noise(data, "tabpfn", variant, cropped=False, output_dir=output_dir)
        plot_metrics_vs_noise(data, "tabpfn", variant, cropped=True, output_dir=output_dir)

    for variant in ["jsd", "informedness"]:
        plot_lr_vs_tabpfn(data, variant, output_dir)

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
