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
from collections import OrderedDict
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
RESULTS_DIR = (
    REPO_ROOT / "reproducibility" / "figures" / "02_perturbation" / "results"
)
OUTPUT_DIR = REPO_ROOT / "reproducibility" / "figures" / "02_perturbation"
STYLE_FILE = Path(__file__).resolve().parent.parent / "polygraph.mplstyle"

DATASETS = ["planar", "lobster", "proteins", "sbm", "ego"]
PERTURBATIONS = [
    "edge_deletion",
    "edge_rewiring",
    "edge_swapping",
    "mixing",
    "edge_addition",
]

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


def _to_hex(color):
    """Convert matplotlib color tuple to hex string."""
    if isinstance(color, str):
        return color
    if hasattr(color, "__len__") and len(color) >= 3:
        return f"#{int(color[0] * 255):02x}{int(color[1] * 255):02x}{int(color[2] * 255):02x}"
    return "#000000"


def setup_plotting():
    if STYLE_FILE.exists():
        plt.style.use(str(STYLE_FILE))
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    # Override font sizes to match original notebook
    plt.rcParams.update(
        {
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


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
                "Perturbation": PERTURBATION_DISPLAY.get(
                    perturbation, perturbation
                ),
                "noise_level": noise,
                "PGS": agg_pgs,
            }
            for desc in DESCRIPTORS:
                key = f"{desc}_{classifier}_{variant}"
                base[f"PGS_{desc}"] = result_row.get(key, np.nan)

            for mmd_key in MMD_METRICS:
                base[mmd_key] = result_row.get(mmd_key, np.nan)

            rows.append(base)
    df = pd.DataFrame(rows)
    # Enforce ordering matching the original paper
    ds_order = [DATASET_DISPLAY.get(d, d) for d in DATASETS]
    pt_order = [PERTURBATION_DISPLAY.get(p, p) for p in PERTURBATIONS]
    df["Dataset"] = pd.Categorical(
        df["Dataset"], categories=ds_order, ordered=True
    )
    df["Perturbation"] = pd.Categorical(
        df["Perturbation"], categories=pt_order, ordered=True
    )
    return df


def _find_saturation_threshold(df_subset: pd.DataFrame) -> float:
    """Find noise level where aggregate PGS first exceeds 0.95."""
    sorted_df = df_subset.sort_values("noise_level")
    mask = sorted_df["PGS"] > 0.95
    if mask.any():
        return float(sorted_df.loc[mask.idxmax(), "noise_level"])
    return float(sorted_df["noise_level"].max())


def _crop_df(df: pd.DataFrame) -> pd.DataFrame:
    """Crop DataFrame to non-saturating PGS range per (dataset, perturbation)."""
    frames = []
    for (ds, pert), group in df.groupby(["dataset", "perturbation"]):  # type: ignore[misc]
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
    return float(rho)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Figure 1 & 2: Correlation bar plots
# ---------------------------------------------------------------------------


def plot_correlation_bars(
    all_data: Dict,
    classifier: str,
    variant: str,
    output_dir: Path,
) -> None:
    """Generate correlation jitter plot: one subplot per perturbation, dots per dataset."""
    df = _build_long_df(all_data, classifier, variant)
    if df.empty:
        logger.warning(
            "No data for correlation plot: clf={}, var={}", classifier, variant
        )
        return

    df_cropped = _crop_df(df)

    # Metrics matching the paper: 5 RBF MMD + aggregate PGD
    metric_cols = [
        "orbit_rbf",
        "degree_rbf",
        "spectral_rbf",
        "clustering_rbf",
        "gin_rbf",
        "PGS",
    ]
    metric_labels = [
        "Orbit RBF",
        "Deg. RBF",
        "Spec. RBF",
        "Clust. RBF",
        "GIN RBF",
        "PGD",
    ]

    # Descriptor-based coloring (matching paper's to_descriptor_color)
    cb = sns.color_palette("colorblind")
    metric_colors = {
        "orbit_rbf": cb[0],
        "degree_rbf": cb[1],
        "spectral_rbf": cb[2],
        "clustering_rbf": cb[3],
        "gin_rbf": cb[4],
        "PGS": "black",
    }

    present_perts = [
        p
        for p in PERTURBATIONS
        if PERTURBATION_DISPLAY[p] in df_cropped["Perturbation"].unique()
    ]
    present_datasets = [
        d
        for d in DATASETS
        if DATASET_DISPLAY[d] in df_cropped["Dataset"].unique()
    ]
    n_plots = len(present_perts)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(15, 3))
    if n_plots == 1:
        axes = [axes]

    for i, pert in enumerate(present_perts):
        ax = axes[i]
        ax.set_xticks(range(len(metric_cols)))
        ax.set_xticklabels(metric_labels, rotation=45)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(PERTURBATION_DISPLAY[pert])
        if i == 0:
            ax.set_ylabel("Spearman Corr.")

        for j, col in enumerate(metric_cols):
            xs, ys = [], []
            for k, ds in enumerate(present_datasets):
                group = df_cropped[
                    (df_cropped["perturbation"] == pert)
                    & (df_cropped["dataset"] == ds)
                ]
                if group.empty:
                    continue
                rho = _compute_spearman(group[col], group["noise_level"])  # type: ignore[arg-type]
                x_offset = k / len(present_datasets) * 0.6 - 0.3
                xs.append(j + x_offset)
                ys.append(rho if np.isfinite(rho) else 0)
            ax.scatter(xs, ys, color=metric_colors[col], s=15)

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
    """Faceted scatter grid matching paper: row=dataset, col=perturbation."""
    df = _build_long_df(all_data, classifier, variant)
    if df.empty:
        logger.warning(
            "No data for metrics_vs_noise: clf={}, var={}", classifier, variant
        )
        return

    # Note: don't remove data points for cropped/full - both plot all noise levels.
    # In the original paper, "cropped" vs "full" refers to different experimental
    # data sets, not filtering. Both have 100 noise levels plotted.

    # Descriptor colors matching original to_descriptor_color
    cb = sns.color_palette("colorblind")
    desc_colors = {
        "orbit": cb[0],
        "degree": cb[1],
        "spectral": cb[2],
        "clustering": cb[3],
        "gin": cb[4],
        "orbit5": cb[5],
    }

    # Metrics: per-descriptor PGS + aggregate (with markers matching paper)
    metrics = OrderedDict(
        [
            ("PGS_orbit", ("Orbit PGD", "o")),
            ("PGS_orbit5", ("Orbit5 PGD", "s")),
            ("PGS_degree", ("Degree PGD", "D")),
            ("PGS_spectral", ("Spectral PGD", "p")),
            ("PGS_clustering", ("Clustering PGD", "X")),
            ("PGS_gin", ("GIN PGD", "P")),
            ("PGS", ("PGD", "*")),
        ]
    )

    # Correlation per (dataset, perturbation)
    corr_map = {}
    for (ds, pert), group in df.groupby(["dataset", "perturbation"]):  # type: ignore[misc]
        corr_map[(ds, pert)] = _compute_spearman(
            group["PGS"],  # type: ignore[arg-type]
            group["noise_level"],  # type: ignore[arg-type]
        )

    # Build long-form scatter DataFrame
    plot_rows = []
    for _, row in df.iterrows():
        ds, pert = row["dataset"], row["perturbation"]
        corr = corr_map.get((ds, pert), np.nan)
        for col, (label, _) in metrics.items():
            val = row.get(col, np.nan)
            if pd.notna(val):  # type: ignore[arg-type]
                plot_rows.append(
                    {
                        "dataset": row["Dataset"],
                        "perturbation": row["Perturbation"],
                        "metric": label,
                        "noise_level": row["noise_level"],
                        "metric_value": val,
                        "correlation": corr,
                    }
                )

    plot_df = pd.DataFrame(plot_rows)
    if plot_df.empty:
        return

    # Enforce facet ordering matching the original paper
    ds_order = [DATASET_DISPLAY.get(d, d) for d in DATASETS]
    pt_order = [PERTURBATION_DISPLAY.get(p, p) for p in PERTURBATIONS]
    plot_df["dataset"] = pd.Categorical(
        plot_df["dataset"], categories=ds_order, ordered=True
    )
    plot_df["perturbation"] = pd.Categorical(
        plot_df["perturbation"], categories=pt_order, ordered=True
    )

    # Color and marker maps keyed by display label
    color_map = {}
    marker_map = {}
    for col, (label, marker) in metrics.items():
        marker_map[label] = marker
        if col == "PGS":
            color_map[label] = "#000000"
        else:
            desc = col.replace("PGS_", "")
            color_map[label] = _to_hex(desc_colors.get(desc, "black"))

    g = sns.relplot(
        data=plot_df,
        x="noise_level",
        y="metric_value",
        col="perturbation",
        row="dataset",
        hue="metric",
        style="metric",
        kind="scatter",
        markers=marker_map,
        height=3,
        aspect=0.8,
        s=20,
        alpha=0.8,
        palette=color_map,
        facet_kws={"margin_titles": True, "sharex": False},
    )

    g.set_xlabels("Noise Level")
    g.set_ylabels("PGD")
    g.set(ylim=(-0.1, 1.05))
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    # Add correlation annotation in subplot titles (matching paper)
    for (row_val, col_val), ax in g.axes_dict.items():
        sub = plot_df[
            (plot_df["dataset"] == row_val)
            & (plot_df["perturbation"] == col_val)
        ]
        if not sub.empty:
            corr = sub.iloc[0]["correlation"]
            title = ax.get_title()
            if not np.isnan(corr):
                ax.set_title(f"{title}\nPGD ρ = {corr:.2f}")

    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        title="Metric",
        title_fontsize=13,
    )
    plt.tight_layout(rect=(0, 0.06, 1, 1))

    variant_label = "jsd" if variant == "jsd" else "informedness"
    crop_label = "cropped" if cropped else "full"
    fname = (
        f"metrics_vs_noise_level_{variant_label}_{classifier}_{crop_label}.pdf"
    )
    out = output_dir / fname
    g.savefig(str(out), bbox_inches="tight")
    plt.close(g.figure)
    logger.success("Saved: {}", out)


# ---------------------------------------------------------------------------
# Figure 7: LR vs TabPFN comparison
# ---------------------------------------------------------------------------


def plot_lr_vs_tabpfn(
    all_data: Dict,
    variant: str,
    output_dir: Path,
) -> None:
    """LR vs TabPFN scatter comparison matching the paper."""
    df_tabpfn = _build_long_df(all_data, "tabpfn", variant)
    df_lr = _build_long_df(all_data, "lr", variant)

    if df_tabpfn.empty or df_lr.empty:
        logger.warning("Missing data for LR vs TabPFN comparison")
        return

    cb = sns.color_palette("colorblind")

    # Build scatter DataFrame (matching original: LR=colors[0], TabPFN=colors[1])
    plot_rows = []
    for _, row in df_lr.iterrows():
        plot_rows.append(
            {
                "dataset": row["Dataset"],
                "perturbation": row["Perturbation"],
                "metric": "LR PGD",
                "noise_level": row["noise_level"],
                "metric_value": row["PGS"],
            }
        )
    for _, row in df_tabpfn.iterrows():
        plot_rows.append(
            {
                "dataset": row["Dataset"],
                "perturbation": row["Perturbation"],
                "metric": "TABPFN PGD",
                "noise_level": row["noise_level"],
                "metric_value": row["PGS"],
            }
        )

    plot_df = pd.DataFrame(plot_rows)
    if plot_df.empty:
        return

    # Enforce facet ordering matching the original paper
    ds_order = [DATASET_DISPLAY.get(d, d) for d in DATASETS]
    pt_order = [PERTURBATION_DISPLAY.get(p, p) for p in PERTURBATIONS]
    plot_df["dataset"] = pd.Categorical(
        plot_df["dataset"], categories=ds_order, ordered=True
    )
    plot_df["perturbation"] = pd.Categorical(
        plot_df["perturbation"], categories=pt_order, ordered=True
    )

    markers = {"LR PGD": "o", "TABPFN PGD": "X"}
    colors = {"LR PGD": _to_hex(cb[0]), "TABPFN PGD": _to_hex(cb[1])}

    g = sns.relplot(
        data=plot_df,
        x="noise_level",
        y="metric_value",
        col="perturbation",
        row="dataset",
        hue="metric",
        style="metric",
        kind="scatter",
        markers=markers,
        height=3,
        aspect=0.8,
        s=20,
        alpha=0.8,
        palette=colors,
        facet_kws={"margin_titles": True, "sharex": False},
    )

    g.set_xlabels("Noise Level")
    g.set_ylabels("PGD")
    g.set(ylim=(-0.1, 1.05))
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        title="Metric",
        title_fontsize=13,
    )
    plt.tight_layout(rect=(0, 0.06, 1, 1))

    variant_label = "jsd" if variant == "jsd" else "informedness"
    fname = f"lr_vs_tabpfn_cropped_{variant_label}.pdf"
    out = output_dir / fname
    g.savefig(str(out), bbox_inches="tight")
    plt.close(g.figure)
    logger.success("Saved: {}", out)


# ---------------------------------------------------------------------------
# Single-dataset perturbation figures (e.g. SBM-only)
# ---------------------------------------------------------------------------


def plot_single_dataset_perturbation(
    all_data: Dict,
    classifier: str,
    variant: str,
    dataset: str,
    perturbations_filter: Optional[List[str]],
    output_dir: Path,
) -> None:
    """Generate a metrics-vs-noise figure for a single dataset.

    If perturbations_filter is None, all available perturbations are plotted
    in a single row (→ ``perturbation_all_{dataset}.pdf``).
    If a single perturbation is given, a single-panel figure is produced
    (→ ``perturbation_{perturbation}_{dataset}.pdf``).
    """
    # Filter data to requested dataset
    filtered = {k: v for k, v in all_data.items() if k[0] == dataset}
    if perturbations_filter:
        filtered = {
            k: v for k, v in filtered.items() if k[1] in perturbations_filter
        }
    if not filtered:
        logger.warning(
            "No data for dataset={}, perturbations={}",
            dataset,
            perturbations_filter,
        )
        return

    df = _build_long_df(filtered, classifier, variant)
    if df.empty:
        return

    # Descriptor colors
    cb = sns.color_palette("colorblind")
    desc_colors = {
        "orbit": cb[0],
        "degree": cb[1],
        "spectral": cb[2],
        "clustering": cb[3],
        "gin": cb[4],
        "orbit5": cb[5],
    }

    metrics = OrderedDict(
        [
            ("PGS_orbit", ("Orbit PGD", "o")),
            ("PGS_orbit5", ("Orbit5 PGD", "s")),
            ("PGS_degree", ("Degree PGD", "D")),
            ("PGS_spectral", ("Spectral PGD", "p")),
            ("PGS_clustering", ("Clustering PGD", "X")),
            ("PGS_gin", ("GIN PGD", "P")),
            ("PGS", ("PGD", "*")),
        ]
    )

    corr_map = {}
    for (ds, pert), group in df.groupby(["dataset", "perturbation"]):  # type: ignore[misc]
        corr_map[(ds, pert)] = _compute_spearman(
            group["PGS"],  # type: ignore[arg-type]
            group["noise_level"],  # type: ignore[arg-type]
        )

    plot_rows = []
    for _, row in df.iterrows():
        ds, pert = row["dataset"], row["perturbation"]
        corr = corr_map.get((ds, pert), np.nan)
        for col, (label, _) in metrics.items():
            val = row.get(col, np.nan)
            if pd.notna(val):  # type: ignore[arg-type]
                plot_rows.append(
                    {
                        "perturbation": row["Perturbation"],
                        "metric": label,
                        "noise_level": row["noise_level"],
                        "metric_value": val,
                        "correlation": corr,
                    }
                )

    plot_df = pd.DataFrame(plot_rows)
    if plot_df.empty:
        return

    pt_order = [PERTURBATION_DISPLAY.get(p, p) for p in PERTURBATIONS]
    plot_df["perturbation"] = pd.Categorical(
        plot_df["perturbation"], categories=pt_order, ordered=True
    )

    color_map = {}
    marker_map = {}
    for col, (label, marker) in metrics.items():
        marker_map[label] = marker
        if col == "PGS":
            color_map[label] = "#000000"
        else:
            desc = col.replace("PGS_", "")
            color_map[label] = _to_hex(desc_colors.get(desc, "black"))

    n_perts = plot_df["perturbation"].nunique()
    g = sns.relplot(
        data=plot_df,
        x="noise_level",
        y="metric_value",
        col="perturbation",
        hue="metric",
        style="metric",
        kind="scatter",
        markers=marker_map,
        height=3,
        aspect=0.8 if n_perts > 1 else 1.2,
        s=20,
        alpha=0.8,
        palette=color_map,
        facet_kws={"margin_titles": True, "sharex": False},
    )

    g.set_xlabels("Noise Level")
    g.set_ylabels("PGD")
    g.set(ylim=(-0.1, 1.05))
    g.set_titles(col_template="{col_name}")

    for col_val, ax in g.axes_dict.items():
        sub = plot_df[plot_df["perturbation"] == col_val]
        if not sub.empty:
            corr = sub.iloc[0]["correlation"]
            title = ax.get_title()
            if not np.isnan(corr):
                ax.set_title(f"{title}\nPGD ρ = {corr:.2f}")

    # Single-row figures need more bottom margin for the legend than multi-row grids
    bottom_margin = 0.18 if n_perts > 1 else 0.25
    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        title="Metric",
        title_fontsize=13,
    )
    plt.tight_layout(rect=(0, bottom_margin, 1, 1))

    if perturbations_filter and len(perturbations_filter) == 1:
        fname = f"perturbation_{perturbations_filter[0]}_{dataset}.pdf"
    else:
        fname = f"perturbation_all_{dataset}.pdf"
    out = output_dir / fname
    g.savefig(str(out), bbox_inches="tight")
    plt.close(g.figure)
    logger.success("Saved: {}", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_results_dir(results_dir: Path) -> Dict[Tuple[str, str], dict]:
    """Load all perturbation JSON results from a directory."""
    data = {}
    if not results_dir.exists():
        return data
    for f in sorted(results_dir.glob("perturbation_*.json")):
        try:
            d = json.loads(f.read_text())
            key = (d["dataset"], d["perturbation"])
            data[key] = d
        except Exception as e:
            logger.warning("Skipping {}: {}", f.name, e)
    return data


@app.command()
def main(
    paper_dir: Optional[Path] = typer.Option(
        None,
        "--paper-dir",
        help="Also copy outputs into this directory (e.g. paper figures/perturbation_experiments/)",
    ),
    results_suffix: str = typer.Option(
        "",
        "--results-suffix",
        help="Suffix for results dir and output files (e.g. _tabpfn_v6)",
    ),
):
    """Generate all perturbation figures for the paper."""
    setup_plotting()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Full-range results (noise in [0, 1])
    results_dir_full = (
        REPO_ROOT
        / "reproducibility"
        / "figures"
        / "02_perturbation"
        / f"results{results_suffix}"
    )
    # Cropped results (noise up to PGS saturation, dense sampling)
    results_dir_cropped = (
        REPO_ROOT
        / "reproducibility"
        / "figures"
        / "02_perturbation"
        / "results_cropped"
    )

    import tempfile
    import shutil

    use_tmp = bool(results_suffix)
    tmp_dir = Path(tempfile.mkdtemp()) if use_tmp else None
    output_dir: Path = tmp_dir if tmp_dir is not None else OUTPUT_DIR
    if use_tmp:
        output_dir.mkdir(parents=True, exist_ok=True)

    data_full = _load_results_dir(results_dir_full)
    data_cropped = _load_results_dir(results_dir_cropped)

    if not data_full:
        logger.error(
            "No full-range results found in {}. Run compute.py first.",
            results_dir_full,
        )
        return

    # Fall back to full data if cropped data is unavailable
    if not data_cropped:
        logger.warning(
            "No cropped results found in {}; using full data for cropped plots.",
            results_dir_cropped,
        )
        data_cropped = data_full

    logger.info(
        "Loaded {} full + {} cropped result files",
        len(data_full),
        len(data_cropped),
    )

    for variant in ["jsd", "informedness"]:
        plot_correlation_bars(data_cropped, "tabpfn", variant, output_dir)
        plot_metrics_vs_noise(
            data_full, "tabpfn", variant, cropped=False, output_dir=output_dir
        )
        plot_metrics_vs_noise(
            data_cropped, "tabpfn", variant, cropped=True, output_dir=output_dir
        )

    for variant in ["jsd", "informedness"]:
        plot_lr_vs_tabpfn(data_cropped, variant, output_dir)

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


@app.command()
def single_dataset(
    dataset: str = typer.Argument(
        ..., help="Dataset name (e.g. sbm, planar, lobster)"
    ),
    results_suffix: str = typer.Option(
        "",
        "--results-suffix",
        help="Suffix for results dir (e.g. _tabpfn_weights_v2.5)",
    ),
    perturbation: Optional[str] = typer.Option(
        None, "--perturbation", help="Single perturbation type; omit for all"
    ),
    classifier: str = typer.Option("tabpfn", "--classifier"),
    variant: str = typer.Option("jsd", "--variant"),
):
    """Generate perturbation figures for a single dataset."""
    setup_plotting()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_dir = (
        REPO_ROOT
        / "reproducibility"
        / "figures"
        / "02_perturbation"
        / f"results{results_suffix}"
    )
    all_data = _load_results_dir(results_dir)
    if not all_data:
        logger.error("No results found in {}", results_dir)
        return

    perts = [perturbation] if perturbation else None
    plot_single_dataset_perturbation(
        all_data, classifier, variant, dataset, perts, OUTPUT_DIR
    )


if __name__ == "__main__":
    app()
