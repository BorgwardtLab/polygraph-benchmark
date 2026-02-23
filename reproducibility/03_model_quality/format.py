#!/usr/bin/env python3
"""Format model quality results as LaTeX tables for the paper.

Produces 6 tables:
  1. digress-pearson-correlation-jsd.tex
  2. correlation-across-training-jsd.tex
  3. digress-pearson-correlation-informedness.tex
  4. correlation-across-training-informedness.tex
  5. digress-denoising-iters-mmd.tex
  6. digress-denoising-iters-pgs-jsd.tex

Usage:
    python format.py
    python format.py --paper-dir /path/to/paper/tables/
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import typer
from loguru import logger
from pyprojroot import here
from scipy import stats

app = typer.Typer()

REPO_ROOT = here()
RESULTS_DIR = REPO_ROOT / "reproducibility" / "figures" / "03_model_quality" / "results"
TABLES_DIR = REPO_ROOT / "reproducibility" / "tables"

DATASETS = ["planar", "sbm", "lobster"]
DATASET_DISPLAY = {
    "planar": "\\textsc{Planar-L}",
    "sbm": "\\textsc{SBM-L}",
    "lobster": "\\textsc{Lobster-L}",
}

MMD_KEYS = ["orbit_mmd", "orbit5_mmd", "degree_mmd", "spectral_mmd", "clustering_mmd", "gin_mmd"]
MMD_SHORT = {
    "orbit_mmd": "Orb.",
    "orbit5_mmd": "Orb5.",
    "degree_mmd": "Deg.",
    "spectral_mmd": "Eig.",
    "clustering_mmd": "Clust.",
    "gin_mmd": "GIN",
}

PGS_SUBSCORE_KEYS = ["orbit_pgs", "orbit5_pgs", "degree_pgs", "spectral_pgs", "clustering_pgs", "gin_pgs"]
PGS_SHORT = {
    "orbit_pgs": "Orb.",
    "orbit5_pgs": "Orb5.",
    "degree_pgs": "Deg.",
    "spectral_pgs": "Eig.",
    "clustering_pgs": "Clust.",
    "gin_pgs": "GIN",
}


def load_results(curve_type: str, dataset: str, variant: str) -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / f"{curve_type}_{dataset}_{variant}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if not data.get("results"):
        return None
    return pd.DataFrame(data["results"])


def _neg_pearson(x, y):
    """Negative Pearson correlation (higher = better correlation with validity)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    r, _ = stats.pearsonr(x[mask], y[mask])
    return -r


def _sign_adj_spearman(series, steps):
    """Sign-adjusted Spearman: positive means metric improves with more steps."""
    mask = np.isfinite(series) & np.isfinite(steps)
    if mask.sum() < 3:
        return np.nan
    rho, _ = stats.spearmanr(steps[mask], series[mask])
    return -rho


# ---------------------------------------------------------------------------
# Table 1 & 3: Pearson correlation of validity with other metrics
# ---------------------------------------------------------------------------

def generate_pearson_correlation_table(variant: str) -> str:
    """Generate Pearson correlation table between validity and other metrics."""
    variant_suffix = variant

    metric_cols = ["polyscore"] + MMD_KEYS
    metric_labels = ["PGS"] + [MMD_SHORT[k] + " MMD" for k in MMD_KEYS]

    lines = []
    lines.append("\\begin{tabular}{l" + "c" * len(metric_cols) + "}")
    lines.append("\\toprule")

    header = ["\\textbf{Experiment}"]
    for label in metric_labels:
        header.append(f"\\textbf{{{label}}}")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for curve_type, curve_label in [("denoising", "Denoising"), ("training", "Training")]:
        row_values = {col: [] for col in metric_cols}

        for ds in DATASETS:
            df = load_results(curve_type, ds, variant_suffix)
            if df is None or "validity" not in df.columns:
                continue
            validity = df["validity"].values
            for col in metric_cols:
                if col in df.columns:
                    corr = _neg_pearson(validity, df[col].values)
                    row_values[col].append(corr)

        row = [curve_label]
        for col in metric_cols:
            vals = row_values[col]
            if vals:
                mean_corr = np.nanmean(vals)
                row.append(f"{mean_corr:.2f}")
            else:
                row.append("-")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2 & 4: Spearman correlation with training iterations
# ---------------------------------------------------------------------------

def generate_spearman_training_table(variant: str) -> str:
    """Generate Spearman correlation table of metrics with training steps."""
    metric_cols = ["validity", "polyscore"] + MMD_KEYS
    metric_labels = ["Val."] + ["PGS"] + [MMD_SHORT[k] + " MMD" for k in MMD_KEYS]

    lines = []
    lines.append("\\begin{tabular}{l" + "c" * len(metric_cols) + "}")
    lines.append("\\toprule")

    header = ["\\textbf{Dataset}"]
    for label in metric_labels:
        header.append(f"\\textbf{{{label}}}")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for ds in DATASETS:
        df = load_results("training", ds, variant)
        if df is None:
            continue

        steps = df["steps"].values.astype(float)
        row = [DATASET_DISPLAY[ds]]

        for col in metric_cols:
            if col in df.columns:
                rho = _sign_adj_spearman(df[col].values.astype(float), steps)
                row.append(f"{rho:.2f}" if not np.isnan(rho) else "-")
            else:
                row.append("-")

        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 5: Denoising iterations MMD values
# ---------------------------------------------------------------------------

def generate_denoising_mmd_table() -> str:
    """Generate table of MMD values per denoising step."""
    df = load_results("denoising", "planar", "jsd")
    if df is None:
        return ""

    df = df.sort_values("steps")
    present_mmds = [k for k in MMD_KEYS if k in df.columns]
    mmd_labels = [MMD_SHORT[k] + " MMD" for k in present_mmds]

    lines = []
    lines.append("\\begin{tabular}{r" + "c" * len(present_mmds) + "}")
    lines.append("\\toprule")

    header = ["\\textbf{Steps}"]
    for label in mmd_labels:
        header.append(f"\\textbf{{{label}}}")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for _, row in df.iterrows():
        cells = [str(int(row["steps"]))]
        for k in present_mmds:
            val = row.get(k, np.nan)
            if np.isnan(val):
                cells.append("-")
            else:
                cells.append(f"{val:.4e}")
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 6: Denoising iterations PGS values
# ---------------------------------------------------------------------------

def generate_denoising_pgs_table() -> str:
    """Generate table of PGS values per denoising step."""
    df = load_results("denoising", "planar", "jsd")
    if df is None:
        return ""

    df = df.sort_values("steps")

    score_cols = ["polyscore"] + [k for k in PGS_SUBSCORE_KEYS if k in df.columns]
    score_labels = ["PGS"] + [PGS_SHORT.get(k, k) for k in PGS_SUBSCORE_KEYS if k in df.columns]

    lines = []
    lines.append("\\begin{tabular}{r" + "c" * len(score_cols) + "}")
    lines.append("\\toprule")

    header = ["\\textbf{Steps}"]
    for label in score_labels:
        header.append(f"\\textbf{{{label}}}")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for _, row in df.iterrows():
        cells = [str(int(row["steps"]))]
        for k in score_cols:
            val = row.get(k, np.nan)
            if np.isnan(val):
                cells.append("-")
            else:
                cells.append(f"{val * 100:.1f}")
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


@app.command()
def main(
    paper_dir: Optional[Path] = typer.Option(
        None, "--paper-dir", help="Copy tables into paper tables/ directory",
    ),
):
    """Generate all model quality LaTeX tables."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    table_map = {
        "digress-pearson-correlation-jsd.tex": lambda: generate_pearson_correlation_table("jsd"),
        "correlation-across-training-jsd.tex": lambda: generate_spearman_training_table("jsd"),
        "digress-pearson-correlation-informedness.tex": lambda: generate_pearson_correlation_table("informedness"),
        "correlation-across-training-informedness.tex": lambda: generate_spearman_training_table("informedness"),
        "digress-denoising-iters-mmd.tex": generate_denoising_mmd_table,
        "digress-denoising-iters-pgs-jsd.tex": generate_denoising_pgs_table,
    }

    for fname, generator in table_map.items():
        content = generator()
        if content:
            out = TABLES_DIR / fname
            out.write_text(content)
            logger.success("Saved: {}", out)
        else:
            logger.warning("No data for {}", fname)

    if paper_dir is not None:
        import shutil
        paper_dir = Path(paper_dir)
        paper_dir.mkdir(parents=True, exist_ok=True)
        for tex in TABLES_DIR.glob("*.tex"):
            if tex.stem in [k.replace(".tex", "") for k in table_map]:
                shutil.copy2(tex, paper_dir / tex.name)
        logger.success("Copied tables to {}", paper_dir)

    logger.success("Done.")


if __name__ == "__main__":
    app()
