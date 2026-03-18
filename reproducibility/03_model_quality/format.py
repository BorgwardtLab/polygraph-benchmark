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
"""

import json
from itertools import groupby
from typing import Optional

import numpy as np
import pandas as pd
import typer
from loguru import logger
from pyprojroot import here
from scipy import stats

app = typer.Typer()

REPO_ROOT = here()
RESULTS_DIR = (
    REPO_ROOT / "reproducibility" / "figures" / "03_model_quality" / "results"
)
TABLES_DIR = REPO_ROOT / "reproducibility" / "tables"

DATASETS = ["planar", "sbm", "lobster"]
DATASET_DISPLAY = {
    "planar": "\\textsc{Planar-L}",
    "sbm": "\\textsc{SBM-L}",
    "lobster": "\\textsc{Lobster-L}",
}

MMD_KEYS = [
    "orbit_mmd",
    "degree_mmd",
    "spectral_mmd",
    "clustering_mmd",
    "gin_mmd",
]
MMD_SHORT = {
    "orbit_mmd": "Orb.",
    "orbit5_mmd": "Orb5.",
    "degree_mmd": "Deg.",
    "spectral_mmd": "Eig.",
    "clustering_mmd": "Clust.",
    "gin_mmd": "GIN",
}
MMD_SUFFIX = "RBF"

PGS_SUBSCORE_KEYS = [
    "orbit4_pgs",
    "orbit5_pgs",
    "degree_pgs",
    "spectral_pgs",
    "clustering_pgs",
    "gin_pgs",
]
PGS_SHORT = {
    "orbit4_pgs": "Orb.",
    "orbit5_pgs": "Orb5.",
    "degree_pgs": "Deg.",
    "spectral_pgs": "Eig.",
    "clustering_pgs": "Clust.",
    "gin_pgs": "GIN",
}


def load_results(
    curve_type: str, dataset: str, variant: str
) -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / f"{curve_type}_{dataset}_{variant}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if not data.get("results"):
        return None
    return pd.DataFrame(data["results"])


def _neg_pearson(x, y) -> float:
    """Negative Pearson correlation (higher = better correlation with validity)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float(np.nan)
    r, _ = stats.pearsonr(x[mask], y[mask])
    return -float(r)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Table 1 & 3: Pearson correlation of validity with other metrics
# ---------------------------------------------------------------------------


def _format_row_with_ranking(
    values: list[float], fmt: str = "{:.2f}"
) -> list[str]:
    """Format values with bold for best and underline for second-best per row."""
    numeric = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
    if len(numeric) < 2:
        return [fmt.format(v) if not np.isnan(v) else "-" for v in values]
    sorted_vals = sorted(numeric, key=lambda x: x[1], reverse=True)
    best_idx = sorted_vals[0][0]
    second_idx = sorted_vals[1][0]
    cells = []
    for i, v in enumerate(values):
        if np.isnan(v):
            cells.append("-")
        elif i == best_idx:
            cells.append(f"\\textbf{{{fmt.format(v)}}}")
        elif i == second_idx:
            cells.append(f"\\underline{{{fmt.format(v)}}}")
        else:
            cells.append(fmt.format(v))
    return cells


def generate_pearson_correlation_table(variant: str) -> str:
    """Generate Pearson correlation table between validity and other metrics.

    Shows per-dataset rows grouped by experiment type, with values multiplied
    by 100. Bold = best metric per row, underline = second best.
    """
    variant_suffix = variant

    metric_cols = ["polyscore"] + MMD_KEYS
    pgd_label = "TV-PGD" if variant == "informedness" else "PGD"
    metric_labels = [pgd_label] + [
        MMD_SHORT[k] + f" {MMD_SUFFIX}" for k in MMD_KEYS
    ]

    # Collect all (curve_type, dataset, values) rows
    rows: list[tuple[str, str, list[float]]] = []
    for curve_type in ["denoising", "training"]:
        for ds in DATASETS:
            df = load_results(curve_type, ds, variant_suffix)
            if df is None or "validity" not in df.columns:
                continue
            validity = df["validity"].values
            values = []
            for col in metric_cols:
                if col in df.columns:
                    values.append(_neg_pearson(validity, df[col].values) * 100)
                else:
                    values.append(np.nan)
            rows.append((curve_type, ds, values))

    # Group rows by curve type
    lines = []
    lines.append("\\begin{tabular}{ll" + "c" * len(metric_cols) + "}")
    lines.append("\\toprule")

    header = ["", "\\textbf{Dataset}"]
    for label in metric_labels:
        header.append(f"\\textbf{{{label}}}")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    grouped = [(k, list(g)) for k, g in groupby(rows, key=lambda x: x[0])]
    for group_idx, (curve_type, group_rows) in enumerate(grouped):
        curve_label = curve_type.capitalize()
        for row_idx, (_, ds, values) in enumerate(group_rows):
            formatted = _format_row_with_ranking(values)
            if len(group_rows) > 1 and row_idx == 0:
                label = f"\\multirow{{{len(group_rows)}}}{{*}}{{{curve_label}}}"
            elif len(group_rows) == 1:
                label = curve_label
            else:
                label = ""
            line = [label, DATASET_DISPLAY[ds]] + formatted
            lines.append(" & ".join(line) + " \\\\")
        if group_idx < len(grouped) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2 & 4: Spearman correlation with training iterations
# ---------------------------------------------------------------------------


def generate_spearman_training_table(variant: str) -> str:
    """Generate Spearman correlation table of metrics with training steps.

    Values multiplied by 100. Bold = best metric per row, underline = second best.
    Validity uses raw Spearman (higher = better), PGD/MMD use sign-adjusted (negated).
    """
    # Metrics where higher values = better (no sign flip needed)
    higher_is_better = {"validity"}
    metric_cols = ["validity", "polyscore"] + MMD_KEYS
    metric_labels = (
        ["Val."] + ["PGD"] + [MMD_SHORT[k] + f" {MMD_SUFFIX}" for k in MMD_KEYS]
    )

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
        values = []

        for col in metric_cols:
            if col in df.columns:
                mask = np.isfinite(df[col].values.astype(float)) & np.isfinite(
                    steps
                )
                if mask.sum() < 3:
                    values.append(np.nan)
                else:
                    rho_val, _ = stats.spearmanr(
                        steps[mask], df[col].values.astype(float)[mask]
                    )
                    rho = float(rho_val)  # type: ignore[arg-type]
                    if col not in higher_is_better:
                        rho = -rho
                    values.append(rho * 100)
            else:
                values.append(np.nan)

        formatted = _format_row_with_ranking(values)
        line = [DATASET_DISPLAY[ds]] + formatted
        lines.append(" & ".join(line) + " \\\\")

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
    mmd_labels = [MMD_SHORT[k] + f" {MMD_SUFFIX}" for k in present_mmds]

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
            if val is None or np.isnan(float(val)):
                cells.append("-")
            else:
                cells.append(f"{val:.4f}")
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 6: Denoising iterations PGS values
# ---------------------------------------------------------------------------


def generate_denoising_pgs_table(variant: str = "jsd") -> str:
    """Generate table of PGS values per denoising step, with optional VUN column."""
    df = load_results("denoising", "planar", variant)
    if df is None:
        return ""

    df = df.sort_values("steps")

    has_vun = "vun" in df.columns and bool(df["vun"].notna().any())

    # VUN comes first (after Steps), then PGD and subscores — matching rebuttal column order
    score_cols = []
    score_labels = []
    if has_vun:
        score_cols.append("vun")
        score_labels.append("VUN")
    score_cols += ["polyscore"] + [
        k for k in PGS_SUBSCORE_KEYS if k in df.columns
    ]
    score_labels += ["PGD"] + [
        PGS_SHORT.get(k, k) for k in PGS_SUBSCORE_KEYS if k in df.columns
    ]

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
            if val is None or pd.isna(val):  # type: ignore[arg-type]
                cells.append("-")
            else:
                cells.append(f"{float(val) * 100:.2f}")
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


@app.command()
def main():
    """Generate all model quality LaTeX tables."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    table_map = {
        "digress-pearson-correlation-jsd.tex": lambda: (
            generate_pearson_correlation_table("jsd")
        ),
        "correlation-across-training-jsd.tex": lambda: (
            generate_spearman_training_table("jsd")
        ),
        "digress-pearson-correlation-informedness.tex": lambda: (
            generate_pearson_correlation_table("informedness")
        ),
        "correlation-across-training-informedness.tex": lambda: (
            generate_spearman_training_table("informedness")
        ),
        "digress-denoising-iters-mmd.tex": generate_denoising_mmd_table,
        "digress-denoising-iters-pgs-jsd.tex": lambda: (
            generate_denoising_pgs_table("jsd")
        ),
        "digress-denoising-iters-pgs-informedness.tex": lambda: (
            generate_denoising_pgs_table("informedness")
        ),
    }

    for fname, generator in table_map.items():
        content = generator()
        if content:
            out = TABLES_DIR / fname
            out.write_text(content)
            logger.success("Saved: {}", out)
        else:
            logger.warning("No data for {}", fname)

    logger.success("Done.")


if __name__ == "__main__":
    app()
