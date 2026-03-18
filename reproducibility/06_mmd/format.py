#!/usr/bin/env python3
"""Format MMD results from JSON into LaTeX tables.

Produces:
  tables/mmd_gtv.tex       (GTV MMD with VUN & PGD columns)
  tables/mmd_rbf_biased.tex
  tables/mmd_rbf_umve.tex

Usage:
    python format.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import typer
from loguru import logger
from pyprojroot import here

sys.path.insert(0, str(here() / "reproducibility"))
from utils.formatting import (
    DATASETS,
    DATASET_DISPLAY,
    MODEL_DISPLAY,
    MODELS,
    best_two,
    fmt_sci,
)

app = typer.Typer()

REPO_ROOT = here()
OUTPUT_DIR = REPO_ROOT / "reproducibility" / "tables"
MMD_RESULTS_DIR = OUTPUT_DIR / "results" / "mmd"
BENCHMARK_RESULTS_DIR = OUTPUT_DIR / "results" / "benchmark"


def load_results(results_dir: Path) -> Dict[str, Dict]:
    all_r: Dict[str, Dict] = {}
    if not results_dir.exists():
        return all_r
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            r = json.load(fh)
        ds, model = r.get("dataset"), r.get("model")
        if ds and model:
            all_r.setdefault(ds, {})[model] = r
    return all_r


_fmt_sci = fmt_sci
_best_two = best_two


def _fmt_pgs(mean: float, std: float, is_best=False, is_second=False) -> str:
    """MMD tables use 3 decimal places for PGD scores (not 1 like benchmark)."""
    if pd.isna(mean):
        return "-"
    text = f"{mean * 100:.3f} $\\pm\\,\\scriptstyle{{{std * 100:.3f}}}$"
    if is_best:
        return f"\\textbf{{{text}}}"
    if is_second:
        return f"\\underline{{{text}}}"
    return text


def generate_gtv_table(mmd_results: Dict, bench_results: Dict) -> str:
    """GTV table matches paper: includes VUN, PGD, and 4 GTV MMD columns."""
    metrics = ["gtv_degree", "gtv_clustering", "gtv_orbit", "gtv_spectral"]
    metric_labels = [
        "GTV MMD$^2$ Deg.",
        "GTV MMD$^2$ Clust.",
        "GTV MMD$^2$ Orb.",
        "GTV MMD$^2$ Eig.",
    ]

    lines = []
    lines.append("\\scalebox{0.6}{")
    lines.append("\\begin{tabular}{ll" + "c" * (2 + len(metrics)) + "}")
    lines.append("\\toprule")
    header = [
        "\\textbf{Dataset}",
        "\\textbf{Model}",
        "\\textbf{VUN ($\\uparrow$)}",
        "\\textbf{PGD ($\\downarrow$)}",
    ] + [f"\\textbf{{{lbl} ($\\downarrow$)}}" for lbl in metric_labels]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for i, ds in enumerate(DATASETS):
        ds_mmd = mmd_results.get(ds, {})
        ds_bench = bench_results.get(ds, {})
        pgs_best, pgs_second = _best_two(ds_bench, "pgs_mean", lower=True)
        vun_best, vun_second = _best_two(ds_bench, "vun", lower=False)

        first = True
        for model in MODELS:
            mmd_r = ds_mmd.get(model, {})
            bench_r = ds_bench.get(model, {})

            row = []
            row.append(DATASET_DISPLAY[ds] if first else "")
            first = False
            row.append(MODEL_DISPLAY.get(model, model))

            vun = bench_r.get("vun")
            if vun is not None and not pd.isna(vun):
                vt = f"{float(vun):.3f}"
                if model == vun_best:
                    vt = f"\\underline{{{vt}}}"
                elif model == vun_second:
                    pass
                row.append(vt)
            else:
                row.append("-")

            pgs_m = bench_r.get("pgs_mean", float("nan"))
            pgs_s = bench_r.get("pgs_std", float("nan"))
            row.append(
                _fmt_pgs(pgs_m, pgs_s, model == pgs_best, model == pgs_second)
            )

            for m_key in metrics:
                best, second = _best_two(ds_mmd, f"{m_key}_mean", lower=True)
                row.append(
                    _fmt_sci(
                        mmd_r.get(f"{m_key}_mean", float("nan")),
                        mmd_r.get(f"{m_key}_std", float("nan")),
                        model == best,
                        model == second,
                    )
                )

            lines.append(" & ".join(row) + " \\\\")
        if i < len(DATASETS) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    return "\n".join(lines)


def _generate_mmd_table(
    mmd_results: Dict, metrics: List[str], metric_labels: List[str]
) -> str:
    """Generic MMD table (RBF biased or UMVE)."""
    lines = []
    lines.append("\\scalebox{0.6}{")
    lines.append("\\begin{tabular}{ll" + "c" * len(metrics) + "}")
    lines.append("\\toprule")
    header = ["\\textbf{Dataset}", "\\textbf{Model}"] + [
        f"\\textbf{{{lbl} ($\\downarrow$)}}" for lbl in metric_labels
    ]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for i, ds in enumerate(DATASETS):
        ds_mmd = mmd_results.get(ds, {})
        first = True
        for model in MODELS:
            r = ds_mmd.get(model, {})
            row = []
            row.append(DATASET_DISPLAY[ds] if first else "")
            first = False
            row.append(MODEL_DISPLAY.get(model, model))

            for m_key in metrics:
                best, second = _best_two(ds_mmd, f"{m_key}_mean", lower=True)
                row.append(
                    _fmt_sci(
                        r.get(f"{m_key}_mean", float("nan")),
                        r.get(f"{m_key}_std", float("nan")),
                        model == best,
                        model == second,
                    )
                )
            lines.append(" & ".join(row) + " \\\\")
        if i < len(DATASETS) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    return "\n".join(lines)


@app.command()
def main():
    """Generate LaTeX tables from pre-computed JSON results."""
    mmd_results = load_results(MMD_RESULTS_DIR)
    bench_results = load_results(BENCHMARK_RESULTS_DIR)

    if not mmd_results:
        logger.error("No MMD results found in {}.", MMD_RESULTS_DIR)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gtv = generate_gtv_table(mmd_results, bench_results)
    (OUTPUT_DIR / "mmd_gtv.tex").write_text(gtv)
    logger.success("Saved: {}", OUTPUT_DIR / "mmd_gtv.tex")

    rbf_biased = _generate_mmd_table(
        mmd_results,
        ["rbf_degree", "rbf_clustering", "rbf_orbit", "rbf_spectral"],
        ["RBF Deg.", "RBF Clust.", "RBF Orb.", "RBF Eig."],
    )
    (OUTPUT_DIR / "mmd_rbf_biased.tex").write_text(rbf_biased)
    logger.success("Saved: {}", OUTPUT_DIR / "mmd_rbf_biased.tex")

    umve = _generate_mmd_table(
        mmd_results,
        ["umve_degree", "umve_clustering", "umve_orbit", "umve_spectral"],
        ["UMVE Deg.", "UMVE Clust.", "UMVE Orb.", "UMVE Eig."],
    )
    (OUTPUT_DIR / "mmd_rbf_umve.tex").write_text(umve)
    logger.success("Saved: {}", OUTPUT_DIR / "mmd_rbf_umve.tex")


if __name__ == "__main__":
    app()
