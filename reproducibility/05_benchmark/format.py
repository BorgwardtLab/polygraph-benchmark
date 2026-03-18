#!/usr/bin/env python3
"""Format benchmark results from JSON into LaTeX tables.

Produces: tables/benchmark_results.tex (matching paper format exactly)

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
    fmt_pgs,
)

app = typer.Typer()

REPO_ROOT = here()
OUTPUT_DIR = REPO_ROOT / "reproducibility" / "tables"
RESULTS_DIR = OUTPUT_DIR / "results" / "benchmark"

SUBSCORES = [
    ("clustering_pgs", "Clust."),
    ("degree_pgs", "Deg."),
    ("gin_pgs", "GIN"),
    ("orbit5_pgs", "Orb5."),
    ("orbit4_pgs", "Orb4."),
    ("spectral_pgs", "Eig."),
]


def load_results(results_dir: Path) -> List[Dict]:
    results = []
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def _reshape(result_list: List[Dict]) -> Dict[str, Dict]:
    all_results: Dict[str, Dict] = {}
    for r in result_list:
        r = r.copy()
        ds = r.pop("dataset", None)
        model = r.pop("model", None)
        r.pop("error", None)
        if ds and model:
            all_results.setdefault(ds, {})[model] = r
    return all_results


_fmt_pgs = fmt_pgs
_best_two = best_two


def generate_benchmark_table(all_results: Dict) -> str:
    lines = []
    lines.append(
        "\\begin{tabular}{llccp{1.6cm}p{1.6cm}p{1.6cm}p{1.6cm}p{1.6cm}p{1.6cm}}"
    )
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Dataset} & \\textbf{Model} &  &  & \\multicolumn{6}{c}{\\textbf{PGD subscores}} \\\\"
    )
    lines.append("\\cmidrule(lr){5-10}")
    lines.append(
        " &  & \\textbf{VUN ($\\uparrow$)} & \\textbf{PGD ($\\downarrow$)} & "
        + " & ".join(
            f"\\textbf{{{label} ($\\downarrow$)}}" for _, label in SUBSCORES
        )
        + " \\\\"
    )
    lines.append("\\midrule")

    for i, dataset in enumerate(DATASETS):
        ds_results = all_results.get(dataset, {})
        pgs_best, pgs_second = _best_two(ds_results, "pgs_mean", lower=True)
        vun_best, vun_second = _best_two(ds_results, "vun", lower=False)
        sub_best = {}
        sub_second = {}
        for key, _ in SUBSCORES:
            mean_key = f"{key}_mean"
            alt_key = f"{key.replace('_pgs', '')}_mean"
            lookup = (
                mean_key
                if any(mean_key in ds_results.get(m, {}) for m in ds_results)
                else alt_key
            )
            sub_best[key], sub_second[key] = _best_two(
                ds_results, lookup, lower=True
            )

        first = True
        for model in MODELS:
            if model not in ds_results:
                continue
            r = ds_results[model]
            row = []
            row.append(DATASET_DISPLAY[dataset] if first else "")
            first = False
            row.append(MODEL_DISPLAY.get(model, model))

            if "vun" in r and not pd.isna(r.get("vun")):
                vun_text = (
                    f"{r['vun'] * 100:.1f}"
                    if isinstance(r["vun"], float)
                    else f"{float(r['vun']) * 100:.1f}"
                )
                if model == vun_best:
                    vun_text = f"\\textbf{{{vun_text}}}"
                elif model == vun_second:
                    vun_text = f"\\underline{{{vun_text}}}"
                row.append(vun_text)
            else:
                row.append("-")

            row.append(
                _fmt_pgs(
                    r.get("pgs_mean", float("nan")),
                    r.get("pgs_std", float("nan")),
                    model == pgs_best,
                    model == pgs_second,
                )
            )

            for key, _ in SUBSCORES:
                row.append(
                    _fmt_pgs(
                        r.get(
                            f"{key}_mean",
                            r.get(
                                f"{key.replace('_pgs', '')}_mean", float("nan")
                            ),
                        ),
                        r.get(
                            f"{key}_std",
                            r.get(
                                f"{key.replace('_pgs', '')}_std", float("nan")
                            ),
                        ),
                        model == sub_best[key],
                        model == sub_second[key],
                    )
                )

            lines.append(" & ".join(row) + " \\\\")

        if i < len(DATASETS) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


@app.command()
def main(
    results_suffix: str = typer.Option(
        "",
        "--results-suffix",
        help="Suffix for results dir and output files (e.g. _tabpfn_v6)",
    ),
):
    """Generate LaTeX tables from pre-computed JSON results."""
    results_dir = OUTPUT_DIR / "results" / f"benchmark{results_suffix}"
    result_list = load_results(results_dir)
    if not result_list:
        logger.error(
            "No results found in {}. Run compute.py first.", results_dir
        )
        return

    all_results = _reshape(result_list)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table = generate_benchmark_table(all_results)
    out = OUTPUT_DIR / f"benchmark_results{results_suffix}.tex"
    out.write_text(table)
    logger.success("Table saved to: {}", out)


if __name__ == "__main__":
    app()
