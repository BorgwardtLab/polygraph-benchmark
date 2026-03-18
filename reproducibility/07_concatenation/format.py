#!/usr/bin/env python3
"""Format concatenation ablation results into LaTeX table.

Produces: tables/concatenation.tex (matching paper format)

Usage:
    python format.py
"""

import json
import sys
from pathlib import Path
from typing import Dict

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
RESULTS_DIR = OUTPUT_DIR / "results" / "concatenation"
BENCHMARK_DIR = OUTPUT_DIR / "results" / "benchmark"


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


_fmt = fmt_pgs
_best_two = best_two


def generate_table(concat_results: Dict, bench_results: Dict) -> str:
    lines = []
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Dataset} & \\textbf{Model} & \\textbf{VUN ($\\uparrow$)} & \\textbf{PGD ($\\downarrow$)} & \\textbf{PGD-Concat. ($\\downarrow$)} \\\\"
    )
    lines.append("\\midrule")

    for i, ds in enumerate(DATASETS):
        ds_concat = concat_results.get(ds, {})
        ds_bench = bench_results.get(ds, {})

        std_best, std_second = _best_two(
            ds_concat, "pgs_standard_mean", lower=True
        )
        cat_best, cat_second = _best_two(
            ds_concat, "pgs_concatenated_mean", lower=True
        )
        vun_best, vun_second = _best_two(ds_bench, "vun", lower=False)

        first = True
        for model in MODELS:
            cr = ds_concat.get(model, {})
            br = ds_bench.get(model, {})

            row = []
            row.append(DATASET_DISPLAY[ds] if first else "")
            first = False
            row.append(MODEL_DISPLAY.get(model, model))

            vun = br.get("vun")
            if vun is not None and not pd.isna(vun):
                vt = f"{float(vun) * 100:.1f}"
                if model == vun_best:
                    vt = f"\\textbf{{{vt}}}"
                elif model == vun_second:
                    vt = f"\\underline{{{vt}}}"
                row.append(vt)
            else:
                row.append("-")

            row.append(
                _fmt(
                    cr.get("pgs_standard_mean", float("nan")),
                    cr.get("pgs_standard_std", float("nan")),
                    model == std_best,
                    model == std_second,
                )
            )
            row.append(
                _fmt(
                    cr.get("pgs_concatenated_mean", float("nan")),
                    cr.get("pgs_concatenated_std", float("nan")),
                    model == cat_best,
                    model == cat_second,
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
    results_dir = OUTPUT_DIR / "results" / f"concatenation{results_suffix}"
    benchmark_dir = OUTPUT_DIR / "results" / f"benchmark{results_suffix}"
    concat_results = load_results(results_dir)
    bench_results = load_results(benchmark_dir)

    if not concat_results:
        logger.error("No concatenation results in {}.", results_dir)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = generate_table(concat_results, bench_results)
    out = OUTPUT_DIR / f"concatenation{results_suffix}.tex"
    out.write_text(table)
    logger.success("Saved: {}", out)


if __name__ == "__main__":
    app()
