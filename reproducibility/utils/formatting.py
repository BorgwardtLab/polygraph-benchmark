"""Shared LaTeX formatting utilities for reproducibility tables.

Provides consistent formatting helpers used across format.py scripts
in experiments 05--08.

Usage (from any reproducibility script)::

    from utils.formatting import fmt_pgs, fmt_sci, best_two
"""

from typing import Dict, Optional, Tuple

import pandas as pd


def fmt_pgs(
    mean: float,
    std: float,
    is_best: bool = False,
    is_second: bool = False,
) -> str:
    """Format a PGD score (×100) as ``X.X ± Y.Y`` with optional bold/underline."""
    if pd.isna(mean):
        return "-"
    text = f"{mean * 100:.1f} $\\pm\\,\\scriptstyle{{{std * 100:.1f}}}$"
    if is_best:
        return f"\\textbf{{{text}}}"
    if is_second:
        return f"\\underline{{{text}}}"
    return text


def fmt_sci(
    mean: float,
    std: float,
    is_best: bool = False,
    is_second: bool = False,
) -> str:
    """Format a value in scientific notation as ``X.XXXe±Y ± Z.ZZZe±W``."""
    if pd.isna(mean):
        return "-"
    text = f"{mean:.3e} $\\pm\\,\\scriptstyle{{{std:.3e}}}$"
    if is_best:
        return f"\\textbf{{{text}}}"
    if is_second:
        return f"\\underline{{{text}}}"
    return text


def best_two(
    results: Dict[str, Dict],
    key: str,
    lower: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """Find the best and second-best model for *key* across *results*.

    Returns ``(best_model, second_best_model)``; either may be ``None``.
    When *lower* is True the smallest value wins (e.g. PGD); otherwise
    the largest value wins (e.g. VUN).
    """
    vals = {
        m: r[key]
        for m, r in results.items()
        if key in r and not pd.isna(r.get(key))
    }
    if not vals:
        return None, None
    s = sorted(vals, key=lambda m: vals[m], reverse=not lower)
    return (
        s[0] if len(s) > 0 else None,
        s[1] if len(s) > 1 else None,
    )


# ── Display-name constants shared across tables ──────────────────────

MODELS = ["AUTOGRAPH", "DIGRESS", "GRAN", "ESGG"]
DATASETS = ["planar", "lobster", "sbm", "proteins"]

DATASET_DISPLAY = {
    "planar": "\\textsc{Planar-L}",
    "lobster": "\\textsc{Lobster-L}",
    "sbm": "\\textsc{SBM-L}",
    "proteins": "Proteins",
}

MODEL_DISPLAY = {
    "AUTOGRAPH": "AutoGraph",
    "DIGRESS": "\\textsc{DiGress}",
    "GRAN": "GRAN",
    "ESGG": "ESGG",
}
