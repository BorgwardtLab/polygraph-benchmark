#!/usr/bin/env python3
"""Generate all subsampling figures for the paper."""

import json
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    AutoMinorLocator,
    FixedLocator,
    FuncFormatter,
    LogLocator,
    NullFormatter,
    ScalarFormatter,
)
from pyprojroot import here

app = typer.Typer()
REPO_ROOT = here()
MMD_RESULTS_DIR = (
    REPO_ROOT
    / "reproducibility"
    / "figures"
    / "01_subsampling"
    / "results"
    / "compute_mmd"
)
PGD_RESULTS_DIR = (
    REPO_ROOT
    / "reproducibility"
    / "figures"
    / "01_subsampling"
    / "results"
    / "compute_pgd"
)
OUTPUT_DIR = REPO_ROOT / "reproducibility" / "figures" / "01_subsampling"
STYLE_FILE = Path(__file__).resolve().parent.parent / "polygraph.mplstyle"
MODELS = ["AUTOGRAPH", "DIGRESS", "GRAN", "ESGG"]
VARIANTS = ["biased", "umve"]
DATASET_ORDER = ["Lobster", "Planar", "SBM"]
DESCRIPTOR_ORDER = ["Orbit", "Orbit 5", "Spec.", "GIN", "Deg.", "Clust."]
SOURCE_ORDER = ["Test Set", "AutoGraph", "DiGress", "GRAN", "ESGG"]
DESCRIPTOR_LABEL_MAP = {
    "orbit4": "Orbit",
    "orbit5": "Orbit 5",
    "spectral": "Spec.",
    "gin": "GIN",
    "degree": "Deg.",
    "clustering": "Clust.",
}
DATASET_LABEL_MAP = {"lobster": "Lobster", "planar": "Planar", "sbm": "SBM"}
MODEL_DISPLAY = {
    "AUTOGRAPH": "AutoGraph",
    "DIGRESS": "DiGress",
    "GRAN": "GRAN",
    "ESGG": "ESGG",
    "test": "Test Set",
}


def _src_color(src):
    p = sns.color_palette("colorblind")
    return {
        "Test Set": p[0],
        "AutoGraph": p[1],
        "DiGress": p[2],
        "GRAN": p[3],
        "ESGG": p[4],
    }.get(src, p[0])


def setup_plotting():
    if STYLE_FILE.exists():
        plt.style.use(str(STYLE_FILE))
    sns.set_style("ticks")
    sns.set_palette("colorblind")


def load_mmd_results():
    if not MMD_RESULTS_DIR.exists():
        return pd.DataFrame()
    recs = [json.loads(f.read_text()) for f in MMD_RESULTS_DIR.glob("*.json")]
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    df["Dataset"] = df["dataset"].map(DATASET_LABEL_MAP)  # type: ignore[arg-type]
    df["Descriptor"] = df["descriptor"].map(DESCRIPTOR_LABEL_MAP)  # type: ignore[arg-type]
    df["Source"] = df["model"].map(MODEL_DISPLAY)  # type: ignore[arg-type]
    return df


def load_pgd_results():
    if not PGD_RESULTS_DIR.exists():
        return pd.DataFrame()
    recs = [json.loads(f.read_text()) for f in PGD_RESULTS_DIR.glob("*.json")]
    return pd.DataFrame(recs) if recs else pd.DataFrame()


def _add_bands(g, df):
    for (rv, cv), ax in g.axes_dict.items():
        sub = df[(df["Dataset"] == rv) & (df["Descriptor"] == cv)]
        for src in sub["Source"].unique():
            s = sub[sub["Source"] == src].sort_values("subsample_size")
            ax.fill_between(
                s["subsample_size"],
                s["mmd_low"],
                s["mmd_high"],
                alpha=0.2,
                color=_src_color(src),
            )


def _logx(g):
    for ax in g.axes.flat:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(LogLocator(base=2, numticks=7))
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda x, _: f"{int(x)}" if float(x).is_integer() else f"{x:g}"
            )
        )
        ax.xaxis.set_minor_locator(
            FixedLocator(
                [
                    v * s
                    for v in (2**n for n in range(3, 13))
                    for s in (1.25, 1.5, 1.75)
                ]
            )
        )
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", length=2, width=0.5)
        ax.tick_params(axis="x", which="major", length=5)
        ax.tick_params(axis="x", which="major", labelrotation=45, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)


def _facet_titles(g):
    g.set_titles(row_template="", col_template="{col_name}", size=14)
    g.fig.subplots_adjust(wspace=0.3, hspace=0.25)
    for (rv, cv), ax in g.axes_dict.items():
        if cv == g.col_names[0]:
            ax.text(
                -0.5,
                0.5,
                str(rv),
                transform=ax.transAxes,
                ha="right",
                va="center",
                rotation=90,
                fontsize=14,
                fontweight="bold",
            )


def _yminor(ax):
    if ax.get_yscale() == "log":
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=(np.arange(2, 10) * 0.1).tolist())
        )
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        # Use scientific notation for small y-values to avoid label overlap
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis="y", which="minor", length=2, width=0.6)


def plot_mmd_combined(df_all, model, variant, output_dir):
    mc = model.lower()
    df = df_all[
        ((df_all["model"].str.lower() == mc) | (df_all["model"] == "test"))
        & (df_all["variant"] == variant)
    ].copy()
    if df.empty:
        return
    srcs = [s for s in SOURCE_ORDER if s in df["Source"].unique()]
    cp = {s: _src_color(s) for s in srcs}
    mode = "log" if variant == "biased" else "linear"
    sfx = "logy" if mode == "log" else "lineary"
    descs = [d for d in DESCRIPTOR_ORDER if d in df["Descriptor"].unique()]
    dsets = [d for d in DATASET_ORDER if d in df["Dataset"].unique()]
    g = sns.relplot(
        data=df,
        x="subsample_size",
        y="mmd_mean",
        hue="Source",
        col="Descriptor",
        row="Dataset",
        kind="line",
        marker="o",
        markersize=4,
        height=2,
        aspect=1.1,
        col_order=descs,
        row_order=dsets,
        hue_order=srcs,
        palette=cp,
        facet_kws={"margin_titles": True, "sharey": False},
    )
    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.5, -0.2),
        title="Graph sources:",
        ncol=2,
        title_fontsize=14,
        fontsize=14,
    )
    _add_bands(g, df)
    _logx(g)
    for ax in g.axes.flat:
        if mode == "log":
            ax.set_yscale("log")
        _yminor(ax)
    _facet_titles(g)
    g.set_axis_labels(
        r"Number of Graphs",
        "MMD",
        fontsize=14,
    )
    out = output_dir / f"{mc}_{variant}_subsampling_{sfx}.pdf"
    g.fig.savefig(str(out), bbox_inches="tight")
    plt.close(g.fig)
    logger.success("Saved: {}", out)


def plot_mmd_individual(
    df_all, model, variant, dataset, descriptor, output_dir
):
    mc = model.lower()
    df = df_all[
        ((df_all["model"].str.lower() == mc) | (df_all["model"] == "test"))
        & (df_all["variant"] == variant)
        & (df_all["dataset"] == dataset)
        & (df_all["descriptor"] == descriptor)
    ].copy()
    if df.empty:
        return
    mode = "log" if variant == "biased" else "linear"
    sfx = "logy" if mode == "log" else "lineary"
    plt.figure(figsize=(3.5, 3.5))
    srcs = [s for s in SOURCE_ORDER if s in df["Source"].unique()]
    for src in srcs:
        s = df[df["Source"] == src].sort_values("subsample_size")
        c = _src_color(src)
        plt.plot(
            s["subsample_size"],
            s["mmd_mean"],
            marker="o",
            markersize=4,
            color=c,
            label=src,
        )
        plt.fill_between(
            s["subsample_size"], s["mmd_low"], s["mmd_high"], alpha=0.2, color=c
        )
    ax = plt.gca()
    ax.set_xscale("log", base=2)
    ax.set_yscale(mode)
    _yminor(ax)
    ax.xaxis.set_major_locator(LogLocator(base=2, numticks=5))
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _: f"{int(x)}" if float(x).is_integer() else f"{x:g}"
        )
    )
    ax.xaxis.set_minor_locator(
        FixedLocator(
            [
                v * s
                for v in (2**n for n in range(3, 13))
                for s in (1.25, 1.5, 1.75)
            ]
        )
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=2, width=0.5)
    ax.tick_params(axis="x", which="major", length=5)
    ax.tick_params(axis="x", which="major", labelrotation=45, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    plt.xlabel("Number of Graphs", fontsize=14)
    ax.set_ylabel("MMD", fontsize=14, labelpad=0)
    ax.yaxis.set_label_coords(-0.2, 0.5)
    ds_l = DATASET_LABEL_MAP.get(dataset, dataset)
    de_l = DESCRIPTOR_LABEL_MAP.get(descriptor, descriptor)
    assert ds_l is not None
    assert de_l is not None
    plt.title(f"RBF MMD {de_l} on {ds_l}", fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    oi = {n: i for i, n in enumerate(SOURCE_ORDER)}
    sp = sorted(zip(handles, labels), key=lambda x: oi.get(x[1], 99))
    la = (
        {"bbox_to_anchor": (0.02, 0.02), "loc": "lower left"}
        if mode == "log"
        else {"bbox_to_anchor": (0.98, 0.98), "loc": "upper right"}
    )
    ax.legend(
        handles=[a for a, _ in sp],
        labels=[b for _, b in sp],
        title="Graph Sources",
        bbox_transform=ax.transAxes,
        frameon=False,
        fontsize=12,
        title_fontsize=12,
        **la,
    )
    plt.tight_layout()
    fn = f"{mc}_{variant}_{ds_l.lower()}_{de_l.lower().replace(' ', '_')}_subsampling_{sfx}.pdf"
    plt.savefig(str(output_dir / fn), bbox_inches="tight")
    plt.close()
    logger.success("Saved: {}", output_dir / fn)


# PGD score key -> display label, in canonical column order
PGD_SCORE_KEYS = [
    "pgd",
    "orbit4",
    "orbit5",
    "spectral",
    "gin",
    "degree",
    "clustering",
]
PGD_SCORE_LABELS: Dict[str, str] = {
    "pgd": "PGD",
    "orbit4": "Orbit PGD",
    "orbit5": "Orbit-5 PGD",
    "spectral": "Spectral PGD",
    "gin": "GIN PGD",
    "degree": "Degree PGD",
    "clustering": "Clustering PGD",
}
PGD_SCORE_ORDER = [PGD_SCORE_LABELS[k] for k in PGD_SCORE_KEYS]


def _reshape_pgd_long(df):
    """Reshape wide PGD DataFrame to long format with Score/mean/low/high columns."""
    df["Dataset"] = df["dataset"].map(DATASET_LABEL_MAP)
    rows = []
    for _, row in df.iterrows():
        base = {
            "dataset": row["dataset"],
            "Dataset": row["Dataset"],
            "model": row["model"],
            "subsample_size": row["subsample_size"],
        }
        for key in PGD_SCORE_KEYS:
            mean_col, std_col = f"{key}_mean", f"{key}_std"
            if mean_col not in df.columns:
                continue
            m = row[mean_col]
            s = row.get(std_col, 0.0) if std_col in df.columns else 0.0
            rows.append(
                {
                    **base,
                    "Score": PGD_SCORE_LABELS[key],
                    "score_key": key,
                    "mean": m,
                    "low": m - s,
                    "high": m + s,
                }
            )
    return pd.DataFrame(rows)


def _model_color(model_code):
    """Canonical model color matching the MMD plots."""
    p = sns.color_palette("colorblind")
    return {"autograph": p[1], "digress": p[2], "gran": p[3], "esgg": p[4]}.get(
        model_code.lower(), p[0]
    )


def _test_color():
    return sns.color_palette("colorblind")[0]


def plot_pgd_combined(df_long, model, output_dir, df_test=None):
    """Combined faceted PGD plot: rows=datasets, cols=scores. Matches original plot_subsampling_pgs.py."""
    mc = model.lower()
    model_display = MODEL_DISPLAY.get(model, model)
    df = df_long[df_long["model"].str.upper() == model].copy()
    if df.empty:
        return

    dsets = [d for d in DATASET_ORDER if d in df["Dataset"].unique()]
    scores = [s for s in PGD_SCORE_ORDER if s in df["Score"].unique()]

    color = _model_color(mc)

    g = sns.relplot(
        data=df,
        x="subsample_size",
        y="mean",
        col="Score",
        row="Dataset",
        kind="line",
        marker="o",
        markersize=4,
        height=2,
        aspect=1.0,
        facet_kws={"margin_titles": True, "sharey": True},
        col_order=scores,
        row_order=dsets,
        errorbar=None,
        color=color,
    )

    # Error bands for model
    for (rv, cv), ax in g.axes_dict.items():
        sub = df[(df["Dataset"] == rv) & (df["Score"] == cv)].sort_values(
            "subsample_size"
        )
        if not sub.empty:
            ax.fill_between(
                sub["subsample_size"],
                sub["low"],
                sub["high"],
                alpha=0.25,
                color=color,
            )

    # Test baseline overlay
    if df_test is not None and not df_test.empty:
        for (rv, cv), ax in g.axes_dict.items():
            tsub = df_test[
                (df_test["Dataset"] == rv) & (df_test["Score"] == cv)
            ].sort_values("subsample_size")
            if not tsub.empty:
                ax.plot(
                    tsub["subsample_size"],
                    tsub["mean"],
                    linestyle="--",
                    linewidth=1.2,
                    color=_test_color(),
                    zorder=3,
                )
                ax.fill_between(
                    tsub["subsample_size"],
                    tsub["low"],
                    tsub["high"],
                    alpha=0.15,
                    color=_test_color(),
                    zorder=2,
                )

    # Axes formatting
    for ax in g.axes.flat:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(LogLocator(base=2, numticks=5))
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda x, _: f"{int(x)}" if float(x).is_integer() else f"{x:g}"
            )
        )
        ax.xaxis.set_minor_locator(
            FixedLocator(
                [
                    v * s
                    for v in (2**n for n in range(3, 13))
                    for s in (1.25, 1.5, 1.75)
                ]
            )
        )
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", length=2, width=0.5)
        ax.tick_params(axis="x", which="major", length=5)
        ax.tick_params(axis="x", which="major", labelrotation=45, labelsize=14)
        ax.tick_params(axis="y", labelsize=14)

    g.set(ylim=(0, 1))

    _facet_titles(g)
    g.set_axis_labels(
        "Number of Graphs",
        "PGD",
        fontsize=14,
    )

    # Legend
    lh = [
        Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label=model_display,
        )
    ]
    if df_test is not None and not df_test.empty:
        lh.append(
            Line2D(
                [0],
                [0],
                color=_test_color(),
                linestyle="--",
                linewidth=1.2,
                label="Test baseline",
            )
        )
    g.fig.legend(
        handles=lh,
        title="Graph Sources:",
        loc="lower center",
        ncol=len(lh),
        frameon=False,
        bbox_to_anchor=(0.5, -0.2),
        borderaxespad=0.5,
        title_fontsize=14,
        fontsize=14,
    )
    g.fig.subplots_adjust(bottom=0.12)

    out = output_dir / f"{mc}_pgd_subsampling.pdf"
    g.fig.savefig(str(out), bbox_inches="tight")
    plt.close(g.fig)
    logger.success("Saved: {}", out)


def plot_pgd_individual(
    df_long, model, dataset, score_key, output_dir, df_test=None
):
    """Individual PGD plot for a single (model, dataset, score) combo."""
    mc = model.lower()
    model_display = MODEL_DISPLAY.get(model, model)
    ds_label = DATASET_LABEL_MAP.get(dataset, dataset)
    score_label = PGD_SCORE_LABELS.get(score_key, score_key)
    assert ds_label is not None
    assert score_label is not None

    df = df_long[
        (df_long["model"].str.upper() == model)
        & (df_long["Dataset"] == ds_label)
        & (df_long["score_key"] == score_key)
    ].copy()
    if df.empty:
        return

    color = _model_color(mc)
    plt.figure(figsize=(3.5, 3.5))
    s = df.sort_values("subsample_size")
    plt.plot(
        s["subsample_size"],
        s["mean"],
        marker="o",
        markersize=4,
        color=color,
        label=model_display,
    )
    plt.fill_between(
        s["subsample_size"], s["low"], s["high"], alpha=0.25, color=color
    )

    # Test baseline overlay
    if df_test is not None and not df_test.empty:
        tsub = df_test[
            (df_test["Dataset"] == ds_label)
            & (df_test["score_key"] == score_key)
        ].sort_values("subsample_size")
        if not tsub.empty:
            plt.plot(
                tsub["subsample_size"],
                tsub["mean"],
                linestyle="--",
                linewidth=1.2,
                color=_test_color(),
                label="Test baseline",
            )
            plt.fill_between(
                tsub["subsample_size"],
                tsub["low"],
                tsub["high"],
                alpha=0.15,
                color=_test_color(),
            )

    ax = plt.gca()
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(LogLocator(base=2, numticks=5))
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _: f"{int(x)}" if float(x).is_integer() else f"{x:g}"
        )
    )
    ax.xaxis.set_minor_locator(
        FixedLocator(
            [
                v * s
                for v in (2**n for n in range(3, 13))
                for s in (1.25, 1.5, 1.75)
            ]
        )
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=2, width=0.5)
    ax.tick_params(axis="x", which="major", length=5)
    ax.tick_params(axis="x", which="major", labelrotation=45, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    plt.xlabel("Number of Graphs", fontsize=14)
    ax.set_ylabel("PGD", fontsize=14, labelpad=0)
    ax.yaxis.set_label_coords(-0.2, 0.5)
    plt.title(f"{score_label} on {ds_label}", fontsize=14)
    ax.legend(
        title="Graph Sources",
        frameon=False,
        fontsize=12,
        title_fontsize=12,
        bbox_to_anchor=(0.98, 0.98),
        loc="upper right",
        bbox_transform=ax.transAxes,
    )
    plt.tight_layout()
    fn = f"{mc}_pgd_{ds_label.lower()}_{score_key}_subsampling.pdf"
    plt.savefig(str(output_dir / fn), bbox_inches="tight")
    plt.close()
    logger.success("Saved: {}", output_dir / fn)


@app.command()
def main(
    mmd_only: bool = typer.Option(False, "--mmd-only"),
    pgd_only: bool = typer.Option(False, "--pgd-only"),
    results_suffix: str = typer.Option(
        "",
        "--results-suffix",
        help="Suffix for results dir and output files (e.g. _tabpfn_v6)",
    ),
):
    setup_plotting()

    pgd_results_dir = (
        REPO_ROOT
        / "reproducibility"
        / "figures"
        / "01_subsampling"
        / "results"
        / f"compute_pgd{results_suffix}"
    )

    # When suffix is provided, generate into a temp dir then copy with suffixed names
    import tempfile
    import shutil

    use_tmp = bool(results_suffix)
    tmp_dir = Path(tempfile.mkdtemp()) if use_tmp else None
    output_dir: Path = tmp_dir if tmp_dir is not None else OUTPUT_DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pgd_only:
        logger.info("Loading MMD results...")
        df_mmd = load_mmd_results()
        if df_mmd.empty:
            logger.error("No MMD results found in {}", MMD_RESULTS_DIR)
        else:
            logger.info("Loaded {} MMD result files", len(df_mmd))
            for m in MODELS:
                for v in VARIANTS:
                    plot_mmd_combined(df_mmd, m, v, output_dir)
            plot_mmd_individual(
                df_mmd, "DIGRESS", "biased", "planar", "orbit5", output_dir
            )
            plot_mmd_individual(
                df_mmd, "DIGRESS", "umve", "planar", "orbit5", output_dir
            )
            plot_mmd_individual(
                df_mmd, "GRAN", "biased", "sbm", "degree", output_dir
            )
            plot_mmd_individual(
                df_mmd, "GRAN", "umve", "sbm", "degree", output_dir
            )
    if not mmd_only:
        logger.info("Loading PGD results from {} ...", pgd_results_dir)
        if not pgd_results_dir.exists():
            logger.error("No PGD results found in {}", pgd_results_dir)
        else:
            recs = [
                json.loads(f.read_text())
                for f in pgd_results_dir.glob("*.json")
            ]
            df_pgd = pd.DataFrame(recs) if recs else pd.DataFrame()
            if df_pgd.empty:
                logger.error("No PGD results found in {}", pgd_results_dir)
            else:
                logger.info("Loaded {} PGD result files", len(df_pgd))
                df_long = _reshape_pgd_long(df_pgd)
                # Split out test baseline if present
                df_test = (
                    df_long[df_long["model"].str.lower() == "test"]
                    if "model" in df_long.columns
                    else pd.DataFrame()
                )
                if not df_long.empty:
                    test_arg = df_test if not df_test.empty else None
                    for m in MODELS:
                        plot_pgd_combined(
                            df_long, m, output_dir, df_test=test_arg
                        )
                        # Individual per-(model, dataset, score) plots
                        for ds in ["lobster", "planar", "sbm"]:
                            for sk in PGD_SCORE_KEYS:
                                plot_pgd_individual(
                                    df_long,
                                    m,
                                    ds,
                                    sk,
                                    output_dir,
                                    df_test=test_arg,
                                )

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
