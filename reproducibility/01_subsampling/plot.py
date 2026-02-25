#!/usr/bin/env python3
"""Generate all subsampling figures for the paper."""
import json
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator
from pyprojroot import here

app = typer.Typer()
REPO_ROOT = here()
MMD_RESULTS_DIR = REPO_ROOT / "reproducibility" / "figures" / "01_subsampling" / "results" / "compute_mmd"
PGD_RESULTS_DIR = REPO_ROOT / "reproducibility" / "figures" / "01_subsampling" / "results" / "compute_pgd"
OUTPUT_DIR = REPO_ROOT / "reproducibility" / "figures" / "01_subsampling"
STYLE_FILE = Path(__file__).resolve().parent.parent / "polygraph.mplstyle"
MODELS = ["AUTOGRAPH", "DIGRESS", "GRAN", "ESGG"]
VARIANTS = ["biased", "umve"]
DATASET_ORDER = ["Lobster", "Planar", "SBM"]
DESCRIPTOR_ORDER = ["Orbit", "Orbit 5", "Spec.", "GIN", "Deg.", "Clust."]
SOURCE_ORDER = ["Test Set", "AutoGraph", "DiGress", "GRAN", "ESGG"]
DESCRIPTOR_LABEL_MAP = {"orbit4": "Orbit", "orbit5": "Orbit 5", "spectral": "Spec.", "gin": "GIN", "degree": "Deg.", "clustering": "Clust."}
DATASET_LABEL_MAP = {"lobster": "Lobster", "planar": "Planar", "sbm": "SBM"}
MODEL_DISPLAY = {"AUTOGRAPH": "AutoGraph", "DIGRESS": "DiGress", "GRAN": "GRAN", "ESGG": "ESGG", "test": "Test Set"}


def _src_color(src):
    p = sns.color_palette("colorblind")
    return {"Test Set": p[0], "AutoGraph": p[1], "DiGress": p[2], "GRAN": p[3], "ESGG": p[4]}.get(src, p[0])


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
    df["Dataset"] = df["dataset"].map(DATASET_LABEL_MAP)
    df["Descriptor"] = df["descriptor"].map(DESCRIPTOR_LABEL_MAP)
    df["Source"] = df["model"].map(MODEL_DISPLAY)
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
            ax.fill_between(s["subsample_size"], s["mmd_low"], s["mmd_high"], alpha=0.2, color=_src_color(src))


def _logx(g):
    for ax in g.axes.flat:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(LogLocator(base=2, numticks=7))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}" if float(x).is_integer() else f"{x:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", which="major", labelrotation=45, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)


def _facet_titles(g):
    g.set_titles(row_template="", col_template="{col_name}", size=14)
    g.fig.subplots_adjust(wspace=0.3, hspace=0.1)
    for (rv, cv), ax in g.axes_dict.items():
        if cv == g.col_names[0]:
            ax.text(-0.5, 0.5, str(rv), transform=ax.transAxes, ha="right", va="center",
                    rotation=90, fontsize=14, fontweight="bold")


def _yminor(ax):
    if ax.get_yscale() == "log":
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="y", which="minor", length=2, width=0.6)


def plot_mmd_combined(df_all, model, variant, output_dir):
    mc = model.lower()
    df = df_all[((df_all["model"].str.lower() == mc) | (df_all["model"] == "test")) & (df_all["variant"] == variant)].copy()
    if df.empty:
        return
    srcs = [s for s in SOURCE_ORDER if s in df["Source"].unique()]
    cp = {s: _src_color(s) for s in srcs}
    mode = "log" if variant == "biased" else "linear"
    sfx = "logy" if mode == "log" else "lineary"
    descs = [d for d in DESCRIPTOR_ORDER if d in df["Descriptor"].unique()]
    dsets = [d for d in DATASET_ORDER if d in df["Dataset"].unique()]
    g = sns.relplot(data=df, x="subsample_size", y="mmd_mean", hue="Source", col="Descriptor",
                    row="Dataset", kind="line", marker="o", markersize=4, height=2.2, aspect=1.0,
                    col_order=descs, row_order=dsets, hue_order=srcs, palette=cp,
                    facet_kws={"sharey": False, "sharex": True}, legend=False)
    _add_bands(g, df)
    _logx(g)
    for ax in g.axes.flat:
        if mode == "log":
            ax.set_yscale("log")
        _yminor(ax)
    _facet_titles(g)
    ml = r" ($\log_{10}$)" if mode == "log" else ""
    g.set_ylabels(f"MMD{ml}", fontsize=14, labelpad=0)
    g.set_xlabels(r"Number of Graphs ($\log_2$)", fontsize=14)
    lh = [Line2D([0], [0], color=cp[s], marker="o", markersize=5, label=s) for s in srcs]
    g.fig.legend(handles=lh, title="Graph Sources", loc="lower center", ncol=len(srcs),
                 bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=12, title_fontsize=12)
    g.fig.subplots_adjust(bottom=0.12)
    out = output_dir / f"{mc}_{variant}_subsampling_{sfx}.pdf"
    g.fig.savefig(str(out), bbox_inches="tight")
    plt.close(g.fig)
    logger.success("Saved: {}", out)


def plot_mmd_individual(df_all, model, variant, dataset, descriptor, output_dir):
    mc = model.lower()
    df = df_all[((df_all["model"].str.lower() == mc) | (df_all["model"] == "test"))
                & (df_all["variant"] == variant) & (df_all["dataset"] == dataset)
                & (df_all["descriptor"] == descriptor)].copy()
    if df.empty:
        return
    mode = "log" if variant == "biased" else "linear"
    sfx = "logy" if mode == "log" else "lineary"
    plt.figure(figsize=(3.5, 3.5))
    srcs = [s for s in SOURCE_ORDER if s in df["Source"].unique()]
    for src in srcs:
        s = df[df["Source"] == src].sort_values("subsample_size")
        c = _src_color(src)
        plt.plot(s["subsample_size"], s["mmd_mean"], marker="o", markersize=4, color=c, label=src)
        plt.fill_between(s["subsample_size"], s["mmd_low"], s["mmd_high"], alpha=0.2, color=c)
    ax = plt.gca()
    ax.set_xscale("log", base=2)
    ax.set_yscale(mode)
    _yminor(ax)
    ax.xaxis.set_major_locator(LogLocator(base=2, numticks=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}" if float(x).is_integer() else f"{x:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="major", labelrotation=45, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    plt.xlabel(r"Number of Graphs ($\log_2$)", fontsize=14)
    ml = r" ($\log_{10}$)" if mode == "log" else ""
    ax.set_ylabel(f"MMD{ml}", fontsize=14, labelpad=0)
    ds_l = DATASET_LABEL_MAP.get(dataset, dataset)
    de_l = DESCRIPTOR_LABEL_MAP.get(descriptor, descriptor)
    plt.title(f"RBF MMD {de_l} on {ds_l}", fontsize=14)
    h, l = ax.get_legend_handles_labels()
    oi = {n: i for i, n in enumerate(SOURCE_ORDER)}
    sp = sorted(zip(h, l), key=lambda x: oi.get(x[1], 99))
    la = {"bbox_to_anchor": (0.02, 0.02), "loc": "lower left"} if mode == "log" else {"bbox_to_anchor": (0.98, 0.98), "loc": "upper right"}
    ax.legend(handles=[a for a, _ in sp], labels=[b for _, b in sp], title="Graph Sources",
              bbox_transform=ax.transAxes, frameon=False, fontsize=12, title_fontsize=12, **la)
    plt.tight_layout()
    fn = f"{mc}_{variant}_{ds_l.lower()}_{de_l.lower().replace(' ', '_')}_subsampling_{sfx}.pdf"
    plt.savefig(str(output_dir / fn), bbox_inches="tight")
    plt.close()
    logger.success("Saved: {}", output_dir / fn)


PGD_SCORE_LABELS: Dict[str, str] = {
    "pgd": "PGD", "clustering_pgs": "Clustering PGD", "degree_pgs": "Degree PGD",
    "gin_pgs": "GIN PGD", "orbit5_pgs": "Orbit-5 PGD", "orbit_pgs": "Orbit PGD",
    "spectral_pgs": "Spectral PGD",
}


def _reshape_pgd_long(df):
    df["Dataset"] = df["dataset"].map(DATASET_LABEL_MAP)
    sc = [c for c in df.columns if c.endswith("_mean") and not c.startswith("_")]
    iv = [c for c in ["dataset", "Dataset", "model", "subsample_size"] if c in df.columns]
    rows = []
    for _, row in df.iterrows():
        for col in sc:
            base = col.replace("_mean", "")
            rows.append({**{k: row[k] for k in iv}, "Score": PGD_SCORE_LABELS.get(base, base),
                         "score_mean": row[col], "score_std": row.get(f"{base}_std", 0.0)})
    return pd.DataFrame(rows)


def plot_pgd_model(df_long, model, output_dir):
    mc = model.lower()
    df = df_long[df_long["model"].str.lower() == mc].copy()
    if df.empty:
        return
    dsets = [d for d in DATASET_ORDER if d in df["Dataset"].unique()]
    so = [s for s in PGD_SCORE_LABELS.values() if s in df["Score"].unique()]
    pal = sns.color_palette("colorblind", n_colors=len(so))
    cm = {s: pal[i] for i, s in enumerate(so)}
    g = sns.relplot(data=df, x="subsample_size", y="score_mean", hue="Score", col="Dataset",
                    kind="line", marker="o", markersize=4, height=3.0, aspect=1.0,
                    col_order=dsets, hue_order=so, palette=cm,
                    facet_kws={"sharey": False, "sharex": True}, legend=True)
    for ds, ax in zip(dsets, g.axes.flat):
        sub = df[df["Dataset"] == ds]
        for sn in so:
            s = sub[sub["Score"] == sn].sort_values("subsample_size")
            if not s.empty:
                ax.fill_between(s["subsample_size"], s["score_mean"] - s["score_std"],
                                s["score_mean"] + s["score_std"], alpha=0.15, color=cm[sn])
    for ax in g.axes.flat:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(LogLocator(base=2, numticks=7))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}" if float(x).is_integer() else f"{x:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", which="major", labelrotation=45, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        _yminor(ax)
    g.set_ylabels("PGD Score", fontsize=14)
    g.set_xlabels(r"Number of Graphs ($\log_2$)", fontsize=14)
    g.set_titles("{col_name}", size=14)
    sns.move_legend(g, "lower center", ncol=min(4, len(so)), frameon=False, fontsize=12, title_fontsize=14)
    g.fig.subplots_adjust(bottom=0.12)
    out = output_dir / f"{mc}_pgs_subsampling.pdf"
    g.fig.savefig(str(out), bbox_inches="tight")
    plt.close(g.fig)
    logger.success("Saved: {}", out)


@app.command()
def main(
    mmd_only: bool = typer.Option(False, "--mmd-only"),
    pgd_only: bool = typer.Option(False, "--pgd-only"),
    paper_dir: Optional[Path] = typer.Option(None, "--paper-dir"),
    results_suffix: str = typer.Option("", "--results-suffix", help="Suffix for results dir and output files (e.g. _tabpfn_v6)"),
):
    setup_plotting()

    pgd_results_dir = REPO_ROOT / "reproducibility" / "figures" / "01_subsampling" / "results" / f"compute_pgd{results_suffix}"

    # When suffix is provided, generate into a temp dir then copy with suffixed names
    import tempfile, shutil
    use_tmp = bool(results_suffix)
    tmp_dir = Path(tempfile.mkdtemp()) if use_tmp else None
    output_dir = tmp_dir if use_tmp else OUTPUT_DIR

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
            plot_mmd_individual(df_mmd, "DIGRESS", "biased", "planar", "orbit5", output_dir)
            plot_mmd_individual(df_mmd, "DIGRESS", "umve", "planar", "orbit5", output_dir)
            plot_mmd_individual(df_mmd, "GRAN", "biased", "sbm", "degree", output_dir)
            plot_mmd_individual(df_mmd, "GRAN", "umve", "sbm", "degree", output_dir)
    if not mmd_only:
        logger.info("Loading PGD results from {} ...", pgd_results_dir)
        if not pgd_results_dir.exists():
            logger.error("No PGD results found in {}", pgd_results_dir)
        else:
            recs = [json.loads(f.read_text()) for f in pgd_results_dir.glob("*.json")]
            df_pgd = pd.DataFrame(recs) if recs else pd.DataFrame()
            if df_pgd.empty:
                logger.error("No PGD results found in {}", pgd_results_dir)
            else:
                logger.info("Loaded {} PGD result files", len(df_pgd))
                df_long = _reshape_pgd_long(df_pgd)
                if not df_long.empty:
                    for m in MODELS:
                        plot_pgd_model(df_long, m, output_dir)

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
