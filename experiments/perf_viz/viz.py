from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Data from your results
data_str = """    orbit_pgs  degree_pgs  spectral_pgs  clustering_pgs   gin_pgs  orbit5_pgs       pgs  epoch  validity
0    0.400352    0.131988      0.240288        0.128536  0.182191    0.410608  0.410608     29  0.476562
1    0.243630    0.089820      0.108705        0.072056  0.072987    0.251779  0.251779     89  0.648438
2    0.219449    0.048881      0.100923        0.047065  0.089690    0.223767  0.223767    149  0.843750
3    0.216334    0.135231      0.158541        0.066641  0.127204    0.224574  0.224574    209  0.804688
4    0.190673    0.118474      0.139397        0.027924  0.135273    0.209162  0.209162    269  0.851562
5    0.155308    0.070952      0.054515        0.047439  0.056027    0.164581  0.164581    329  0.843750
6    0.130822    0.070719      0.078966        0.048026  0.085749    0.139959  0.139959    389  0.945312
7    0.171321    0.136389      0.143469        0.000017  0.159024    0.169810  0.171321    449  0.929688
8    0.096745    0.052319      0.050594        0.037623  0.069971    0.104000  0.104000    509  0.937500
9    0.131539    0.121095      0.116686        0.025867  0.114064    0.126064  0.126064    569  0.945312
10   0.133893    0.022938      0.043554        0.000017  0.107844    0.118981  0.118981    629  0.953125
11   0.221982    0.235934      0.206099        0.070961  0.211161    0.229116  0.229116    689  0.960938
12   0.059359    0.053177      0.000000        0.000000  0.010137    0.064118  0.064118    749  0.976562
13   0.135614    0.096585      0.096669        0.016325  0.131010    0.131528  0.135614    809  0.976562
14   0.114707    0.084702      0.056347        0.024936  0.105541    0.122984  0.122984    869  0.945312
15   0.147980    0.123172      0.071013        0.000000  0.136269    0.157177  0.157177    929  0.984375"""

# Parse the data
df = pd.read_csv(StringIO(data_str), sep="\s+")
df = df.sort_values("epoch")  # Sort by epoch for better visualization

# Dynamically identify metric columns
pgs_cols = sorted([c for c in df.columns if c.endswith("_pgs") and c != "pgs"])
mmd_cols = sorted([c for c in df.columns if c.endswith("_mmd")])
has_pgs = "pgs" in df.columns
has_validity = "validity" in df.columns
has_epoch = "epoch" in df.columns


def plot_unavailable(ax, title=""):
    ax.text(
        0.5,
        0.5,
        "Data not available",
        ha="center",
        va="center",
        fontsize=12,
        color="grey",
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


# Set up the plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# Create a comprehensive set of plots
fig, axs = plt.subplots(3, 4, figsize=(20, 15))
axs = axs.ravel()

# 1. Validity vs Epoch
ax = axs[0]
if has_validity and has_epoch:
    ax.plot(df["epoch"], df["validity"], "o-", linewidth=2, markersize=6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validity")
    ax.set_title("Model Validity Over Training")
    ax.grid(True, alpha=0.3)
else:
    plot_unavailable(ax, "Model Validity Over Training")

# 2. PGS Metrics vs Epoch
ax = axs[1]
if pgs_cols and has_epoch:
    for col in pgs_cols:
        ax.plot(
            df["epoch"],
            df[col],
            "o-",
            label=col.replace("_pgs", ""),
            linewidth=2,
            markersize=4,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PGS Score")
    ax.set_title("PGS Metrics Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    plot_unavailable(ax, "PGS Metrics Over Training")

# 3. MMD Metrics vs Epoch
ax = axs[2]
if mmd_cols and has_epoch:
    for col in mmd_cols:
        ax.plot(
            df["epoch"],
            df[col],
            "o-",
            label=col.replace("_mmd", ""),
            linewidth=2,
            markersize=4,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MMD Score")
    ax.set_title("MMD Metrics Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    plot_unavailable(ax, "MMD Metrics Over Training")

# 4. Overall PGS vs Epoch
ax = axs[3]
if has_pgs and has_epoch:
    ax.plot(
        df["epoch"], df["pgs"], "o-", linewidth=2, markersize=6, color="red"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Overall PGS")
    ax.set_title("Overall PGS Score Over Training")
    ax.grid(True, alpha=0.3)
else:
    plot_unavailable(ax, "Overall PGS Score Over Training")

# 5. Correlation heatmap between PGS metrics
ax = axs[4]
corr_cols = pgs_cols + (["pgs"] if has_pgs else [])
if len(corr_cols) > 1:
    pgs_corr = df[corr_cols].corr()
    sns.heatmap(
        pgs_corr,
        annot=True,
        cmap="RdBu_r",
        center=0,
        square=True,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title("PGS Metrics Correlation")
else:
    plot_unavailable(ax, "PGS Metrics Correlation")

# 6. Correlation heatmap between MMD metrics
ax = axs[5]
if len(mmd_cols) > 1:
    mmd_corr = df[mmd_cols].corr()
    sns.heatmap(
        mmd_corr,
        annot=True,
        cmap="RdBu_r",
        center=0,
        square=True,
        fmt=".3f",
        ax=ax,
    )
    ax.set_title("MMD Metrics Correlation")
else:
    plot_unavailable(ax, "MMD Metrics Correlation")

# 7. Validity vs PGS scatter
ax = axs[6]
if has_pgs and has_validity and has_epoch:
    scatter = ax.scatter(
        df["pgs"],
        df["validity"],
        c=df["epoch"],
        cmap="viridis",
        s=60,
        alpha=0.7,
    )
    ax.set_xlabel("Overall PGS")
    ax.set_ylabel("Validity")
    ax.set_title("Validity vs PGS (colored by epoch)")
    plt.colorbar(scatter, ax=ax, label="Epoch")
else:
    plot_unavailable(ax, "Validity vs PGS")

# 8. Box plots of PGS metrics
ax = axs[7]
if pgs_cols:
    df[pgs_cols].boxplot(ax=ax)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("PGS Score")
    ax.set_title("Distribution of PGS Metrics")
else:
    plot_unavailable(ax, "Distribution of PGS Metrics")

# 9. Box plots of MMD metrics
ax = axs[8]
if mmd_cols:
    df[mmd_cols].boxplot(ax=ax)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("MMD Score")
    ax.set_title("Distribution of MMD Metrics")
else:
    plot_unavailable(ax, "Distribution of MMD Metrics")

# 10. Convergence analysis: Rolling average of validity
ax = axs[9]
if has_validity and has_epoch and len(df) >= 3:
    window_size = 3
    rolling_validity = (
        df["validity"].rolling(window=window_size, center=True).mean()
    )
    ax.plot(df["epoch"], df["validity"], "o-", alpha=0.5, label="Raw")
    ax.plot(
        df["epoch"],
        rolling_validity,
        "r-",
        linewidth=3,
        label=f"Rolling Avg (window={window_size})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validity")
    ax.set_title("Validity Convergence Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    plot_unavailable(ax, "Validity Convergence Analysis")

# 11. Performance evolution: Best metrics over time
ax = axs[10]
plotted = False
if has_epoch:
    if has_pgs:
        best_pgs = df["pgs"].cummin()
        ax.plot(df["epoch"], best_pgs, "b-", linewidth=2, label="Best PGS")
        plotted = True
    if has_validity:
        best_validity = df["validity"].cummax()
        ax.plot(
            df["epoch"], best_validity, "g-", linewidth=2, label="Best Validity"
        )
        plotted = True

if plotted:
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Best Performance Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    plot_unavailable(ax, "Best Performance Evolution")

# 12. Metric improvement rate
ax = axs[11]
metrics_for_improvement = []
if has_validity:
    metrics_for_improvement.append("validity")
metrics_for_improvement.extend(pgs_cols)
metrics_for_improvement.extend(mmd_cols)

if metrics_for_improvement and len(df) > 1:
    first_epoch = df.iloc[0]
    last_epoch = df.iloc[-1]

    improvements = []
    metric_names = []

    for metric in metrics_for_improvement:
        if metric in df.columns and first_epoch[metric] is not None:
            # Avoid division by zero
            if abs(first_epoch[metric]) > 1e-9:
                improvement = (
                    (last_epoch[metric] - first_epoch[metric])
                    / abs(first_epoch[metric])
                ) * 100
                improvements.append(improvement)
                metric_names.append(metric)

    if improvements:
        # Sort by metric name for consistent plotting
        sorted_metrics = sorted(zip(metric_names, improvements))
        metric_names, improvements = zip(*sorted_metrics)

        colors = ["green" if x > 0 else "red" for x in improvements]
        ax.barh(metric_names, improvements, color=colors, alpha=0.7)
        ax.set_xlabel("Improvement (%)")
        ax.set_title("Metric Improvement (First to Last Epoch)")
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    else:
        plot_unavailable(ax, "Metric Improvement Rate")
else:
    plot_unavailable(ax, "Metric Improvement Rate")

plt.tight_layout()
plt.savefig(
    "experiments/perf_viz/figures/benchmark_results_analysis.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# Summary statistics
print("=== BENCHMARK RESULTS SUMMARY ===")
if has_validity:
    print("\n1. VALIDITY STATISTICS:")
    print(f"   Initial Validity: {df['validity'].iloc[0]:.3f}")
    print(f"   Final Validity: {df['validity'].iloc[-1]:.3f}")
    print(f"   Max Validity: {df['validity'].max():.3f}")
    if df["validity"].iloc[0] != 0:
        print(
            f"   Validity Improvement: {((df['validity'].iloc[-1] - df['validity'].iloc[0]) / df['validity'].iloc[0] * 100):+.1f}%"
        )

if has_pgs:
    print("\n2. OVERALL PGS STATISTICS:")
    print(f"   Initial PGS: {df['pgs'].iloc[0]:.3f}")
    print(f"   Final PGS: {df['pgs'].iloc[-1]:.3f}")
    print(f"   Best PGS: {df['pgs'].min():.3f}")
    if abs(df["pgs"].iloc[0]) > 1e-9:
        print(
            f"   PGS Improvement: {((df['pgs'].iloc[-1] - df['pgs'].iloc[0]) / abs(df['pgs'].iloc[0]) * 100):+.1f}%"
        )

if (has_validity or has_pgs) and has_epoch:
    print("\n3. BEST PERFORMING EPOCHS:")
    if has_validity:
        print(
            f"   Highest Validity: Epoch {df.loc[df['validity'].idxmax(), 'epoch']} (validity: {df['validity'].max():.3f})"
        )
    if has_pgs:
        print(
            f"   Lowest PGS: Epoch {df.loc[df['pgs'].idxmin(), 'epoch']} (PGS: {df['pgs'].min():.3f})"
        )

if has_pgs and has_validity:
    print("\n4. METRIC CORRELATIONS:")
    print(
        "   PGS vs Validity correlation:",
        f"{df['pgs'].corr(df['validity']):.3f}",
    )

if has_validity:
    print("\n5. TOP 5 EPOCHS BY VALIDITY:")
    top_validity_cols = (
        ["epoch", "validity", "pgs"] if has_pgs else ["epoch", "validity"]
    )
    top_validity = df.nlargest(5, "validity")[top_validity_cols]
    print(top_validity.to_string(index=False))

if has_pgs:
    print("\n6. TOP 5 EPOCHS BY PGS (lowest is best):")
    top_pgs_cols = (
        ["epoch", "validity", "pgs"] if has_validity else ["epoch", "pgs"]
    )
    top_pgs = df.nsmallest(5, "pgs")[top_pgs_cols]
    print(top_pgs.to_string(index=False))
