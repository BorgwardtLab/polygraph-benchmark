import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger
from pyprojroot import here

METRIC_HIGHER_IS_BETTER: Dict[str, bool] = {
    "VUN": True,
}

METRIC_ORDER: List[str] = [
    "VUN",
    "MMD_DEGREE",
    "MMD_CLUSTERING",
    "MMD_ORBIT",
    "MMD_SPECTRE",
]

DATASET_DISPLAY_NAMES: Dict[str, str] = {
    "PLANAR": "Planar",
    "LOBSTER": "Lobster",
    "SBM": "SBM",
}

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "DIGRESS": "\\textsc{DiGress}",
    "GRAN": "GRAN",
}

METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "VUN": "VUN",
    "MMD_DEGREE": "MMD Deg.",
    "MMD_CLUSTERING": "MMD Clust.",
    "MMD_ORBIT": "MMD Orb.",
    "MMD_SPECTRE": "MMD Eig.",
}


def get_display_name(
    original_name: str, display_mapping: Dict[str, str]
) -> str:
    """Get display name from mapping, fallback to original if not found."""
    result = display_mapping.get(original_name, original_name)
    logger.debug(f"Display mapping: '{original_name}' -> '{result}'")
    return result


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """Loads and preprocesses the benchmark results from a CSV file."""
    df = pd.read_csv(file_path)

    logger.debug("Unique datasets: %s", sorted(df["dataset"].unique()))
    logger.debug("Unique models: %s", sorted(df["model"].unique()))
    logger.debug("Unique metrics: %s", sorted(df["metric"].unique()))

    cols_to_keep = [
        "model",
        "dataset",
        "metric",
        "valid_unique_novel_low",
        "valid_unique_novel_high",
        "valid_unique_novel_mle",
        "low",
        "high",
        "mean",
        "std",
    ]
    df = df[cols_to_keep]

    df["valid_unique_novel_low"] = df["valid_unique_novel_low"].replace(
        -1.0, pd.NA
    )
    df["valid_unique_novel_high"] = df["valid_unique_novel_high"].replace(
        -1.0, pd.NA
    )

    df["low"] = df["valid_unique_novel_low"].combine_first(df["low"])
    df["high"] = df["valid_unique_novel_high"].combine_first(df["high"])
    df["mean"] = df["valid_unique_novel_mle"].combine_first(df["mean"])

    df = df.drop(
        columns=[
            "valid_unique_novel_low",
            "valid_unique_novel_high",
            "valid_unique_novel_mle",
            "std",
        ]
    )
    df = df.groupby(["model", "dataset", "metric"]).mean().reset_index()
    return df


def format_number(n: float) -> str:
    """Formats a number to 3 decimal places or scientific notation."""
    if pd.isna(n):
        return ""
    if n == 0 or (0.001 <= abs(n) < 1000):
        return f"{n:.3f}"
    else:
        return f"{n:.2e}"


def get_best_and_second_best_models(
    metric_means: pd.Series, higher_is_better: bool
) -> Tuple[Optional[str], Optional[str]]:
    """Sorts models by performance and returns the best and second-best."""
    if metric_means.empty:
        return None, None

    sorted_models = metric_means.sort_values(ascending=not higher_is_better)
    best_model = sorted_models.index[0] if len(sorted_models) > 0 else None
    second_best_model = (
        sorted_models.index[1] if len(sorted_models) > 1 else None
    )
    return best_model, second_best_model


def get_best_models(
    metric_means: pd.Series, higher_is_better: bool
) -> Optional[str]:
    """Sorts models by performance and returns the best."""
    best_model, _ = get_best_and_second_best_models(
        metric_means, higher_is_better
    )
    return best_model


def format_cell(
    mean_val: float,
    low_val: float,
    high_val: float,
    model: str,
    best_model: Optional[str],
    second_best_model: Optional[str] = None,
) -> str:
    """Formats a single cell for the LaTeX table."""
    if pd.isna(mean_val):
        return "-"

    mean_str = format_number(mean_val)

    if pd.isna(low_val) or pd.isna(high_val):
        # Use makecell with empty second line to align with CI values
        cell_str = f"\\makecell{{{mean_str} \\\\ \\scriptsize{{ }}}}"
    else:
        low_str = format_number(low_val)
        high_str = format_number(high_val)
        cell_str = f"\\makecell{{{mean_str} \\\\ \\scriptsize{{({low_str}, {high_str})}}}}"

    if model == best_model:
        cell_str = f"\\textcolor{{cbred}}{{{cell_str}}}"
    elif model == second_best_model:
        cell_str = f"\\textcolor{{cbblue}}{{{cell_str}}}"
    return cell_str


def generate_merged_latex_table(df: pd.DataFrame) -> str:
    """Generates a single LaTeX table with all datasets separated by rules."""

    datasets = sorted(df["dataset"].unique())

    metrics_available = [m for m in METRIC_ORDER if m in df["metric"].unique()]
    metric_display_names = [
        f"\\textbf{{{get_display_name(m, METRIC_DISPLAY_NAMES)}}}"
        for m in metrics_available
    ]
    header_line = (
        " & ".join(
            ["\\textbf{Dataset}", "\\textbf{Model}"] + metric_display_names
        )
        + " \\\\"
    )

    table_lines = []
    table_lines.append("\\begin{table*}")
    table_lines.append("\\centering")
    table_lines.append(
        "\\setlength{\\tabcolsep}{4pt}  % Reduce horizontal spacing between columns"
    )
    table_lines.append(
        "\\renewcommand{\\arraystretch}{0.9}  % Reduce vertical spacing between rows"
    )
    table_lines.append("\\caption{Benchmark results across all datasets.}")
    table_lines.append("\\label{tab:merged_results}")
    table_lines.append("\\scalebox{0.85}{")
    table_lines.append(
        "\\begin{tabular}{ll" + "c" * len(metrics_available) + "}"
    )
    table_lines.append("\\toprule")
    table_lines.append(header_line)
    table_lines.append("\\midrule")

    for dataset_idx, dataset_name in enumerate(datasets):
        dataset_df = df[df["dataset"] == dataset_name]

        pivot_df = dataset_df.pivot(
            index="model", columns="metric", values=["mean", "low", "high"]
        )

        best_models_by_metric = {}
        second_best_models_by_metric = {}

        for metric in metrics_available:
            if metric in pivot_df.columns.levels[1]:
                is_higher_better = METRIC_HIGHER_IS_BETTER.get(metric, False)
                metric_means = pivot_df.loc[:, ("mean", metric)].dropna()
                best_model, second_best_model = get_best_and_second_best_models(
                    metric_means, is_higher_better
                )
                best_models_by_metric[metric] = best_model
                second_best_models_by_metric[metric] = second_best_model

        models = sorted(pivot_df.index)

        for model_idx, model in enumerate(models):
            cells = []

            if model_idx == 0:
                dataset_display = get_display_name(
                    dataset_name, DATASET_DISPLAY_NAMES
                )
                cells.append(f"{dataset_display}")
            else:
                cells.append("")

            model_display = get_display_name(model, MODEL_DISPLAY_NAMES)
            cells.append(model_display)

            for metric in metrics_available:
                if metric in pivot_df.columns.levels[1]:
                    mean_val = pivot_df.loc[model, ("mean", metric)]
                    low_val = pivot_df.loc[model, ("low", metric)]
                    high_val = pivot_df.loc[model, ("high", metric)]

                    formatted_cell = format_cell(
                        mean_val,
                        low_val,
                        high_val,
                        model,
                        best_models_by_metric.get(metric),
                        second_best_models_by_metric.get(metric),
                    )
                    cells.append(formatted_cell)
                else:
                    cells.append("-")

            table_lines.append(" & ".join(cells) + " \\\\")

        if dataset_idx < len(datasets) - 1:
            table_lines.append("\\midrule")

    table_lines.append("\\bottomrule")
    table_lines.append("\\end{tabular}")
    table_lines.append("}")
    table_lines.append("\\end{table*}")

    return "\n".join(table_lines)


def generate_latex_table_for_dataset(
    dataset_df: pd.DataFrame, dataset_name: str
) -> str:
    """Generates a styled LaTeX table for a given dataset."""
    pivot_df = dataset_df.pivot(
        index="model", columns="metric", values=["mean", "low", "high"]
    )

    formatted_df = pd.DataFrame(
        index=pivot_df.index, columns=pivot_df.columns.levels[1]
    )

    best_models_by_metric = {}
    second_best_models_by_metric = {}

    for metric in pivot_df.columns.levels[1]:
        is_higher_better = METRIC_HIGHER_IS_BETTER.get(metric, False)
        metric_means = pivot_df.loc[:, ("mean", metric)].dropna()
        best_model, second_best_model = get_best_and_second_best_models(
            metric_means, is_higher_better
        )
        best_models_by_metric[metric] = best_model
        second_best_models_by_metric[metric] = second_best_model

    for model in pivot_df.index:
        for metric in formatted_df.columns:
            mean_val = pivot_df.loc[model, ("mean", metric)]
            low_val = pivot_df.loc[model, ("low", metric)]
            high_val = pivot_df.loc[model, ("high", metric)]

            formatted_df.loc[model, metric] = format_cell(
                mean_val,
                low_val,
                high_val,
                model,
                best_models_by_metric[metric],
                second_best_models_by_metric[metric],
            )

    # Reindex by METRIC_ORDER on columns instead of rows
    formatted_df = formatted_df.reindex(columns=METRIC_ORDER).dropna(
        axis=1, how="all"
    )

    # Apply display names to column headers (with bold)
    display_columns = {
        col: f"\\textbf{{{get_display_name(col, METRIC_DISPLAY_NAMES)}}}"
        for col in formatted_df.columns
    }
    formatted_df = formatted_df.rename(columns=display_columns)

    # Apply display names to row indices (models)
    display_index = {
        idx: get_display_name(idx, MODEL_DISPLAY_NAMES)
        for idx in formatted_df.index
    }
    formatted_df = formatted_df.rename(index=display_index)

    dataset_display = get_display_name(dataset_name, DATASET_DISPLAY_NAMES)
    latex_table = formatted_df.style.to_latex(
        column_format="l" + "c" * len(formatted_df.columns),
        caption=f"Results for {dataset_display} dataset.",
        label=f"tab:{dataset_name.lower()}_results",
        hrules=True,
    )
    return latex_table


def generate_complete_latex_document(
    latex_content: str, merged: bool = False
) -> str:
    """Generates a complete LaTeX document."""
    document_header = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{makecell}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage[dvipsnames]{xcolor}
\usepackage{graphicx}

% Define colorblind-friendly colors (Okabe & Ito palette)
\definecolor{cbred}{RGB}{213,94,0}
\definecolor{cbblue}{RGB}{0,114,178}

\begin{document}
"""

    document_footer = r"""
\end{document}"""

    if merged:
        full_document = document_header + latex_content + "\n" + document_footer
    else:
        full_document = (
            document_header
            + "\n\n".join(latex_content)
            + "\n"
            + document_footer
        )

    return full_document


def main():
    """Main function to generate and save LaTeX document."""
    # Configure logging for the main function
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    results_path = (
        here() / "experiments" / "model_benchmark" / "results" / "results.csv"
    )
    df = load_and_preprocess_data(results_path)

    # Generate merged table
    merged_table = generate_merged_latex_table(df)
    complete_document = generate_complete_latex_document(
        merged_table, merged=True
    )

    # Save to file
    output_path = (
        here() / "experiments" / "model_benchmark" / "benchmark_results.tex"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(complete_document)

    logger.info("Merged LaTeX document saved to: %s", output_path)


if __name__ == "__main__":
    main()
