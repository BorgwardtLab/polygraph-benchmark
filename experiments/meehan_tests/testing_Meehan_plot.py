import os
import re
from typing import Tuple

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from pyprojroot import here

from polygraph.datasets import (
    LobsterGraphDataset,
    PlanarGraphDataset,
    SBMGraphDataset,
)


def setup_plotting_parameters(
    resolution: int = 600, size: Tuple[float, float] = (4.8, 4)
) -> None:
    plt.rcParams["figure.figsize"] = size
    plt.rcParams["savefig.dpi"] = resolution

    cmfont = font_manager.FontProperties(
        fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf"
    )
    font_name = cmfont.get_name()

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [font_name]
    mpl.rcParams["font.monospace"] = [font_name]
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["axes.unicode_minus"] = False


def format_descriptor(descriptor: str) -> str:
    """Format a descriptor string by splitting before the first capitalized character in camelCase.

    Args:
        descriptor: The descriptor string to format

    Returns:
        The formatted descriptor string
    """
    if not descriptor or len(descriptor) <= 1:
        return descriptor

    for i in range(1, len(descriptor)):
        if descriptor[i].isupper():
            return descriptor[:i] + " " + descriptor[i:]

    return descriptor


def format_generated_set_name(generated_set_type: str, n_train: int) -> str:
    if "gran" in generated_set_type:
        model_name = "GRAN"
    elif "autograph" in generated_set_type:
        model_name = "AutoGraph"
    elif "digress" in generated_set_type:
        model_name = "DiGress"
    else:
        raise ValueError(f"Unknown generated set type: {generated_set_type}")

    return f"{model_name} (Trained on {int(n_train)} Samples)"


def format_dataset(dataset: str) -> str:
    if "lobster" in dataset:
        return "Lobster"
    elif "sbm" in dataset:
        return "SBM"
    elif "planar" in dataset:
        return "Planar"


def add_training_sample_size(df: pd.DataFrame) -> pd.DataFrame:
    n_train_lobster_fixed = len(LobsterGraphDataset(split="train"))
    n_train_sbm_fixed = len(SBMGraphDataset(split="train"))
    n_train_planar_fixed = len(PlanarGraphDataset(split="train"))
    n_val_lobster_fixed = len(LobsterGraphDataset(split="val"))
    n_val_sbm_fixed = len(SBMGraphDataset(split="val"))
    n_val_planar_fixed = len(PlanarGraphDataset(split="val"))
    n_test_lobster_fixed = len(LobsterGraphDataset(split="test"))
    n_test_sbm_fixed = len(SBMGraphDataset(split="test"))
    n_test_planar_fixed = len(PlanarGraphDataset(split="test"))
    n_train_procedural = 8192

    # gran_val_procedural = 256
    # gran_test_procedural = 256
    # gran_train_procedural = 8192

    # digress_val_procedural = 1024
    # digress_test_procedural = 1024
    # digress_train_procedural = 8192

    # autograph_val_procedural = 1024
    # autograph_test_procedural = 1024
    # autograph_train_procedural = 8192

    logger.info("Adding training sample size to dataframe.")
    # ! We're adding the test and val sample sizes from the fixed datasets
    for generated_graph_type in df.generated_graph_type.unique():
        for dataset_type in df.dataset.unique():
            if "procedural" in generated_graph_type and "sbm" in dataset_type:
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_train"
                ] = n_train_procedural
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_val"
                ] = n_val_sbm_fixed
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_test"
                ] = n_test_sbm_fixed
            elif (
                "procedural" in generated_graph_type
                and "planar" in dataset_type
            ):
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_train"
                ] = n_train_procedural
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_val"
                ] = n_val_planar_fixed
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_test"
                ] = n_test_planar_fixed
            elif (
                "procedural" in generated_graph_type
                and "lobster" in dataset_type
            ):
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_train"
                ] = n_train_procedural
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_val"
                ] = n_val_lobster_fixed
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_test"
                ] = n_test_lobster_fixed

            if "fixed" in generated_graph_type and "sbm" in dataset_type:
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_train"
                ] = n_train_sbm_fixed
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_val"
                ] = n_val_sbm_fixed
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_test"
                ] = n_test_sbm_fixed
            elif "fixed" in generated_graph_type and "planar" in dataset_type:
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_train"
                ] = n_train_planar_fixed
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_val"
                ] = n_val_planar_fixed
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_test"
                ] = n_test_planar_fixed
            elif "fixed" in generated_graph_type and "lobster" in dataset_type:
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_train"
                ] = n_train_lobster_fixed
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_val"
                ] = n_val_lobster_fixed
                df.loc[
                    df.generated_graph_type == generated_graph_type, "n_test"
                ] = n_test_lobster_fixed

    return df


def make_plot(df):
    df.rename(
        columns={
            "train_test_distance_p_value": "Overfitting test",
            "train_test_celled_distance_p_value": "Overfitting celled test",
        },
        inplace=True,
    )
    for training_set in df.training_dataset.unique():
        df_training_set = df[df.training_dataset == training_set]
        for dataset in df_training_set.dataset.unique():
            df_training_set_dataset = df_training_set[
                df_training_set.dataset == dataset
            ]
            for descriptor in df_training_set_dataset.descriptor.unique():
                for model in df_training_set_dataset.model.unique():
                    df_descriptor = df_training_set_dataset[
                        (df_training_set_dataset.descriptor == descriptor)
                        & (df_training_set_dataset.model == model)
                    ]
                    if df_descriptor.empty:
                        continue
                    # Create a single plot
                    fig, ax = plt.subplots(figsize=(8, 5))

                    # Get the first value (previously shown in first plot)
                    first_value = df_descriptor["Overfitting test"].iloc[0]
                    first_value_text = (
                        "NULL" if pd.isna(first_value) else f"{first_value:.2e}"
                    )

                    # Plot for Overfitting celled test
                    sns.barplot(
                        data=df_descriptor.fillna(-0.05),
                        x="k",
                        y="Overfitting celled test",
                        ax=ax,
                    )
                    ax.set_title(
                        f"Overfitting Tests -- Model: {model}, Dataset: {dataset}, \n "
                        f"Training Set: {training_set}, Descriptor: {format_descriptor(descriptor)}\n"
                        f"Overfitting Test p-value: {first_value_text}"
                    )
                    ax.set_ylim(
                        -0.15, 1.05
                    )  # Extended range to show null values
                    ax.set_xlabel("k")
                    ax.set_ylabel("p-value")
                    # Add text for null values and p-values
                    for i, v in enumerate(
                        df_descriptor["Overfitting celled test"]
                    ):
                        if pd.isna(v):
                            ax.text(i, -0.1, "NULL", ha="center")
                        else:
                            ax.text(i, v + 0.02, f"{v:.2e}", ha="center")
                    plt.tight_layout()
                    plt.savefig(
                        here()
                        / "experiments"
                        / "figures"
                        / f"testing_{dataset}_{training_set}_{descriptor}_{model}.pdf"
                    )
                    plt.close()


def format_generated_graph_type(generated_graph_type: str) -> str:
    if "gran" in generated_graph_type:
        model_name = "GRAN"
    elif "autograph" in generated_graph_type:
        model_name = "AutoGraph"
    elif "digress" in generated_graph_type:
        model_name = "DiGress"
    else:
        raise ValueError(
            f"Unknown generated graph type: {generated_graph_type}"
        )

    if "fixed" in generated_graph_type:
        return f"{model_name} (Fixed)"
    elif "procedural" in generated_graph_type:
        return f"{model_name} (Procedural)"
    else:
        raise ValueError(
            f"Unknown generated graph type: {generated_graph_type}"
        )


def make_table(df):
    def get_cell_error_code(message):
        if isinstance(message, str):
            if re.match(
                r"Error computing pval_celled: Cell (\d+) lacks test", message
            ):
                return "Cells Empty"
            elif re.match(r"Error computing pval_celled: boolean", message):
                return "No samples"
        return message

    reason_mapping = {
        "Successfully computed pval_celled": "NaN",
        "No graphs generated for this configuration": "No Graphs",
    }

    df["reason"] = df["reason"].apply(get_cell_error_code)
    df["reason"] = df["reason"].map(lambda x: reason_mapping.get(x, x))
    df_table_wo_overfit_test = df.drop(columns=["train_test_distance_p_value"])

    df_table_wo_overfit_test["k"] = df_table_wo_overfit_test.k.apply(
        lambda x: f"Underfitting test (k={x})"
    )

    df_table_wo_overfit_test = df_table_wo_overfit_test.pivot(
        index=[
            "dataset",
            "training_dataset",
            "generated_graph_type",
            "descriptor",
        ],
        columns=["k"],
        values="train_test_celled_distance_p_value",
    )

    df_table_overfit_test = df.drop(
        columns=["k", "train_test_celled_distance_p_value"]
    ).drop_duplicates(
        subset=[
            "dataset",
            "training_dataset",
            "generated_graph_type",
            "descriptor",
        ]
    )
    df_table = pd.merge(
        df_table_wo_overfit_test,
        df_table_overfit_test,
        on=[
            "dataset",
            "training_dataset",
            "generated_graph_type",
            "descriptor",
        ],
    )
    df_table = df_table.rename(
        columns={
            "train_test_distance_p_value": "Overfitting test",
        }
    )
    df_table = add_training_sample_size(df_table)
    df_table["descriptor"] = df_table["descriptor"].apply(format_descriptor)
    df_table = df_table.drop(columns=["training_set_size", "test_set_size"])

    underfitting_test_cols = [
        col
        for col in df_table.columns
        if col.startswith("Underfitting test (k=")
    ]
    test_columns_to_fill = underfitting_test_cols
    if "Overfitting test" in df_table.columns:
        test_columns_to_fill.append("Overfitting test")

    for col_name in test_columns_to_fill:
        if col_name in df_table.columns:  # Ensure column exists
            df_table[col_name] = df_table[col_name].fillna(df_table["reason"])

        # for generated_graph_type in df_table.generated_graph_type.unique():
    for dataset in df_table.dataset.unique():
        model_data = df_table.loc[
            (df_table.dataset == dataset)
            & (df_table.training_dataset == "procedural")
        ]
        model_data["generated_graph_type"] = model_data[
            "generated_graph_type"
        ].apply(format_generated_graph_type)
        model_data = model_data.drop(columns=["training_dataset"])
        model_data = model_data.set_index(
            [
                # "dataset",
                # "training_dataset",
                "generated_graph_type",
                "descriptor",
            ]
        )
        model_data = model_data.drop(columns=["n_train", "n_val", "n_test"])

        current_data_columns = list(model_data.columns)

        new_multi_index_tuples = []
        ordered_original_column_names = []

        underfitting_cols_to_process = []
        for col_name in current_data_columns:
            if col_name.startswith("Underfitting test (k="):
                underfitting_cols_to_process.append(col_name)

        def sort_key_k_val(name_of_col):
            match = re.search(r"k=(\d+)", name_of_col)
            return int(match.group(1)) if match else float("inf")

        underfitting_cols_to_process.sort(key=sort_key_k_val)

        for col_name in underfitting_cols_to_process:
            match = re.search(r"(k=\d+)", col_name)
            sub_header = match.group(1) if match else ""
            new_multi_index_tuples.append(("Underfitting tests", sub_header))
            ordered_original_column_names.append(col_name)

        overfitting_col_name_const = "Overfitting test"
        if overfitting_col_name_const in current_data_columns:
            new_multi_index_tuples.append(("Overfitting test", ""))
            ordered_original_column_names.append(overfitting_col_name_const)

        unprocessed_cols = [
            col
            for col in current_data_columns
            if col not in ordered_original_column_names
        ]
        for col_name in unprocessed_cols:
            new_multi_index_tuples.append(("Other Data", col_name))
            ordered_original_column_names.append(col_name)

        if ordered_original_column_names:
            model_data = model_data[ordered_original_column_names]

            model_data.columns = pd.MultiIndex.from_tuples(
                new_multi_index_tuples,
                names=[
                    "",
                    "",
                ],
            )
        model_data = model_data[["Overfitting test", "Underfitting tests"]]
        model_data = model_data.rename_axis(
            index={
                # "dataset": "Dataset",
                "generated_graph_type": "Generated Graph Type",
                "descriptor": "Descriptor",
            }
        )

        def bold_small_pvals(x):
            if isinstance(x, (int, float)) and not pd.isna(x) and x < 0.001:
                return f"\\textbf{{{x:.3e}}}"
            elif isinstance(x, (int, float)) and not pd.isna(x):
                return f"{x:.3e}"
            return x

        formatters = {}
        for col in model_data.columns:
            formatters[col] = bold_small_pvals

        latex_content = model_data.to_latex(
            index=True,
            multicolumn=True,
            multirow=True,
            caption=f"Testing results for {format_dataset(dataset)} dataset.",
            label=f"tab:testing_{dataset}",
            formatters=formatters,
            multicolumn_format="c",
            column_format="c" * (len(model_data.columns.levels[1]) + 1),
        )

        lines = latex_content.split("\n")
        bottomrule_index = next(
            (i for i, line in enumerate(lines) if "\\bottomrule" in line), -1
        )
        if bottomrule_index > 0 and (
            "\\cline" in lines[bottomrule_index - 1]
            or "\\midrule" in lines[bottomrule_index - 1]
        ):
            lines.pop(bottomrule_index - 1)
        latex_content = "\n".join(lines)

        if "\\begin{table}" not in latex_content:
            latex_content = (
                "\\begin{table}[htbp]\n"
                "\\centering\n"
                f"{latex_content}\n"
                "\\end{table}"
            )

        with open(
            here() / "experiments" / "tables" / f"testing_{dataset}.tex",
            "w",
        ) as f:
            f.write(latex_content)

    return df_table


def main():
    setup_plotting_parameters(size=(8, 5))
    df = pd.read_csv(here() / "experiments" / "results" / "testing_results.csv")
    # make_plot(df)
    os.makedirs(here() / "experiments" / "tables", exist_ok=True)
    make_table(df)


if __name__ == "__main__":
    main()
