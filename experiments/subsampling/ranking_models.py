import pandas as pd
from loguru import logger
from pyprojroot import here


def main():
    df = pd.read_csv(here() / "experiments/results/bootstrapping.csv")
    df = df[df.generation_procedure == "procedural"]
    for dataset in df["dataset_type"].unique():
        for descriptor in df["descriptor"].unique():
            df_subset = df[
                (df["dataset_type"] == dataset)
                & (df["descriptor"] == descriptor)
                & (df["n_graphs"] == 1024)
                & (df["variant"] == "umve")
                & (df["test_set_type"] != "test")
            ]
            df_subset = df_subset.sort_values(
                by="mmd_results_mean", ascending=True
            )
            rankings = df_subset.test_set_type.to_list()
            logger.info(
                f"Dataset: {dataset}, Descriptor: {descriptor}, Ranking: {rankings}"
            )


def get_unique_lists_order_matters(list_of_lists):
    """
    Finds unique sublists from a list of lists where the order of elements
    within each sublist matters for uniqueness.

    Args:
        list_of_lists: A list of lists (e.g., [[1, 2], [3, 4], [1, 2]]).

    Returns:
        A list of unique sublists, preserving the order of first appearance.
    """
    seen_tuples = set()
    unique_sublists = []
    for sublist in list_of_lists:
        # Ensure sublist elements are hashable; tuples are suitable.
        # If sublist elements are mutable (e.g. other lists), they must also be converted.
        try:
            sublist_tuple = tuple(sublist)
        except TypeError:
            # Handle cases where sublist elements might not be directly hashable
            # One common way is to convert them to strings or other immutable representations
            sublist_tuple = tuple(
                str(item) for item in sublist
            )  # Example conversion

        if sublist_tuple not in seen_tuples:
            seen_tuples.add(sublist_tuple)
            unique_sublists.append(list(sublist))  # Convert back to list
    return unique_sublists


if __name__ == "__main__":
    main()
