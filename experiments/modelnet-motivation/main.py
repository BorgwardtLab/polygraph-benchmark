from loguru import logger
from argparse import ArgumentParser
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from functools import wraps
from typing import Collection
import networkx as nx
import pandas as pd

from polygraph.datasets import PointCloudGraphDataset, ModelNet10GraphDataset
from polygraph.utils.graph_descriptors import (
    OrbitCounts,
    DegreeHistogram,
    ClusteringHistogram,
    EigenvalueHistogram,
    RandomGIN,
)
from polygraph.metrics.gran import (
    RBFClusteringMMD2,
    RBFOrbitMMD2,
    RBFSpectralMMD2,
)
from polygraph.metrics.gin import (
    RBFGraphNeuralNetworkMMD2,
)
from polygraph.metrics.base.mmd import MaxDescriptorMMD2
from polygraph.utils.kernels import AdaptiveRBFKernel


def memoize_individual_descriptors(verbose: bool = False):
    """Monkey-patch all graph descriptor __call__ methods with memoization"""

    # Dictionary to hold all caches and stats
    caches = {}
    cache_stats = {}

    descriptor_classes = [
        OrbitCounts,
        DegreeHistogram,
        ClusteringHistogram,
        EigenvalueHistogram,
        RandomGIN,
    ]

    for desc_class in descriptor_classes:
        class_name = desc_class.__name__

        # Store original method
        original_call = desc_class.__call__

        # Create cache and stats for this class
        caches[class_name] = {}

        def create_memoized_call(class_name, original_call):
            @wraps(original_call)
            def memoized_call(self, graphs):
                cache_keys = tuple(id(g) for g in graphs)
                known_keys = [
                    graph_id
                    for graph_id in cache_keys
                    if graph_id in caches[class_name]
                ]
                unknown_keys = [
                    graph_id
                    for graph_id in cache_keys
                    if graph_id not in caches[class_name]
                ]
                unknown_graphs = [g for g in graphs if id(g) in unknown_keys]

                if verbose:
                    print(
                        f"{class_name} cache hits: {len(known_keys)}, misses: {len(unknown_keys)}"
                    )

                if len(unknown_keys) > 0:
                    unknown_results = list(original_call(self, unknown_graphs))
                else:
                    unknown_results = []

                for graph_id, result in zip(unknown_keys, unknown_results):
                    caches[class_name][graph_id] = result

                all_results = [
                    caches[class_name][graph_id] for graph_id in cache_keys
                ]
                return np.stack(all_results, axis=0)

            return memoized_call

        # Apply monkey-patch
        desc_class.__call__ = create_memoized_call(class_name, original_call)

    return caches, cache_stats


class RBFDenseDegreeMMD2(MaxDescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=DegreeHistogram(max_degree=100),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            variant="biased",
        )


class AggregateMMD:
    def __init__(self, reference):
        self._metrics = {
            "orbit_mmd": RBFOrbitMMD2(reference),
            "degree_mmd": RBFDenseDegreeMMD2(reference),
            "spectral_mmd": RBFSpectralMMD2(reference),
            "clustering_mmd": RBFClusteringMMD2(reference),
            "gin_mmd": RBFGraphNeuralNetworkMMD2(reference),
        }

    def compute(self, data):
        return {
            key: metric.compute(data) for key, metric in self._metrics.items()
        }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pointcloud")
    parser.add_argument("--num-subsamples", type=int, default=25)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    )
    args = parser.parse_args()

    if args.dataset == "pointcloud":
        ds_cls = PointCloudGraphDataset
    elif args.dataset == "modelnet10":
        ds_cls = ModelNet10GraphDataset
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    train = list(ds_cls("train").to_nx())
    val = list(ds_cls("val").to_nx())
    test = list(ds_cls("test").to_nx())
    all_graphs = train + val + test

    logger.info(f"Dataset sizes: {len(train)}, {len(val)}, {len(test)}")
    logger.disable("polygraph")

    memoize_individual_descriptors(verbose=False)

    rng = np.random.default_rng(42)

    all_samples = defaultdict(lambda: defaultdict(list))

    for size in tqdm(args.sizes, ascii=False, desc="Sample Sizes"):
        assert size <= len(all_graphs) // 2, (
            "Size must be less than half of the dataset"
        )
        for _ in tqdm(
            range(args.num_subsamples), leave=False, desc="Subsamples"
        ):
            indices = rng.choice(len(all_graphs), size=2 * size, replace=False)
            subset = [all_graphs[i] for i in indices]
            reference, data = subset[:size], subset[size:]
            mmd = AggregateMMD(reference)
            mmd_values = mmd.compute(data)
            for key, value in mmd_values.items():
                all_samples[size][key].append(value)

    mean = pd.DataFrame(
        {
            size: {key: np.mean(values) for key, values in samples.items()}
            for size, samples in all_samples.items()
        }
    )
    std = pd.DataFrame(
        {
            size: {key: np.std(values) for key, values in samples.items()}
            for size, samples in all_samples.items()
        }
    )

    print(mean)
    print(std)
