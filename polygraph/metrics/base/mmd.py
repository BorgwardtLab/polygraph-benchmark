"""Squared Maximum Mean Discrepancy (MMD) metrics for comparing graph distributions.

This module provides classes for computing MMD-based distances between sets of graphs.
We provide both single MMD estimates and uncertainty estimates through subsampling.

Available metrics:
    - [`DescriptorMMD2`][polygraph.metrics.base.mmd.DescriptorMMD2]: MMD with a single kernel
    - [`MaxDescriptorMMD2`][polygraph.metrics.base.mmd.MaxDescriptorMMD2]: Maximum MMD across multiple kernel hyperparameters (e.g. RBF bandwidths)
    - [`DescriptorMMD2Interval`][polygraph.metrics.base.mmd.DescriptorMMD2Interval]: Confidence intervals for MMD with a single kernel
    - [`MaxDescriptorMMD2Interval`][polygraph.metrics.base.mmd.MaxDescriptorMMD2Interval]: Confidence intervals for maximum MMD across multiple kernel hyperparameters

MMD metrics are initialized with a kernel function (see [`DescriptorKernel`][polygraph.utils.kernels.DescriptorKernel]) and a collection of reference graphs.

Example:
    ```python
    from polygraph.metrics.base import DescriptorMMD2, MaxDescriptorMMD2, DescriptorMMD2Interval
    from polygraph.utils.graph_descriptors import SparseDegreeHistogram
    from polygraph.utils.kernels import AdaptiveRBFKernel
    import networkx as nx
    import numpy as np

    reference_graphs = [nx.erdos_renyi_graph(10, 0.5) for _ in range(10)]
    generated_graphs = [nx.erdos_renyi_graph(10, 0.5) for _ in range(10)]

    kernel = AdaptiveRBFKernel(descriptor_fn=SparseDegreeHistogram(), bw=0.1)
    mmd = DescriptorMMD2(reference_graphs=reference_graphs, kernel=kernel)
    mmd_value = mmd.compute(generated_graphs)
    print(mmd_value)    # A single float value

    mmd_w_uncertainty = DescriptorMMD2Interval(reference_graphs=reference_graphs, kernel=kernel)
    mmd_interval = mmd_w_uncertainty.compute(generated_graphs, subsample_size=5, num_samples=100)
    print(mmd_interval)    # Named tuple with mean, standard deviation, and confidence interval bounds

    multi_kernel = AdaptiveRBFKernel(descriptor_fn=SparseDegreeHistogram(), bw=np.array([0.1, 0.2]))
    mmd = MaxDescriptorMMD2(reference_graphs=reference_graphs, kernel=multi_kernel)
    mmd_value = mmd.compute(generated_graphs)
    print(mmd_value)    # A single float value
    ```
"""

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Collection, Dict, Literal, Union

import networkx as nx
import numpy as np

from polygraph.utils.kernels import DescriptorKernel, GramBlocks
from polygraph.utils.mmd_utils import mmd_from_gram

__all__ = [
    "DescriptorMMD2",
    "MaxDescriptorMMD2",
    "MMDInterval",
    "DescriptorMMD2Interval",
    "MaxDescriptorMMD2Interval",
]


MMDInterval = namedtuple("MMDInterval", ["mean", "std", "low", "high"])


class DescriptorMMD2:
    """Computes squared MMD between reference and generated graphs using a kernel.

    Args:
        reference_graphs: Collection of graphs to compare against
        kernel: Kernel function for comparing graphs
        variant: Which MMD estimator to use ('biased', 'umve', or 'ustat')
    """

    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        kernel: DescriptorKernel,
        variant: Literal["biased", "umve", "ustat"] = "biased",
    ):
        self._kernel = kernel
        self._variant = variant
        self._reference_descriptions = self._kernel.featurize(reference_graphs)

    def compute(
        self, generated_graphs: Collection[nx.Graph]
    ) -> Union[float, np.ndarray]:
        """Computes MMD² between reference and generated graphs.

        Args:
            generated_graphs: Collection of graphs to evaluate

        Returns:
            MMD² value(s). Returns array if kernel has multiple parameters.
        """
        descriptions = self._kernel.featurize(generated_graphs)
        ref_vs_ref, ref_vs_gen, gen_vs_gen = self._kernel(
            self._reference_descriptions, descriptions
        )
        return mmd_from_gram(
            ref_vs_ref, gen_vs_gen, ref_vs_gen, variant=self._variant
        )


class MaxDescriptorMMD2(DescriptorMMD2):
    """Computes maximum MMD² across multiple kernel parameters.

    Similar to DescriptorMMD2 but takes the maximum across different kernel parameters
    (e.g., bandwidths). The kernel must support multiple parameters.

    Args:
        reference_graphs: Collection of graphs to compare against
        kernel: Kernel function with multiple parameters
        variant: Which MMD estimator to use ('biased', 'umve', or 'ustat')

    Raises:
        ValueError: If kernel does not have multiple parameters
    """

    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        kernel: DescriptorKernel,
        variant: Literal["biased", "umve", "ustat"] = "biased",
    ):
        super().__init__(
            reference_graphs=reference_graphs, kernel=kernel, variant=variant
        )
        if not self._kernel.num_kernels > 1:
            raise ValueError(
                "Must provide several kernels, i.e. a kernel with multiple parameters"
            )

    def compute(self, generated_graphs: Collection[nx.Graph]) -> float:
        """Computes maximum MMD² between reference and generated graphs.

        Args:
            generated_graphs: Collection of graphs to evaluate

        Returns:
            Maximum MMD² value across kernel parameters
        """
        multi_kernel_result = super().compute(generated_graphs)
        idx = int(np.argmax(multi_kernel_result))
        return multi_kernel_result[idx]


class _DescriptorMMD2Interval(ABC):
    """Base class for computing MMD² confidence intervals through subsampling."""

    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        kernel: DescriptorKernel,
        variant: Literal["biased", "umve", "ustat"] = "biased",
    ):
        self._kernel = kernel
        self._variant = variant
        self._reference_descriptions = self._kernel.featurize(reference_graphs)

    def _generate_mmd_samples(
        self,
        generated_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
    ) -> np.ndarray:
        descriptions = self._kernel.featurize(
            generated_graphs,
        )
        mmd_samples = []
        rng = np.random.default_rng(42)

        ref_vs_ref, ref_vs_gen, gen_vs_gen = self._kernel.pre_gram(
            self._reference_descriptions, descriptions
        )

        for _ in range(num_samples):
            ref_idxs = rng.choice(
                len(ref_vs_ref), size=subsample_size, replace=False
            )
            gen_idxs = rng.choice(
                len(gen_vs_gen), size=subsample_size, replace=False
            )
            sub_ref_vs_ref = ref_vs_ref[ref_idxs][:, ref_idxs]
            sub_gen_vs_gen = gen_vs_gen[gen_idxs][:, gen_idxs]
            sub_ref_vs_gen = ref_vs_gen[ref_idxs][:, gen_idxs]
            sub_ref_vs_ref, sub_ref_vs_gen, sub_gen_vs_gen = self._kernel.adapt(
                GramBlocks(sub_ref_vs_ref, sub_ref_vs_gen, sub_gen_vs_gen)
            )
            mmd_samples.append(
                mmd_from_gram(
                    sub_ref_vs_ref,
                    sub_gen_vs_gen,
                    sub_ref_vs_gen,
                    variant=self._variant,
                )
            )

        mmd_samples = np.array(mmd_samples)
        return mmd_samples

    @abstractmethod
    def compute(*args, **kwargs) -> Union[MMDInterval, Dict[str, float]]: ...


class DescriptorMMD2Interval(_DescriptorMMD2Interval):
    """Computes MMD² confidence intervals using subsampling.

    Estimates uncertainty in MMD² by repeatedly computing it on random subsamples
    of the reference and generated graphs.

    Args:
        reference_graphs: Collection of graphs to compare against
        kernel: Kernel function for comparing graphs
        variant: Which MMD estimator to use ('biased', 'umve', or 'ustat')
    """

    def compute(
        self,
        generated_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
        coverage: float = 0.95,
        as_scalar_value_dict: bool = False,
    ) -> Union[MMDInterval, Dict[str, float]]:
        """Computes MMD² confidence intervals through subsampling.

        Args:
            generated_graphs: Collection of graphs to evaluate
            subsample_size: Number of graphs to use in each MMD² sample, should be consistent with the sample size in point estimates.
            num_samples: Number of MMD² samples to generate
            coverage: Confidence level to compute upper and lower bounds

        Returns:
            Named tuple with mean, standard deviation, and confidence interval bounds
        """
        mmd_samples = self._generate_mmd_samples(
            generated_graphs=generated_graphs,
            subsample_size=subsample_size,
            num_samples=num_samples,
        )
        low, high = (
            np.quantile(mmd_samples, (1 - coverage) / 2, axis=0),
            np.quantile(mmd_samples, coverage + (1 - coverage) / 2, axis=0),
        )
        avg = np.mean(mmd_samples, axis=0)
        std = np.std(mmd_samples, axis=0)
        if as_scalar_value_dict:
            return {
                "mean": avg,
                "std": std,
                "low": low,
                "high": high,
            }
        return MMDInterval(mean=avg, std=std, low=low, high=high)


class MaxDescriptorMMD2Interval(_DescriptorMMD2Interval):
    """Computes confidence intervals for maximum MMD² across kernel parameters.

    Similar to DescriptorMMD2Interval but takes the maximum across different kernel
    parameters for each subsample. I.e., it quantifies the uncertainty of the point estimates
    made in [`MaxDescriptorMMD2`][polygraph.metrics.base.mmd.MaxDescriptorMMD2].

    Args:
        reference_graphs: Collection of graphs to compare against
        kernel: Kernel function with multiple parameters
        variant: Which MMD estimator to use ('biased', 'umve', or 'ustat')

    Raises:
        ValueError: If kernel does not have multiple parameters
    """

    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        kernel: DescriptorKernel,
        variant: Literal["biased", "umve", "ustat"] = "biased",
    ):
        super().__init__(
            reference_graphs=reference_graphs, kernel=kernel, variant=variant
        )
        if not self._kernel.num_kernels > 1:
            raise ValueError(
                "Must provide several kernels, i.e. either a kernel with multiple parameters"
            )

    def compute(
        self,
        generated_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
        coverage: float = 0.95,
        as_scalar_value_dict: bool = False,
    ) -> Union[MMDInterval, Dict[str, float]]:
        """Computes confidence intervals for maximum MMD² through subsampling.

        Args:
            generated_graphs: Collection of graphs to evaluate
            subsample_size: Number of graphs to use in each subsample
            num_samples: Number of subsamples to generate
            coverage: Confidence level (e.g., 0.95 for 95% intervals)

        Returns:
            Named tuple with mean, standard deviation, and confidence interval bounds for the maximum MMD² across kernel parameters
        """
        mmd_samples = self._generate_mmd_samples(
            generated_graphs=generated_graphs,
            subsample_size=subsample_size,
            num_samples=num_samples,
        )
        assert mmd_samples.ndim == 2
        mmd_samples = np.max(mmd_samples, axis=1)
        low, high = (
            np.quantile(mmd_samples, (1 - coverage) / 2, axis=0),
            np.quantile(mmd_samples, coverage + (1 - coverage) / 2, axis=0),
        )
        avg = np.mean(mmd_samples, axis=0)
        std = np.std(mmd_samples, axis=0)
        if as_scalar_value_dict:
            return {
                "mean": avg,
                "std": std,
                "low": low,
                "high": high,
            }
        return MMDInterval(mean=avg, std=std, low=low, high=high)
