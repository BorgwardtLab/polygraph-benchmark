from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Collection, Literal, Tuple

import networkx as nx
import numpy as np

from graph_gen_gym.metrics.utils.graph_descriptors import (
    ClusteringHistogram,
    DegreeHistogram,
    EigenvalueHistogram,
    OrbitCounts,
)
from graph_gen_gym.metrics.utils.kernels import (
    AdaptiveRBFKernel,
    DescriptorKernel,
    GaussianTV,
    StackedKernel,
)
from graph_gen_gym.metrics.utils.mmd_utils import mmd_from_gram, mmd_ustat_var

MMDWithVariance = namedtuple("MMDWithVariance", ["ustat", "std"])
MMDInterval = namedtuple("MMDInterval", ["mean", "std", "low", "high"])


class DescriptorMMD2:
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        kernel: DescriptorKernel,
        variant: Literal["biased", "umve", "ustat", "ustat-var"] = "biased",
    ):
        self._kernel = kernel
        self._variant = variant
        self._reference_descriptions = self._kernel.featurize(reference_graphs)

    def compute(self, generated_graphs: Collection[nx.Graph]):
        descriptions = self._kernel.featurize(
            generated_graphs,
        )
        ref_vs_ref, ref_vs_gen, gen_vs_gen = self._kernel(
            self._reference_descriptions, descriptions
        )
        if self._variant == "ustat-var":
            assert (
                self._kernel.num_kernels == 1
            ), "Only single kernel supported for USTAT-VAR"
            mmd = mmd_from_gram(ref_vs_ref, gen_vs_gen, ref_vs_gen, variant="ustat")
            var = mmd_ustat_var(ref_vs_ref, ref_vs_gen, gen_vs_gen)
            return MMDWithVariance(ustat=mmd, std=np.sqrt(var))
        return mmd_from_gram(ref_vs_ref, gen_vs_gen, ref_vs_gen, variant=self._variant)


class MaxDescriptorMMD2(DescriptorMMD2):
    """
    Compute the maximal MMD across different kernel choices.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel.num_kernels > 1:
            raise ValueError(
                f"Must provide several kernels, i.e. either a {StackedKernel.__name__} or a kernel with multiple parameters"
            )

    def compute(
        self, generated_graphs: Collection[nx.Graph]
    ) -> Tuple[float, DescriptorKernel]:
        multi_kernel_result = super().compute(generated_graphs)
        idx = np.argmax(multi_kernel_result)
        return multi_kernel_result[idx], self._kernel.get_subkernel(idx)


class _MMD2Intereval(ABC):
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
    ) -> MMDInterval:
        descriptions = self._kernel.featurize(
            generated_graphs,
        )
        ref_vs_ref, ref_vs_gen, gen_vs_gen = self._kernel(
            self._reference_descriptions, descriptions
        )
        mmd_samples = []
        rng = np.random.default_rng(42)
        for _ in range(num_samples):
            ref_idxs = rng.choice(len(ref_vs_ref), size=subsample_size, replace=False)
            gen_idxs = rng.choice(len(gen_vs_gen), size=subsample_size, replace=False)
            sub_ref_vs_ref = ref_vs_ref[ref_idxs][:, ref_idxs]
            sub_gen_vs_gen = gen_vs_gen[gen_idxs][:, gen_idxs]
            sub_ref_vs_gen = ref_vs_gen[ref_idxs][:, gen_idxs]
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
    def compute(*args, **kwargs) -> MMDInterval:
        ...


class DescriptorMMD2Interval(_MMD2Intereval):
    def compute(
        self,
        generated_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
        coverage: float = 0.95,
    ) -> MMDInterval:
        mmd_samples = self._generate_mmd_samples(
            generated_graphs=generated_graphs,
            subsample_size=subsample_size,
            num_samples=num_samples,
        )
        low, high = np.quantile(mmd_samples, (1 - coverage) / 2, axis=0), np.quantile(
            mmd_samples, coverage + (1 - coverage) / 2, axis=0
        )
        avg = np.mean(mmd_samples, axis=0)
        std = np.std(mmd_samples, axis=0)
        return MMDInterval(mean=avg, std=std, low=low, high=high)


class MaxDescriptorMMD2Interval(_MMD2Intereval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel.num_kernels > 1:
            raise ValueError(
                f"Must provide several kernels, i.e. either a {StackedKernel.__name__} or a kernel with multiple parameters"
            )
        if isinstance(self._kernel, AdaptiveRBFKernel):
            raise ValueError(
                "Cannot use AdaptiveRBFKernel with uncertainty quantification. Use an RBFKernel with various bandwidths instead."
            )

    def compute(
        self,
        generated_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
        coverage: float = 0.95,
    ) -> MMDInterval:
        mmd_samples = self._generate_mmd_samples(
            generated_graphs=generated_graphs,
            subsample_size=subsample_size,
            num_samples=num_samples,
        )
        assert mmd_samples.ndim == 2
        mmd_samples = np.max(mmd_samples, axis=1)
        low, high = np.quantile(mmd_samples, (1 - coverage) / 2, axis=0), np.quantile(
            mmd_samples, coverage + (1 - coverage) / 2, axis=0
        )
        avg = np.mean(mmd_samples, axis=0)
        std = np.std(mmd_samples, axis=0)
        return MMDInterval(mean=avg, std=std, low=low, high=high)


class GRANOrbitMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(descriptor_fn=OrbitCounts(), bw=30),
        )


class GRANClusteringMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(descriptor_fn=ClusteringHistogram(bins=100), bw=1.0 / 10),
        )


class GRANDegreeMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph], max_degree: int):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(
                descriptor_fn=DegreeHistogram(max_degree=max_degree), bw=1.0
            ),
        )


class GRANSpectralMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(descriptor_fn=EigenvalueHistogram(), bw=1.0),
        )
