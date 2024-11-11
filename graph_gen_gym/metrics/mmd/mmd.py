from collections import namedtuple
from functools import partial
from typing import Iterable, Literal, Tuple

import networkx as nx
import numpy as np

from graph_gen_gym.datasets.abstract_dataset import AbstractDataset
from graph_gen_gym.metrics.mmd.graph_descriptors import ClusteringHistogram, OrbitCounts
from graph_gen_gym.metrics.mmd.kernels import DescriptorKernel, GaussianTV
from graph_gen_gym.metrics.mmd.utils import mmd_from_gram, mmd_ustat_var

MMDInterval = namedtuple("MMDInterval", ["ustat", "std"])


class DescriptorMMD2:
    def __init__(
        self,
        reference_graphs: Iterable[nx.Graph],
        kernel: DescriptorKernel,
        variant: Literal["biased", "umve", "ustat", "ustat-var"] = "biased",
    ):
        self._kernel = kernel
        self._variant = variant
        self._reference_descriptions = self._kernel.featurize(reference_graphs)

        self._ref_vs_ref = self._kernel(
            self._reference_descriptions, self._reference_descriptions
        )

    def compute(self, generated_graphs: Iterable[nx.Graph]):
        descriptions = self._kernel.featurize(
            generated_graphs,
        )
        gen_vs_gen = self._kernel(descriptions, descriptions)
        gen_vs_ref = self._kernel(descriptions, self._reference_descriptions)
        if self._variant == "ustat-var":
            mmd = mmd_from_gram(
                gen_vs_gen, self._ref_vs_ref, gen_vs_ref, variant="ustat"
            )
            var = mmd_ustat_var(gen_vs_gen, self._ref_vs_ref, gen_vs_ref)
            return MMDInterval(ustat=mmd, std=np.sqrt(var))
        return mmd_from_gram(
            gen_vs_gen, self._ref_vs_ref, gen_vs_ref, variant=self._variant
        )


class MaxDescriptorMMD2(DescriptorMMD2):
    def compute(
        self, generated_graphs: Iterable[nx.Graph]
    ) -> Tuple[float, DescriptorKernel]:
        multi_kernel_result = super().compute(generated_graphs)
        assert multi_kernel_result.ndim == 1
        idx = np.argmax(multi_kernel_result)
        return multi_kernel_result[idx], self._kernel.get_subkernel(idx)


class OrbitMM2(DescriptorMMD2):
    def __init__(self, reference_graphs: AbstractDataset):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor_fn=OrbitCounts(),
            kernel=GaussianTV(bw=80),
        )


class ClusteringMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: AbstractDataset):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor_fn=ClusteringHistogram(bins=100),
            kernel=GaussianTV(bw=1.0 / 10),
        )
