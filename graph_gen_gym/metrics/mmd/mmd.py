from collections import namedtuple
from functools import partial
from typing import Callable, Iterable, Literal

import networkx as nx
import numpy as np

from graph_gen_gym.datasets.abstract_dataset import AbstractDataset
from graph_gen_gym.metrics.mmd.graph_descriptors import (
    clustering_descriptor,
    degree_descriptor,
    orbit_descriptor,
    spectral_descriptor,
)
from graph_gen_gym.metrics.mmd.kernels import DescriptorKernel, GaussianTV
from graph_gen_gym.metrics.mmd.utils import (
    _get_batch_description,
    _pad_arrays,
    mmd_from_gram,
    mmd_ustat_var,
)

MMDInterval = namedtuple("MMDInterval", ["ustat", "std"])


class DescriptorMMD2:
    def __init__(
        self,
        reference_graphs: AbstractDataset,
        descriptor_fn: Callable[[nx.Graph], np.ndarray],
        kernel: DescriptorKernel,
        variant: Literal["biased", "umve", "ustat", "ustat-var"] = "biased",
        zero_padding: bool = False,
    ):
        self._descriptor_fn = descriptor_fn
        self._kernel = kernel
        self._variant = variant
        self._zero_padding = zero_padding
        self._reference_descriptions = _get_batch_description(
            reference_graphs.to_nx(), self._descriptor_fn, self._zero_padding
        )
        assert self._reference_descriptions.ndim == 2 and len(
            self._reference_descriptions
        ) == len(reference_graphs)
        self._ref_vs_ref = self._gram_matrix(
            self._reference_descriptions, self._reference_descriptions
        )

    def _gram_matrix(self, descriptors1, descriptors2):
        descriptors1, descriptors2 = _pad_arrays(
            descriptors1, descriptors2, self._zero_padding
        )
        return self._kernel(descriptors1, descriptors2)

    def compute(self, generated_graphs: Iterable[nx.Graph]):
        descriptions = _get_batch_description(
            generated_graphs, self._descriptor_fn, self._zero_padding
        )
        gen_vs_gen = self._gram_matrix(descriptions, descriptions)
        gen_vs_ref = self._gram_matrix(descriptions, self._reference_descriptions)
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
    def compute(self, generated_graphs: Iterable[nx.Graph]):
        multi_kernel_result = super().compute(generated_graphs)
        assert multi_kernel_result.ndim == 1
        return np.max(multi_kernel_result)


class DegreeMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: AbstractDataset):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor_fn=degree_descriptor,
            kernel=GaussianTV(bw=1.0),
            zero_padding=True,
        )


class SpectralMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: AbstractDataset):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor_fn=spectral_descriptor,
            kernel=GaussianTV(bw=1.0),
        )


class OrbitMM2(DescriptorMMD2):
    def __init__(self, reference_graphs: AbstractDataset):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor_fn=orbit_descriptor,
            kernel=GaussianTV(bw=80),
        )


class ClusteringMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: AbstractDataset):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor_fn=partial(clustering_descriptor, bins=100),
            kernel=GaussianTV(bw=1.0 / 10),
        )
