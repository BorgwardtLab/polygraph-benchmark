from functools import partial
from typing import Callable, Iterable, Literal, Union

import networkx as nx
import numpy as np

from graph_gen_gym.datasets.abstract_dataset import AbstractDataset
from graph_gen_gym.metrics.graph_descriptors import (
    clustering_descriptor,
    degree_descriptor,
    orbit_descriptor,
    spectral_descriptor,
)


class DescriptorMMD:
    def __init__(
        self,
        reference_graphs: AbstractDataset,
        descriptor_fn: Callable[[nx.Graph], np.ndarray],
        kernel: Literal["gaussian_tv", "laplace"] = "gaussian_tv",
        kernel_param: Union[float, np.ndarray] = 1.0,
    ):
        self._descriptor_fn = descriptor_fn
        self._kernel_param = kernel_param
        self._kernel = kernel
        self._reference_descriptions = self._get_batch_description(
            reference_graphs.to_nx()
        )
        assert self._reference_descriptions.ndim == 2 and len(
            self._reference_descriptions
        ) == len(reference_graphs)
        self._reference_vs_reference = self._disc(
            self._reference_descriptions, self._reference_descriptions
        )

    def _get_batch_description(self, graphs: Iterable[nx.Graph]) -> np.ndarray:
        return np.stack([self._descriptor_fn(graph) for graph in graphs])

    def _disc(self, descriptors1, descriptors2):
        comparison = np.expand_dims(descriptors1, 1) - np.expand_dims(descriptors2, 0)

        if isinstance(self._kernel_param, np.ndarray):
            if self._kernel_param.ndim != 1:
                raise ValueError(
                    f"The parameter `kernel_param` parameter must be a scalar or 1-dimensional. Got {self._kernel_param.ndim} dimensions."
                )
            comparison = np.expand_dims(comparison, -1)

        comparison = np.abs(comparison).sum(axis=2)
        if self._kernel == "gaussian_tv":
            comparison = np.exp(
                -((comparison / 2) ** 2) / (2 * self._kernel_param**2)
            )
        elif self._kernel == "laplace":
            comparison = np.exp(-self._kernel_param * comparison)
        else:
            raise ValueError(
                f"Kernel '{self._kernel}' is invalid, expected 'gaussian_tv' or 'laplace'."
            )

        assert comparison.shape[:2] == (len(descriptors1), len(descriptors2))
        return comparison.sum(axis=(0, 1)) / (len(descriptors1) * len(descriptors2))

    def compute(self, generated_graphs: Iterable[nx.Graph]):
        descriptions = self._get_batch_description(generated_graphs)
        generated_vs_reference = self._disc(descriptions, self._reference_descriptions)
        generated_vs_generated = self._disc(descriptions, descriptions)
        return (
            generated_vs_generated
            + self._reference_vs_reference
            - 2 * generated_vs_reference
        )


class SpectreDegMMD(DescriptorMMD):
    def __init__(self, reference_graphs: AbstractDataset):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor_fn=degree_descriptor,
            kernel="gaussian_tv",
            kernel_param=1.0,
        )


class SpectreSpectralMMD(DescriptorMMD):
    def __init__(self, reference_graphs: AbstractDataset):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor_fn=spectral_descriptor,
            kernel="gaussian_tv",
            kernel_param=1.0,
        )


class SpectreOrbitMMD(DescriptorMMD):
    def __init__(self, reference_graphs: AbstractDataset):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor_fn=orbit_descriptor,
            kernel="gaussian_tv",
            kernel_param=80,
        )


class SpectreClusteringMMD(DescriptorMMD):
    def __init__(self, reference_graphs: AbstractDataset):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor_fn=partial(clustering_descriptor, bins=100),
            kernel="gaussian_tv",
            kernel_param=1.0 / 10,
        )
