from collections import namedtuple
from functools import partial
from typing import Callable, Iterable, Literal, Tuple, Union

import networkx as nx
import numpy as np
from scipy import stats

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
        zero_padding: bool = False,
    ):
        self._descriptor_fn = descriptor_fn
        self._kernel_param = kernel_param
        self._kernel = kernel
        self._zero_padding = zero_padding
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
        descriptions = [self._descriptor_fn(graph) for graph in graphs]
        if self._zero_padding:
            max_length = max(len(descr) for descr in descriptions)
            descriptions = [
                np.concatenate((descr, np.zeros(max_length - len(descr))))
                for descr in descriptions
            ]
        return np.stack(descriptions)

    def _pad_to_match(
        self, descriptors1, descriptors2
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert descriptors1.ndim == 2 and descriptors2.ndim == 2
        if descriptors1.shape[1] == descriptors2.shape[1]:
            return descriptors1, descriptors2
        if self._zero_padding:
            max_length = max(descriptors1.shape[1], descriptors2.shape[1])
            descriptors1 = np.concatenate(
                (
                    descriptors1,
                    np.zeros(
                        (descriptors1.shape[0], max_length - descriptors1.shape[1])
                    ),
                ),
                axis=1,
            )
            descriptors2 = np.concatenate(
                (
                    descriptors2,
                    np.zeros(
                        (descriptors2.shape[0], max_length - descriptors2.shape[1])
                    ),
                ),
                axis=1,
            )
            return descriptors1, descriptors2
        raise ValueError(
            "Dimensions of descriptors does not match but `zero_padding` was not set to `True`."
        )

    def _disc(self, descriptors1, descriptors2):
        descriptors1, descriptors2 = self._pad_to_match(descriptors1, descriptors2)
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


class BootStrapMMDTest(DescriptorMMD):
    """TODO: Here we are  performing the test with the biased MMD estimate, while the papers usually use the unbiased one. Are we allowed to do this?"""

    def _mmd(self, desc1, desc2):
        return (
            self._disc(desc1, desc1)
            + self._disc(desc2, desc2)
            - 2 * self._disc(desc1, desc2)
        )

    def compute(self, generated_graphs: Iterable[nx.Graph], num_samples: int = 1000):
        descriptions = self._get_batch_description(generated_graphs)
        gen_desc, ref_desc = self._pad_to_match(
            descriptions, self._reference_descriptions
        )
        n = len(gen_desc)
        if len(gen_desc) != len(ref_desc):
            raise ValueError
        realized_mmd = self._mmd(gen_desc, ref_desc)
        mmd_samples = []
        agg_desc = np.concatenate((gen_desc, ref_desc), axis=0)
        for _ in range(num_samples):
            np.random.shuffle(agg_desc)
            desc1, desc2 = agg_desc[:n], agg_desc[n:]
            assert len(desc1) == len(desc2)
            mmd_samples.append(self._mmd(desc1, desc2))
        mmd_samples = np.array(mmd_samples)
        q = stats.percentileofscore(mmd_samples, realized_mmd, "strict") / 100
        return 1 - q


class SpectreDegMMD(DescriptorMMD):
    def __init__(self, reference_graphs: AbstractDataset):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor_fn=degree_descriptor,
            kernel="gaussian_tv",
            kernel_param=1.0,
            zero_padding=True,
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
