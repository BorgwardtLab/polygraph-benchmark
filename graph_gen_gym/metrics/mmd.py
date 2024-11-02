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
from graph_gen_gym.metrics.kernels import DescriptorKernel, GaussianTV


def _get_batch_description(
    graphs: Iterable[nx.Graph],
    descriptor_fn: Callable[[nx.Graph], np.ndarray],
    zero_padding: bool,
) -> np.ndarray:
    descriptions = [descriptor_fn(graph) for graph in graphs]
    if zero_padding:
        max_length = max(len(descr) for descr in descriptions)
        descriptions = [
            np.concatenate((descr, np.zeros(max_length - len(descr))))
            for descr in descriptions
        ]
    return np.stack(descriptions)


def _pad_arrays(
    x: np.ndarray, y: np.ndarray, zero_padding: bool
) -> Tuple[np.ndarray, np.ndarray]:
    assert x.ndim == 2 and y.ndim == 2
    if x.shape[1] == y.shape[1]:
        return x, y
    if zero_padding:
        max_length = max(x.shape[1], y.shape[1])
        x = np.concatenate(
            (
                x,
                np.zeros((x.shape[0], max_length - x.shape[1])),
            ),
            axis=1,
        )
        y = np.concatenate(
            (
                y,
                np.zeros((y.shape[0], max_length - y.shape[1])),
            ),
            axis=1,
        )
        return x, y
    raise ValueError(
        "Dimensions of descriptors does not match but `zero_padding` was not set to `True`."
    )


def mmd_from_gram(kxx, kyy, kxy, variant: Literal["biased", "unbiased", "ustat"]):
    assert kxx.shape[0] == kxx.shape[1] and kyy.shape[0] == kyy.shape[1]
    n, m = kxx.shape[0], kyy.shape[1]
    assert kxy.shape[:2] == (n, m)

    if variant == "biased":
        xvx = kxx.sum(axis=(0, 1)) / (n**2)
        yvy = kyy.sum(axis=(0, 1)) / (m**2)
        xvy = kxy.sum(axis=(0, 1)) / (n * m)
    elif variant in ["unbiased", "ustat"]:
        xvx = (kxx.sum(axis=(0, 1)) - np.trace(kxx, axis1=0, axis2=1)) / (n * (n - 1))
        yvy = (kyy.sum(axis=(0, 1)) - np.trace(kyy, axis1=0, axis2=1)) / (m * (m - 1))
        if variant == "ustat":
            if n != m:
                raise RuntimeError
            xvy = (kxy.sum(axis=(0, 1)) - np.trace(kxy, axis1=0, axis2=1)) / (
                n * (n - 1)
            )
        else:
            xvy = kxy.sum(axis=(0, 1)) / (n * m)
    else:
        raise ValueError

    return xvx + yvy - 2 * xvy


class DescriptorMMD:
    def __init__(
        self,
        reference_graphs: AbstractDataset,
        descriptor_fn: Callable[[nx.Graph], np.ndarray],
        kernel: DescriptorKernel,
        variant: Literal["biased", "unbiased", "ustat"] = "biased",
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
        descriptions = self._get_batch_description(generated_graphs)
        gen_vs_gen = self._gram_matrix(descriptions, descriptions)
        gen_vs_ref = self._gram_matrix(descriptions, self._reference_descriptions)
        return mmd_from_gram(
            gen_vs_gen, self._ref_vs_ref, gen_vs_ref, variant=self._variant
        )


class BootStrapMMDTest(DescriptorMMD):
    def __init__(
        self,
        reference_graphs: AbstractDataset,
        descriptor_fn: Callable[[nx.Graph], np.ndarray],
        kernel: DescriptorKernel,
        zero_padding: bool = False,
    ):
        self._descriptor_fn = descriptor_fn
        self._kernel = kernel
        self._zero_padding = zero_padding
        self._reference_descriptions = _get_batch_description(
            reference_graphs.to_nx(), self._descriptor_fn, self._zero_padding
        )
        assert self._reference_descriptions.ndim == 2 and len(
            self._reference_descriptions
        ) == len(reference_graphs)

    def _mmd_from_full(self, gram_matrix):
        assert len(gram_matrix) % 2 == 0
        n = len(gram_matrix) // 2
        kx = gram_matrix[:n, :n]
        ky = gram_matrix[n:, n:]
        kxy = gram_matrix[:n, n:]
        return mmd_from_gram(kx, ky, kxy, "ustat")

    def compute(self, generated_graphs: Iterable[nx.Graph], num_samples: int = 1000):
        descriptions = _get_batch_description(
            generated_graphs, self._descriptor_fn, self._zero_padding
        )
        gen_desc, ref_desc = _pad_arrays(
            descriptions, self._reference_descriptions, self._zero_padding
        )
        n = len(gen_desc)
        if len(gen_desc) != len(ref_desc):
            raise ValueError
        agg_desc = np.concatenate([gen_desc, ref_desc], axis=0)
        full_gram_matrix = self._kernel(agg_desc, agg_desc)
        realized_mmd = self._mmd_from_full(full_gram_matrix)

        mmd_samples = []
        permutation = np.arange(2 * n)
        for _ in range(num_samples):
            np.random.shuffle(permutation)
            full_gram_matrix = full_gram_matrix[permutation, :]
            full_gram_matrix = full_gram_matrix[:, permutation]
            mmd_samples.append(self._mmd_from_full(full_gram_matrix))
        mmd_samples = np.array(mmd_samples)
        q = stats.percentileofscore(mmd_samples, realized_mmd, "strict") / 100
        return 1 - q


class SpectreDegMMD(DescriptorMMD):
    def __init__(self, reference_graphs: AbstractDataset):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor_fn=degree_descriptor,
            kernel=GaussianTV(bw=1.0),
            zero_padding=True,
        )


class SpectreSpectralMMD(DescriptorMMD):
    def __init__(self, reference_graphs: AbstractDataset):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor_fn=spectral_descriptor,
            kernel=GaussianTV(bw=1.0),
        )


class SpectreOrbitMMD(DescriptorMMD):
    def __init__(self, reference_graphs: AbstractDataset):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor_fn=orbit_descriptor,
            kernel=GaussianTV(bw=80),
        )


class SpectreClusteringMMD(DescriptorMMD):
    def __init__(self, reference_graphs: AbstractDataset):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor_fn=partial(clustering_descriptor, bins=100),
            kernel=GaussianTV(bw=1.0 / 10),
        )
