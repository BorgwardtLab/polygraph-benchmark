from typing import Callable, Iterable

import networkx as nx
import numpy as np
from scipy import stats

from graph_gen_gym.datasets.abstract_dataset import AbstractDataset
from graph_gen_gym.metrics.mmd.kernels import DescriptorKernel
from graph_gen_gym.metrics.mmd.utils import (
    _get_batch_description,
    _pad_arrays,
    mmd_from_gram,
)


class BootStrapMMDTest:
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
