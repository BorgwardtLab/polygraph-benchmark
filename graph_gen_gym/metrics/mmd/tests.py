from typing import Callable, Iterable, Literal

import networkx as nx
import numpy as np

from graph_gen_gym.datasets.abstract_dataset import AbstractDataset
from graph_gen_gym.metrics.mmd.kernels import DescriptorKernel
from graph_gen_gym.metrics.mmd.utils import (
    _get_batch_description,
    _pad_arrays,
    mmd_from_gram,
)


def _sample_from_null_distribution(
    full_gram_matrix: np.ndarray,
    n_samples: int,
    variant: Literal["biased", "umve", "ustat"] = "ustat",
) -> np.ndarray:
    assert (
        full_gram_matrix.shape[0] == full_gram_matrix.shape[1]
        and full_gram_matrix.shape[0] % 2 == 0
    )
    n = full_gram_matrix.shape[0] // 2
    mmd_samples = []
    permutation = np.arange(2 * n)
    for _ in range(n_samples):
        np.random.shuffle(permutation)
        full_gram_matrix = full_gram_matrix[permutation, :]
        full_gram_matrix = full_gram_matrix[:, permutation]
        kx = full_gram_matrix[:n, :n]
        ky = full_gram_matrix[n:, n:]
        kxy = full_gram_matrix[:n, n:]
        mmd_samples.append(mmd_from_gram(kx, ky, kxy, variant))
    mmd_samples = np.array(mmd_samples)
    return mmd_samples


def _realized_mmd_and_samples(
    desc1: np.ndarray, desc2: np.ndarray, kernel: Callable, num_samples: int
):
    assert len(desc1) == len(desc2)
    n = len(desc1)
    agg_desc = np.concatenate([desc1, desc2], axis=0)
    full_val_gram = kernel(agg_desc, agg_desc)
    mmd_samples = _sample_from_null_distribution(full_val_gram, num_samples, "ustat")
    realized_mmd2 = mmd_from_gram(
        full_val_gram[:n, :n],
        full_val_gram[n:, n:],
        full_val_gram[:n, n:],
        variant="ustat",
    )
    return realized_mmd2, mmd_samples


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

    def compute(self, generated_graphs: Iterable[nx.Graph], num_samples: int = 1000):
        descriptions = _get_batch_description(
            generated_graphs, self._descriptor_fn, self._zero_padding
        )
        gen_desc, ref_desc = _pad_arrays(
            descriptions, self._reference_descriptions, self._zero_padding
        )
        realized_mmd, mmd_samples = _realized_mmd_and_samples(
            gen_desc, ref_desc, self._kernel, num_samples
        )
        assert realized_mmd.ndim == 0 and mmd_samples.ndim == 1

        q = np.sum(mmd_samples < realized_mmd) / len(mmd_samples)
        return 1 - q


class OptimizedPValue:
    def __init__(
        self,
        val_graphs: AbstractDataset,
        test_graphs: AbstractDataset,
        descriptor_fn: Callable[[nx.Graph], np.ndarray],
        kernel: DescriptorKernel,
        zero_padding: bool = False,
    ):
        self._descriptor_fn = descriptor_fn
        self._kernel = kernel
        self._zero_padding = zero_padding
        self._test_descriptions = _get_batch_description(
            test_graphs.to_nx(), self._descriptor_fn, self._zero_padding
        )
        self._val_descriptions = _get_batch_description(
            val_graphs.to_nx(), self._descriptor_fn, self._zero_padding
        )

    def compute(
        self,
        generated_val: Iterable[nx.Graph],
        generated_test: Iterable[nx.Graph],
        num_bootstrap_samples: int = 500,
    ):
        descriptions_genval = _get_batch_description(
            generated_val, self._descriptor_fn, self._zero_padding
        )
        desc1, desc2 = _pad_arrays(
            descriptions_genval, self._val_descriptions, self._zero_padding
        )
        realized_mmd2, mmd_samples = _realized_mmd_and_samples(
            desc1, desc2, kernel=self._kernel, num_samples=num_bootstrap_samples
        )
        assert (
            mmd_samples.ndim == 2
            and realized_mmd2.ndim == 1
            and realized_mmd2.shape[0] == mmd_samples.shape[1]
        )
        q = np.sum(mmd_samples < np.expand_dims(realized_mmd2, axis=0), axis=0) / len(
            mmd_samples
        )
        optimal_kernel_idx = np.argmax(q)

        # Now, we can compute a p-value with the optimal kernel
        descriptions_gentest = _get_batch_description(
            generated_test, self._descriptor_fn, self._zero_padding
        )
        desc1, desc2 = _pad_arrays(
            descriptions_gentest, self._test_descriptions, self._zero_padding
        )
        realized_mmd2, mmd_samples = _realized_mmd_and_samples(
            desc1,
            desc2,
            kernel=lambda a, b: self._kernel(a, b)[..., optimal_kernel_idx],
            num_samples=num_bootstrap_samples,
        )
        assert mmd_samples.ndim == 1 and realized_mmd2.ndim == 0
        q = np.sum(mmd_samples < realized_mmd2) / len(mmd_samples)
        return 1 - q
