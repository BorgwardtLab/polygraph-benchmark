from typing import Callable, Iterable, Literal

import networkx as nx
import numpy as np
from scipy import stats

from graph_gen_gym.datasets.abstract_dataset import AbstractDataset
from graph_gen_gym.metrics.mmd.kernels import DescriptorKernel
from graph_gen_gym.metrics.mmd.utils import (
    _get_batch_description,
    _pad_arrays,
    mmd_from_gram,
    mmd_ustat_var,
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
        n = len(gen_desc)
        if len(gen_desc) != len(ref_desc):
            raise ValueError
        agg_desc = np.concatenate([gen_desc, ref_desc], axis=0)
        full_gram_matrix = self._kernel(agg_desc, agg_desc)
        realized_mmd = mmd_from_gram(
            full_gram_matrix[:n, :n],
            full_gram_matrix[n:, n:],
            full_gram_matrix[:n, n:],
            variant="ustat",
        )
        mmd_samples = _sample_from_null_distribution(
            full_gram_matrix, n_samples=num_samples, variant="ustat"
        )

        q = stats.percentileofscore(mmd_samples, realized_mmd, "strict") / 100
        return 1 - q


class MaximallyPowerfulMMDTest:
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
        confidence_interval: float = 0.95,
    ):
        descriptions_genval = _get_batch_description(
            generated_val, self._descriptor_fn, self._zero_padding
        )
        desc1, desc2 = _pad_arrays(
            descriptions_genval, self._val_descriptions, self._zero_padding
        )

        # We first find the optimal kernel to maximize test power
        assert len(desc1) == len(desc2)
        n = len(desc1)
        agg_val_desc = np.concatenate([desc1, desc2], axis=0)
        full_val_gram = self._kernel(agg_val_desc, agg_val_desc)
        assert full_val_gram.ndim == 3
        mmd_samples = _sample_from_null_distribution(full_val_gram, 1000, "ustat")
        assert mmd_samples.ndim == 2
        threshold = np.quantile(mmd_samples, confidence_interval, axis=0)
        assert threshold.ndim == 1 and len(threshold) == mmd_samples.shape[1]
        realized_mmd2 = mmd_from_gram(
            full_val_gram[:n, :n],
            full_val_gram[n:, n:],
            full_val_gram[:n, n:],
            variant="ustat",
        )
        realized_variance = mmd_ustat_var(
            full_val_gram[:n, :n], full_val_gram[n:, n:], full_val_gram[:n, n:]
        )
        objective = realized_mmd2 / np.sqrt(realized_variance) - threshold / (
            n * np.sqrt(realized_variance)
        )
        assert objective.ndim == 1
        optimal_kernel_idx = np.argmax(objective)

        # Now, we can compute a p-value with the optimal kernel
        descriptions_gentest = _get_batch_description(
            generated_test, self._descriptor_fn, self._zero_padding
        )
        desc1, desc2 = _pad_arrays(
            descriptions_gentest, self._test_descriptions, self._zero_padding
        )
        assert len(desc1) == len(desc2)
        n = len(desc1)
        agg_test_desc = np.concatenate([desc1, desc2], axis=0)
        full_test_gram = self._kernel(agg_test_desc, agg_test_desc)
        assert full_test_gram.ndim == 3
        full_test_gram = full_test_gram[..., optimal_kernel_idx]
        mmd_samples = _sample_from_null_distribution(full_test_gram, 1000, "ustat")
        assert mmd_samples.ndim == 1, mmd_samples.shape
        realized_mmd2 = mmd_from_gram(
            full_test_gram[:n, :n],
            full_test_gram[n:, n:],
            full_test_gram[:n, n:],
            variant="ustat",
        )
        print(realized_mmd2)
        q = stats.percentileofscore(mmd_samples, realized_mmd2, "strict") / 100
        return 1 - q
