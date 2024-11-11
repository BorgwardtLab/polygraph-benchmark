from typing import Collection, Literal

import networkx as nx
import numpy as np

from graph_gen_gym.metrics.mmd.kernels import DescriptorKernel
from graph_gen_gym.metrics.mmd.utils import mmd_from_gram


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
        reference_graphs: Collection[nx.Graph],
        kernel: DescriptorKernel,
    ):
        if kernel.num_kernels > 1:
            raise ValueError(
                "Can only perform hypothesis test with a single kernel at a time"
            )

        self._kernel = kernel
        self._reference_descriptions = self._kernel.featurize(reference_graphs)
        self._ref_vs_ref = self._kernel(
            self._reference_descriptions, self._reference_descriptions
        )
        self._num_graphs = len(reference_graphs)

    def compute(self, generated_graphs: Collection[nx.Graph], num_samples: int = 1000):
        assert len(generated_graphs) == self._num_graphs
        descriptions = self._kernel.featurize(generated_graphs)

        gen_vs_gen = self._kernel(descriptions, descriptions)
        ref_vs_gen = self._kernel(self._reference_descriptions, descriptions)
        full_gram_matrix = np.zeros((2 * self._num_graphs, 2 * self._num_graphs))

        full_gram_matrix[: self._num_graphs, : self._num_graphs] = self._ref_vs_ref
        full_gram_matrix[: self._num_graphs, self._num_graphs :] = ref_vs_gen
        full_gram_matrix[self._num_graphs :, : self._num_graphs] = np.swapaxes(
            ref_vs_gen, 0, 1
        )
        full_gram_matrix[self._num_graphs :, self._num_graphs :] = gen_vs_gen
        assert (full_gram_matrix == full_gram_matrix.T).all()

        realized_mmd = mmd_from_gram(
            self._ref_vs_ref, gen_vs_gen, ref_vs_gen, variant="ustat"
        )
        mmd_samples = _sample_from_null_distribution(
            full_gram_matrix, n_samples=num_samples, variant="ustat"
        )
        assert realized_mmd.ndim == 0 and mmd_samples.ndim == 1

        q = np.sum(mmd_samples < realized_mmd) / len(mmd_samples)
        return 1 - q
