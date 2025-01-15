from typing import Collection, Literal

import networkx as nx
import numpy as np

from graph_gen_gym.utils.kernels import DescriptorKernel, GramBlocks
from graph_gen_gym.utils.mmd_utils import full_gram_from_blocks, mmd_from_gram

__all__ = ["BootStrapMMDTest", "BootStrapMaxMMDTest"]


class _BootStrapTestBase:
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        kernel: DescriptorKernel,
    ):
        self._kernel = kernel
        self._reference_descriptions = self._kernel.featurize(reference_graphs)
        self._num_graphs = len(reference_graphs)

    def _sample_from_null_distribution(
        self,
        pre_gram_matrix: np.ndarray,
        n_samples: int,
        variant: Literal["biased", "umve", "ustat"] = "ustat",
        seed: int = 42,
    ) -> np.ndarray:
        assert (
            pre_gram_matrix.shape[0] == pre_gram_matrix.shape[1]
            and pre_gram_matrix.shape[0] % 2 == 0
        )
        rng = np.random.default_rng(seed)
        n = pre_gram_matrix.shape[0] // 2
        mmd_samples = []
        permutation = np.arange(2 * n)
        for _ in range(n_samples):
            rng.shuffle(permutation)
            pre_gram_matrix = pre_gram_matrix[permutation, :]
            pre_gram_matrix = pre_gram_matrix[:, permutation]
            kx = pre_gram_matrix[:n, :n]
            ky = pre_gram_matrix[n:, n:]
            kxy = pre_gram_matrix[:n, n:]
            kx, kxy, ky = self._kernel.adapt(GramBlocks(kx, kxy, ky))
            mmd_samples.append(mmd_from_gram(kx, ky, kxy, variant))
        mmd_samples = np.array(mmd_samples)
        return mmd_samples

    def _get_realized_and_samples(
        self, generated_graphs: Collection[nx.Graph], num_samples: int = 1000
    ):
        assert len(generated_graphs) == self._num_graphs
        descriptions = self._kernel.featurize(generated_graphs)

        pre_ref_vs_ref, pre_ref_vs_gen, pre_gen_vs_gen = self._kernel.pre_gram(
            self._reference_descriptions, descriptions
        )

        ref_vs_ref, ref_vs_gen, gen_vs_gen = self._kernel.adapt(
            GramBlocks(pre_ref_vs_ref, pre_ref_vs_gen, pre_gen_vs_gen)
        )
        realized_mmd = mmd_from_gram(
            ref_vs_ref, gen_vs_gen, ref_vs_gen, variant="ustat"
        )

        full_pre_matrix = full_gram_from_blocks(
            pre_ref_vs_ref, pre_ref_vs_gen, pre_gen_vs_gen
        )
        mmd_samples = self._sample_from_null_distribution(
            full_pre_matrix, n_samples=num_samples, variant="ustat", seed=42
        )
        assert len(mmd_samples) == num_samples

        return realized_mmd, mmd_samples


class BootStrapMMDTest(_BootStrapTestBase):
    def compute(self, generated_graphs: Collection[nx.Graph], num_samples: int = 1000):
        realized_mmd, mmd_samples = self._get_realized_and_samples(
            generated_graphs, num_samples
        )
        return np.sum(mmd_samples >= realized_mmd, axis=0) / len(mmd_samples)


class BootStrapMaxMMDTest(_BootStrapTestBase):
    def compute(self, generated_graphs: Collection[nx.Graph], num_samples: int = 1000):
        if self._kernel.num_kernels == 1:
            raise ValueError(f"{self.__name__} requires kernel with `num_kernels > 1`.")

        realized_mmd, mmd_samples = self._get_realized_and_samples(
            generated_graphs, num_samples
        )
        assert realized_mmd.ndim == 1 and mmd_samples.ndim == 2
        realized_mmd = np.max(realized_mmd)
        mmd_samples = np.max(mmd_samples, axis=1)
        return np.sum(mmd_samples >= realized_mmd, axis=0) / len(mmd_samples)
