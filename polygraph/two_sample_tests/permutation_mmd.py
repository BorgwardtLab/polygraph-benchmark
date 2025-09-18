from typing import Collection, Literal, Generic

import numpy as np

from polygraph import GraphType
from polygraph.utils.kernels import DescriptorKernel, GramBlocks
from polygraph.utils.mmd_utils import (
    full_gram_from_blocks,
    mmd_from_full_gram,
    mmd_from_gram,
)

__all__ = ["BootStrapMMDTest", "BootStrapMaxMMDTest"]


class _BootStrapTestBase(Generic[GraphType]):
    _variant: Literal["biased", "umve", "ustat"]

    def __init__(
        self,
        reference_graphs: Collection[GraphType],
        kernel: DescriptorKernel[GraphType],
        variant: Literal["biased", "umve", "ustat"] = "ustat",
    ):
        self._kernel = kernel
        self._reference_descriptions = self._kernel.featurize(reference_graphs)
        self._num_ref_graphs = len(reference_graphs)
        self._variant = variant

    def _sample_from_null_distribution(
        self,
        pre_gram_matrix: np.ndarray,
        n_samples: int,
        seed: int = 42,
    ) -> np.ndarray:
        assert pre_gram_matrix.shape[0] == pre_gram_matrix.shape[1]
        rng = np.random.default_rng(seed)
        n_generated = pre_gram_matrix.shape[0] - self._num_ref_graphs
        mmd_samples = []
        permutation = np.arange(n_generated + self._num_ref_graphs)
        for _ in range(n_samples):
            rng.shuffle(permutation)
            if self._kernel.is_adaptive:
                # Bandwith may change depending on the data split
                pre_gram_matrix = pre_gram_matrix[permutation, :]
                pre_gram_matrix = pre_gram_matrix[:, permutation]

                kx = pre_gram_matrix[:n_generated, :n_generated]
                ky = pre_gram_matrix[n_generated:, n_generated:]
                kxy = pre_gram_matrix[:n_generated, n_generated:]
                kx, kxy, ky = self._kernel.adapt(GramBlocks(kx, kxy, ky))
                mmd_samples.append(mmd_from_gram(kx, ky, kxy, self._variant))
            else:
                # Gram matrix is fixed
                x_idx = permutation[:n_generated]
                y_idx = permutation[n_generated:]
                mmd_samples.append(
                    mmd_from_full_gram(
                        pre_gram_matrix, x_idx, y_idx, self._variant
                    )
                )

        mmd_samples = np.array(mmd_samples)
        return mmd_samples

    def _get_realized_and_samples(
        self, generated_graphs: Collection[GraphType], num_samples: int = 1000
    ):
        descriptions = self._kernel.featurize(generated_graphs)

        pre_ref_vs_ref, pre_ref_vs_gen, pre_gen_vs_gen = self._kernel.pre_gram(
            self._reference_descriptions, descriptions
        )

        ref_vs_ref, ref_vs_gen, gen_vs_gen = self._kernel.adapt(
            GramBlocks(pre_ref_vs_ref, pre_ref_vs_gen, pre_gen_vs_gen)
        )
        realized_mmd = mmd_from_gram(
            ref_vs_ref, gen_vs_gen, ref_vs_gen, self._variant
        )

        full_pre_matrix = full_gram_from_blocks(
            pre_ref_vs_ref, pre_ref_vs_gen, pre_gen_vs_gen
        )
        mmd_samples = self._sample_from_null_distribution(
            full_pre_matrix, n_samples=num_samples, seed=42
        )
        assert len(mmd_samples) == num_samples

        return realized_mmd, mmd_samples


class BootStrapMMDTest(_BootStrapTestBase[GraphType], Generic[GraphType]):
    def compute(
        self,
        generated_graphs: Collection[GraphType],
        num_samples: int = 1000,
    ):
        if self._kernel.num_kernels != 1:
            raise ValueError(
                f"{self.__class__.__name__} requires kernel with `num_kernels == 1`."
            )

        realized_mmd, mmd_samples = self._get_realized_and_samples(
            generated_graphs, num_samples
        )
        return np.sum(mmd_samples >= realized_mmd, axis=0) / len(mmd_samples)


class BootStrapMaxMMDTest(_BootStrapTestBase[GraphType], Generic[GraphType]):
    def compute(
        self, generated_graphs: Collection[GraphType], num_samples: int = 1000
    ):
        if self._kernel.num_kernels == 1:
            raise ValueError(
                f"{self.__class__.__name__} requires kernel with `num_kernels > 1`."
            )

        realized_mmd, mmd_samples = self._get_realized_and_samples(
            generated_graphs, num_samples
        )
        assert isinstance(realized_mmd, np.ndarray) and isinstance(
            mmd_samples, np.ndarray
        )
        assert realized_mmd.ndim == 1 and mmd_samples.ndim == 2
        realized_mmd = np.max(realized_mmd)
        mmd_samples = np.max(mmd_samples, axis=1)
        return np.sum(mmd_samples >= realized_mmd, axis=0) / len(mmd_samples)
