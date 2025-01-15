from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Collection, Literal, Union

import networkx as nx
import numpy as np

from graph_gen_gym.utils.kernels import DescriptorKernel, GramBlocks
from graph_gen_gym.utils.mmd_utils import mmd_from_gram


__all__ = [
    "DescriptorMMD2",
    "MaxDescriptorMMD2",
    "MMDInterval",
    "DescriptorMMD2Interval",
    "MaxDescriptorMMD2Interval",
]


MMDInterval = namedtuple("MMDInterval", ["mean", "std", "low", "high"])


class DescriptorMMD2:
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        kernel: DescriptorKernel,
        variant: Literal["biased", "umve", "ustat"] = "biased",
    ):
        self._kernel = kernel
        self._variant = variant
        self._reference_descriptions = self._kernel.featurize(reference_graphs)

    def compute(
        self, generated_graphs: Collection[nx.Graph]
    ) -> Union[float, np.ndarray]:
        descriptions = self._kernel.featurize(
            generated_graphs,
        )
        ref_vs_ref, ref_vs_gen, gen_vs_gen = self._kernel(
            self._reference_descriptions, descriptions
        )
        return mmd_from_gram(ref_vs_ref, gen_vs_gen, ref_vs_gen, variant=self._variant)


class MaxDescriptorMMD2(DescriptorMMD2):
    """
    Compute the maximal MMD across different kernel choices.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel.num_kernels > 1:
            raise ValueError(
                "Must provide several kernels, i.e. a kernel with multiple parameters"
            )

    def compute(self, generated_graphs: Collection[nx.Graph]) -> float:
        multi_kernel_result = super().compute(generated_graphs)
        idx = int(np.argmax(multi_kernel_result))
        return multi_kernel_result[idx]


class _DescriptorMMD2Interval(ABC):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        kernel: DescriptorKernel,
        variant: Literal["biased", "umve", "ustat"] = "biased",
    ):
        self._kernel = kernel
        self._variant = variant
        self._reference_descriptions = self._kernel.featurize(reference_graphs)

    def _generate_mmd_samples(
        self,
        generated_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
    ) -> MMDInterval:
        descriptions = self._kernel.featurize(
            generated_graphs,
        )
        mmd_samples = []
        rng = np.random.default_rng(42)

        ref_vs_ref, ref_vs_gen, gen_vs_gen = self._kernel.pre_gram(
            self._reference_descriptions, descriptions
        )

        for _ in range(num_samples):
            ref_idxs = rng.choice(len(ref_vs_ref), size=subsample_size, replace=False)
            gen_idxs = rng.choice(len(gen_vs_gen), size=subsample_size, replace=False)
            sub_ref_vs_ref = ref_vs_ref[ref_idxs][:, ref_idxs]
            sub_gen_vs_gen = gen_vs_gen[gen_idxs][:, gen_idxs]
            sub_ref_vs_gen = ref_vs_gen[ref_idxs][:, gen_idxs]
            sub_ref_vs_ref, sub_ref_vs_gen, sub_gen_vs_gen = self._kernel.adapt(
                GramBlocks(sub_ref_vs_ref, sub_ref_vs_gen, sub_gen_vs_gen)
            )
            mmd_samples.append(
                mmd_from_gram(
                    sub_ref_vs_ref,
                    sub_gen_vs_gen,
                    sub_ref_vs_gen,
                    variant=self._variant,
                )
            )

        mmd_samples = np.array(mmd_samples)
        return mmd_samples

    @abstractmethod
    def compute(*args, **kwargs) -> MMDInterval: ...


class DescriptorMMD2Interval(_DescriptorMMD2Interval):
    def compute(
        self,
        generated_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
        coverage: float = 0.95,
    ) -> MMDInterval:
        mmd_samples = self._generate_mmd_samples(
            generated_graphs=generated_graphs,
            subsample_size=subsample_size,
            num_samples=num_samples,
        )
        low, high = (
            np.quantile(mmd_samples, (1 - coverage) / 2, axis=0),
            np.quantile(mmd_samples, coverage + (1 - coverage) / 2, axis=0),
        )
        avg = np.mean(mmd_samples, axis=0)
        std = np.std(mmd_samples, axis=0)
        return MMDInterval(mean=avg, std=std, low=low, high=high)


class MaxDescriptorMMD2Interval(_DescriptorMMD2Interval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel.num_kernels > 1:
            raise ValueError(
                "Must provide several kernels, i.e. either a kernel with multiple parameters"
            )

    def compute(
        self,
        generated_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
        coverage: float = 0.95,
    ) -> MMDInterval:
        mmd_samples = self._generate_mmd_samples(
            generated_graphs=generated_graphs,
            subsample_size=subsample_size,
            num_samples=num_samples,
        )
        assert mmd_samples.ndim == 2
        mmd_samples = np.max(mmd_samples, axis=1)
        low, high = (
            np.quantile(mmd_samples, (1 - coverage) / 2, axis=0),
            np.quantile(mmd_samples, coverage + (1 - coverage) / 2, axis=0),
        )
        avg = np.mean(mmd_samples, axis=0)
        std = np.std(mmd_samples, axis=0)
        return MMDInterval(mean=avg, std=std, low=low, high=high)
