from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Callable, Iterable, List, Union

import networkx as nx
import numpy as np
from scipy.sparse import csr_array

GraphDescriptorFn = Callable[[Iterable[nx.Graph]], np.ndarray]

GramBlocks = namedtuple("GramBlocks", ["ref_vs_ref", "ref_vs_gen", "gen_vs_gen"])


class DescriptorKernel(ABC):
    def __init__(self, descriptor_fn: GraphDescriptorFn):
        self._descriptor_fn = descriptor_fn

    @abstractmethod
    def gram(self, x: Any, y: Any) -> np.ndarray:
        ...

    @abstractmethod
    def get_subkernel(self, idx: int) -> "DescriptorKernel":
        ...

    @property
    @abstractmethod
    def num_kernels(self) -> int:
        ...

    def featurize(self, graphs: Iterable[nx.Graph]) -> Any:
        return self._descriptor_fn(graphs)

    def __call__(self, ref: Any, gen: Any) -> GramBlocks:
        ref_vs_ref, ref_vs_gen, gen_vs_gen = (
            self.gram(ref, ref),
            self.gram(ref, gen),
            self.gram(gen, gen),
        )
        assert ref_vs_ref.ndim == ref_vs_gen.ndim and ref_vs_ref.ndim == gen_vs_gen.ndim
        if self.num_kernels > 1:
            assert ref_vs_ref.ndim == 3 and ref_vs_ref.shape[2] == self.num_kernels
        else:
            assert ref_vs_ref.ndim == 2
        return GramBlocks(ref_vs_ref, ref_vs_gen, gen_vs_gen)


class StackedKernel(DescriptorKernel):
    def __init__(self, kernels: List[DescriptorKernel]):
        self._kernels = kernels
        self._kernel_count = np.array([kernel.num_kernels for kernel in self._kernels])
        self._num_kernels = np.sum(self._kernel_count)
        self._cumsum_num_kernels = np.cumsum(self._kernel_count) - self._kernel_count

    def featurize(self, graphs: Iterable[nx.Graph]) -> List:
        return [kernel.featurize(graphs) for kernel in self._kernels]

    def __call__(self, ref: List, gen: List) -> GramBlocks:
        assert len(ref) == len(gen) and len(ref) == len(self._kernels)
        results: LinearKernel[GramBlocks] = [
            kernel(feat_ref, feat_gen)
            for kernel, feat_ref, feat_gen in zip(self._kernels, ref, gen)
        ]
        results = [
            [
                np.expand_dims(block, axis=-1) if block.ndim == 2 else block
                for block in result
            ]
            for result in results
        ]
        concatenated_blocks = [
            np.concatenate(blocks, axis=-1) for blocks in zip(*results)
        ]
        return GramBlocks(*concatenated_blocks)

    def gram(self, x: Any, y: Any) -> Any:
        raise NotImplementedError

    @property
    def num_kernels(self) -> int:
        return self._num_kernels

    def get_subkernel(self, idx: int) -> DescriptorKernel:
        idx1 = np.searchsorted(self._cumsum_num_kernels, idx, side="left") - 1
        idx2 = idx - self._cumsum_num_kernels[idx1]
        return self._kernels[idx1].get_subkernel(idx2)


class LaplaceKernel(DescriptorKernel):
    def __init__(self, descriptor_fn: GraphDescriptorFn, lbd: Union[float, np.ndarray]):
        super().__init__(descriptor_fn)
        self.lbd = lbd

    def get_subkernel(self, idx: int) -> DescriptorKernel:
        assert isinstance(self.lbd, np.ndarray)
        return LaplaceKernel(self._descriptor_fn, self.lbd[idx])

    @property
    def num_kernels(self) -> int:
        if isinstance(self.lbd, np.ndarray):
            assert self.lbd.ndim == 1
            return self.lbd.size
        return 1

    def gram(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert x.ndim == 2 and y.ndim == 2

        comparison = np.expand_dims(x, 1) - np.expand_dims(y, 0)

        if isinstance(self.lbd, np.ndarray):
            if self.lbd.ndim != 1:
                raise ValueError(
                    f"The parameter `kernel_param` parameter must be a scalar or 1-dimensional. Got {self.lbd.ndim} dimensions."
                )
            comparison = np.expand_dims(comparison, -1)

        comparison = np.abs(comparison).sum(axis=2)
        comparison = np.exp(-self.lbd * comparison)
        assert comparison.shape[:2] == (len(x), len(y))
        return comparison


class GaussianTV(DescriptorKernel):
    def __init__(self, descriptor_fn: GraphDescriptorFn, bw: Union[float, np.ndarray]):
        super().__init__(descriptor_fn)
        self.bw = bw

    def get_subkernel(self, idx: int) -> DescriptorKernel:
        assert isinstance(self.bw, np.ndarray)
        return GaussianTV(self._descriptor_fn, self.bw[idx])

    @property
    def num_kernels(self) -> int:
        if isinstance(self.bw, np.ndarray):
            assert self.bw.ndim == 1
            return self.bw.size
        return 1

    def gram(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert x.ndim == 2 and y.ndim == 2

        comparison = np.expand_dims(x, 1) - np.expand_dims(y, 0)

        if isinstance(self.bw, np.ndarray):
            if self.bw.ndim != 1:
                raise ValueError(
                    f"The parameter `kernel_param` parameter must be a scalar or 1-dimensional. Got {self.bw.ndim} dimensions."
                )
            comparison = np.expand_dims(comparison, -1)

        comparison = np.abs(comparison).sum(axis=2)
        comparison = np.exp(-((comparison / 2) ** 2) / (2 * self.bw**2))
        assert comparison.shape[:2] == (len(x), len(y))
        return comparison


class RBFKernel(DescriptorKernel):
    def __init__(
        self, descriptor_fn: GraphDescriptorFn, bw: Union[float, np.ndarray]
    ) -> None:
        super().__init__(descriptor_fn)
        self.bw = bw

    def get_subkernel(self, idx: int) -> DescriptorKernel:
        assert isinstance(self.bw, np.ndarray)
        return RBFKernel(self._descriptor_fn, self.bw[idx])

    @property
    def num_kernels(self) -> int:
        if isinstance(self.bw, np.ndarray):
            assert self.bw.ndim == 1
            return self.bw.size
        return 1

    def gram(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert x.ndim == 2 and y.ndim == 2

        comparison = np.expand_dims(x, 1) - np.expand_dims(y, 0)

        if isinstance(self.bw, np.ndarray):
            if self.bw.ndim != 1:
                raise ValueError(
                    f"The parameter `kernel_param` parameter must be a scalar or 1-dimensional. Got {self.bw.ndim} dimensions."
                )
            comparison = np.expand_dims(comparison, -1)

        comparison = (comparison**2).sum(axis=2)
        comparison = np.exp(-comparison / (2 * self.bw**2))
        assert comparison.shape[:2] == (len(x), len(y))
        return comparison


class AdaptiveRBFKernel(DescriptorKernel):
    def __init__(
        self,
        descriptor_fn: GraphDescriptorFn,
        bw: Union[float, np.ndarray],
        variant="mean",
    ) -> None:
        super().__init__(descriptor_fn)
        self.bw = bw
        self._variant = variant

    def get_subkernel(self, idx: int) -> DescriptorKernel:
        assert isinstance(self.bw, np.ndarray)
        return AdaptiveRBFKernel(self._descriptor_fn, self.bw[idx])

    @property
    def num_kernels(self) -> int:
        if isinstance(self.bw, np.ndarray):
            assert self.bw.ndim == 1
            return self.bw.size
        return 1

    def __call__(self, ref: np.ndarray, gen: np.ndarray):
        assert ref.ndim == 2 and gen.ndim == 2
        ref_ref = ((np.expand_dims(ref, 1) - np.expand_dims(ref, 0)) ** 2).sum(-1)
        ref_gen = ((np.expand_dims(ref, 1) - np.expand_dims(gen, 0)) ** 2).sum(-1)
        gen_gen = ((np.expand_dims(gen, 1) - np.expand_dims(gen, 0)) ** 2).sum(-1)

        if self._variant == "mean":
            mult = np.sqrt(ref_gen.mean())
        elif self._variant == "median":
            mult = np.sqrt(ref_gen.median())
        else:
            raise NotImplementedError

        bw = mult * self.bw

        if isinstance(bw, np.ndarray):
            if bw.ndim != 1:
                raise ValueError(
                    f"The parameter `kernel_param` parameter must be a scalar or 1-dimensional. Got {self.bw.ndim} dimensions."
                )
            ref_ref = np.expand_dims(ref_ref, -1)
            ref_gen = np.expand_dims(ref_gen, -1)
            gen_gen = np.expand_dims(gen_gen, -1)

        ref_ref = np.exp(-ref_ref / (2 * bw**2))
        ref_gen = np.exp(-ref_gen / (2 * bw**2))
        gen_gen = np.exp(-gen_gen / (2 * bw**2))
        return GramBlocks(ref_ref, ref_gen, gen_gen)

    def gram(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinearKernel(DescriptorKernel):
    @property
    def num_kernels(self) -> int:
        return 1

    def get_subkernel(self, idx: int) -> DescriptorKernel:
        assert idx == 0
        return LinearKernel(self._descriptor_fn)

    def gram(
        self, x: Union[np.ndarray, csr_array], y: Union[np.ndarray, csr_array]
    ) -> np.ndarray:
        assert x.ndim == 2 and y.ndim == 2
        result = x @ y.transpose()
        if isinstance(result, np.ndarray):
            return result
        return result.toarray()
