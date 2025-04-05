from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Callable, Iterable, Union
from typing_extensions import TypeAlias

import networkx as nx
import numpy as np
from scipy.sparse import csr_array
from sklearn.metrics import pairwise_distances
from polygrapher.utils.sparse_dist import sparse_dot_product, sparse_euclidean, sparse_manhattan

GraphDescriptorFn = Callable[[Iterable[nx.Graph]], np.ndarray]
MatrixLike: TypeAlias = Union[np.ndarray, csr_array]

GramBlocks = namedtuple("GramBlocks", ["ref_vs_ref", "ref_vs_gen", "gen_vs_gen"])


class DescriptorKernel(ABC):
    def __init__(self, descriptor_fn: GraphDescriptorFn):
        self._descriptor_fn = descriptor_fn

    @abstractmethod
    def pre_gram_block(self, x: Any, y: Any) -> np.ndarray: ...

    @abstractmethod
    def get_subkernel(self, idx: int) -> "DescriptorKernel": ...

    @property
    @abstractmethod
    def is_adative(self) -> bool: ...

    @property
    @abstractmethod
    def num_kernels(self) -> int: ...

    def featurize(self, graphs: Iterable[nx.Graph]) -> Any:
        return self._descriptor_fn(graphs)

    def pre_gram(self, ref: Any, gen: Any) -> GramBlocks:
        ref_vs_ref, ref_vs_gen, gen_vs_gen = (
            self.pre_gram_block(ref, ref),
            self.pre_gram_block(ref, gen),
            self.pre_gram_block(gen, gen),
        )
        assert ref_vs_ref.ndim == ref_vs_gen.ndim and ref_vs_ref.ndim == gen_vs_gen.ndim
        assert ref_vs_ref.shape[:2] == (ref.shape[0], ref.shape[0])
        assert ref_vs_gen.shape[:2] == (ref.shape[0], gen.shape[0])
        assert gen_vs_gen.shape[:2] == (gen.shape[0], gen.shape[0])
        return GramBlocks(ref_vs_ref, ref_vs_gen, gen_vs_gen)

    def adapt(self, blocks: GramBlocks) -> GramBlocks:
        return blocks

    def __call__(self, ref: Any, gen: Any) -> GramBlocks:
        return self.adapt(self.pre_gram(ref, gen))


class LaplaceKernel(DescriptorKernel):
    def __init__(self, descriptor_fn: GraphDescriptorFn, lbd: Union[float, np.ndarray]):
        super().__init__(descriptor_fn)
        self.lbd = lbd

    def get_subkernel(self, idx: int) -> DescriptorKernel:
        assert isinstance(self.lbd, np.ndarray)
        return LaplaceKernel(self._descriptor_fn, self.lbd[idx])

    @property
    def is_adative(self) -> bool:
        False

    @property
    def num_kernels(self) -> int:
        if isinstance(self.lbd, np.ndarray):
            assert self.lbd.ndim == 1
            return self.lbd.size
        return 1

    def pre_gram_block(self, x: MatrixLike, y: MatrixLike) -> np.ndarray:
        assert x.ndim == 2 and y.ndim == 2

        if isinstance(x, csr_array) and isinstance(y, csr_array):
            comparison = sparse_manhattan(x, y)
        else:
            comparison = pairwise_distances(x, y, metric="l1")

        if isinstance(self.lbd, np.ndarray):
            if self.lbd.ndim != 1:
                raise ValueError(
                    f"The parameter `kernel_param` parameter must be a scalar or 1-dimensional. Got {self.lbd.ndim} dimensions."
                )
            comparison = np.expand_dims(comparison, -1)

        comparison = np.exp(-self.lbd * comparison)
        assert comparison.shape[:2] == (x.shape[0], y.shape[0])
        return comparison


class GaussianTV(DescriptorKernel):
    def __init__(self, descriptor_fn: GraphDescriptorFn, bw: Union[float, np.ndarray]):
        super().__init__(descriptor_fn)
        self.bw = bw

    def get_subkernel(self, idx: int) -> DescriptorKernel:
        assert isinstance(self.bw, np.ndarray)
        return GaussianTV(self._descriptor_fn, self.bw[idx])

    @property
    def is_adative(self) -> bool:
        return False

    @property
    def num_kernels(self) -> int:
        if isinstance(self.bw, np.ndarray):
            assert self.bw.ndim == 1
            return self.bw.size
        return 1

    def pre_gram_block(self, x: MatrixLike, y: MatrixLike) -> np.ndarray:
        assert x.ndim == 2 and y.ndim == 2

        if isinstance(x, csr_array) and isinstance(y, csr_array):
            comparison = sparse_manhattan(x, y)
        else:
            comparison = pairwise_distances(x, y, metric="l1")

        if isinstance(self.bw, np.ndarray):
            if self.bw.ndim != 1:
                raise ValueError(
                    f"The parameter `kernel_param` parameter must be a scalar or 1-dimensional. Got {self.bw.ndim} dimensions."
                )
            comparison = np.expand_dims(comparison, -1)

        comparison = np.exp(-((comparison / 2) ** 2) / (2 * self.bw**2))
        assert comparison.shape[:2] == (x.shape[0], y.shape[0])
        return comparison


class RBFKernel(DescriptorKernel):
    def __init__(
        self, descriptor_fn: GraphDescriptorFn, bw: Union[float, np.ndarray]
    ) -> None:
        super().__init__(descriptor_fn)
        self.bw = bw

    def get_subkernel(self, idx: int) -> DescriptorKernel:
        assert isinstance(self.bw, np.ndarray)
        assert isinstance(idx, int), type(idx)
        return RBFKernel(self._descriptor_fn, self.bw[idx])

    @property
    def is_adative(self) -> bool:
        return False

    @property
    def num_kernels(self) -> int:
        if isinstance(self.bw, np.ndarray):
            assert self.bw.ndim == 1
            return self.bw.size
        return 1

    def pre_gram_block(self, x: MatrixLike, y: MatrixLike) -> np.ndarray:
        assert x.ndim == 2 and y.ndim == 2

        if isinstance(x, csr_array) and isinstance(y, csr_array):
            comparison = sparse_euclidean(x, y) ** 2
        else:
            comparison = pairwise_distances(x, y, metric="l2") ** 2

        if isinstance(self.bw, np.ndarray):
            if self.bw.ndim != 1:
                raise ValueError(
                    f"The parameter `kernel_param` parameter must be a scalar or 1-dimensional. Got {self.bw.ndim} dimensions."
                )
            comparison = np.expand_dims(comparison, -1)

        comparison = np.exp(-comparison / (2 * self.bw**2))
        assert comparison.shape[:2] == (x.shape[0], y.shape[0])
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
        return AdaptiveRBFKernel(
            self._descriptor_fn, self.bw[idx], variant=self._variant
        )

    @property
    def is_adative(self) -> bool:
        return True

    @property
    def num_kernels(self) -> int:
        if isinstance(self.bw, np.ndarray):
            assert self.bw.ndim == 1
            return self.bw.size
        return 1

    def pre_gram_block(self, x: Any, y: Any) -> np.ndarray:
        if isinstance(x, csr_array) and isinstance(y, csr_array):
            comparison = sparse_euclidean(x, y) ** 2
        else:
            comparison = pairwise_distances(x, y, metric="l2") ** 2
        return comparison

    def adapt(self, blocks: GramBlocks) -> GramBlocks:
        ref_ref, ref_gen, gen_gen = blocks

        if self._variant == "mean":
            mult = np.sqrt(np.mean(ref_gen)) if np.mean(ref_gen) > 0 else 1
        elif self._variant == "median":
            mult = np.sqrt(np.median(ref_gen)) if np.median(ref_gen) > 0 else 1
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


class LinearKernel(DescriptorKernel):
    @property
    def num_kernels(self) -> int:
        return 1

    @property
    def is_adative(self) -> bool:
        return False

    def get_subkernel(self, idx: int) -> DescriptorKernel:
        assert idx == 0, idx
        return LinearKernel(self._descriptor_fn)

    def pre_gram_block(self, x: MatrixLike, y: MatrixLike) -> np.ndarray:
        assert x.ndim == 2 and y.ndim == 2

        if isinstance(x, csr_array) and isinstance(y, csr_array):
            result = sparse_dot_product(x, y)
        else:
            result = x @ y.transpose()

        if isinstance(result, np.ndarray):
            return result
        return result.toarray()