from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, List, Union

import networkx as nx
import numpy as np

GraphDescriptorFn = Callable[[Iterable[nx.Graph]], np.ndarray]


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

    def __call__(self, x: Any, y: Any) -> np.ndarray:
        kxy = self.gram(x, y)
        if self.num_kernels > 1:
            assert kxy.ndim == 3 and kxy.shape[2] == self.num_kernels, kxy.shape
        else:
            assert kxy.ndim == 2
        return kxy


class StackedKernel(DescriptorKernel):
    def __init__(self, kernels: List[DescriptorKernel]):
        self._kernels = kernels
        self._kernel_count = np.array([kernel.num_kernels for kernel in self._kernels])
        self._num_kernels = np.sum(self._kernel_count)
        self._cumsum_num_kernels = np.cumsum(self._kernel_count) - self._kernel_count

    def featurize(self, graphs: Iterable[nx.Graph]) -> List:
        return [kernel.featurize(graphs) for kernel in self._kernels]

    def gram(self, x: List, y: List) -> np.ndarray:
        assert len(x) == len(y) and len(x) == len(self._kernels)
        results = [
            kernel(feat_x, feat_y)
            for kernel, feat_x, feat_y in zip(self._kernels, x, y)
        ]
        results = [
            np.expand_dims(result, axis=-1) if result.ndim == 2 else result
            for result in results
        ]
        return np.concatenate(results, axis=-1)

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


class LinearKernel(DescriptorKernel):
    @property
    def num_kernels(self) -> int:
        return 1

    def get_subkernel(self, idx: int) -> DescriptorKernel:
        assert idx == 0
        return LinearKernel(self._descriptor_fn)

    def gram(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x @ y.T
