from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class DescriptorKernel(ABC):
    @abstractmethod
    def gram(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def get_subkernel(self, idx: int) -> "DescriptorKernel":
        ...

    @property
    @abstractmethod
    def num_kernels(self) -> int:
        ...

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert x.ndim == 2 and y.ndim == 2
        kxy = self.gram(x, y)
        assert kxy.shape[0] == len(x) and kxy.shape[1] == len(y)
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

    def gram(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        results = [kernel(x, y) for kernel in self._kernel]
        results = [
            np.expand_dims(result, axis=-1) if result.ndim == 2 else result
            for result in results
        ]
        return np.concatenate(results, axis=-1)

    @property
    def num_kernels(self) -> int:
        return self._num_kernels

    def get_subkernel(self, idx: int) -> DescriptorKernel:
        idx1 = np.searchsorted(self._cumsum_num_kernels, side="left") - 1
        idx2 = idx - self._cumsum_num_kernels[idx1]
        return self._kernelsp[idx1].get_subkernel(idx2)


class LaplaceKernel(DescriptorKernel):
    def __init__(self, lbd: Union[float, np.ndarray]):
        self.lbd = lbd

    def get_subkernel(self, idx: int) -> DescriptorKernel:
        assert isinstance(self.lbd, np.ndarray)
        return LaplaceKernel(self.lbd[idx])

    @property
    def num_kernels(self) -> int:
        if isinstance(self.lbd, np.ndarray):
            assert self.lbd.ndim == 1
            return self.lbd.size()
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
    def __init__(self, bw: Union[float, np.ndarray]):
        self.bw = bw

    def get_subkernel(self, idx: int) -> DescriptorKernel:
        assert isinstance(self.bw, np.ndarray)
        return GaussianTV(self.bw[idx])

    @property
    def num_kernels(self) -> int:
        if isinstance(self.bw, np.ndarray):
            assert self.bw.ndim == 1
            return self.bw.size()
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
    def __init__(self, bw: Union[float, np.ndarray]) -> None:
        self.bw = bw

    def get_subkernel(self, idx: int) -> DescriptorKernel:
        assert isinstance(self.bw, np.ndarray)
        return RBFKernel(self.bw[idx])

    @property
    def num_kernels(self) -> int:
        if isinstance(self.bw, np.ndarray):
            assert self.bw.ndim == 1
            return self.bw.size()
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
        return LinearKernel()

    def gram(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x @ y.T
