from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class DescriptorKernel(ABC):
    @abstractmethod
    def gram(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ...

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert x.ndim == 2 and y.ndim == 2
        kxy = self.gram(x, y)
        assert kxy.shape[0] == len(x) and kxy.shape[1] == len(y)
        return kxy


class LaplaceKernel(DescriptorKernel):
    def __init__(self, lbd: Union[float, np.ndarray]):
        self.lbd = lbd

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
