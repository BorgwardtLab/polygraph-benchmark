"""Kernel functions for comparing graphs via descriptor functions.

This module implements various kernel functions that measure similarities between graphs.
Each kernel is initialized with a descriptor function that converts graphs into vector
representations. The kernel then computes similarities between these feature vectors.

Descriptor functions accept iterables of graphs (networkx graphs or rdkit molecules) and return descriptions as dense or sparse arrays.
Specifically, they should either return a dense `numpy.ndarray` or sparse `scipy.sparse.csr_array` of shape `(n_graphs, n_features)`.

Available kernels:
    - [`RBFKernel`][polygraph.utils.kernels.RBFKernel]: Standard Gaussian/RBF kernel using $\\ell^2$ distance
    - [`LaplaceKernel`][polygraph.utils.kernels.LaplaceKernel]: Exponential kernel using $\\ell^1$ distance
    - [`GaussianTV`][polygraph.utils.kernels.GaussianTV]: Non-positive definite Gaussian kernel using $\\ell^1$ distance
    - [`AdaptiveRBFKernel`][polygraph.utils.kernels.AdaptiveRBFKernel]: RBF kernel with data-dependent bandwidth adaptation
    - [`LinearKernel`][polygraph.utils.kernels.LinearKernel]: Simple dot product kernel

Example:
    ```python
    import numpy as np
    from polygraph.utils.kernels import RBFKernel
    import networkx as nx
    from typing import Iterable

    def my_descriptor(graphs: Iterable[nx.Graph]) -> np.ndarray:
        hists = [nx.degree_histogram(graph) for graph in graphs]
        hists = [
            np.concatenate([hist, np.zeros(128 - len(hist))], axis=0)
            for hist in hists
        ]
        hists = np.stack(hists, axis=0)
        return hists / hists.sum(axis=1, keepdims=True) # shape: (n_graphs, n_features)


    ref_graphs = [nx.erdos_renyi_graph(10, 0.3) for _ in range(100)]
    gen_graphs = [nx.erdos_renyi_graph(10, 0.3) for _ in range(100)]

    # Create kernel with multiple bandwidths
    kernel = RBFKernel(my_descriptor, bw=np.array([0.1, 1.0, 10.0]))

    ref_desc = kernel.featurize(ref_graphs) # shape: (n_ref_graphs, n_features)
    gen_desc = kernel.featurize(gen_graphs) # shape: (n_gen_graphs, n_features)

    # Compare reference and generated graph descriptors
    ref_vs_ref, ref_vs_gen, gen_vs_gen = kernel(ref_desc, gen_desc)
    ```
"""

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Iterable, Union, Literal, Generic
from typing_extensions import TypeAlias

import numpy as np
from scipy.sparse import csr_array
from sklearn.metrics import pairwise_distances

from polygraph import GraphType
from polygraph.utils.sparse_dist import (
    sparse_dot_product,
    sparse_euclidean,
    sparse_manhattan,
)
from polygraph.utils.descriptors import GraphDescriptor


MatrixLike: TypeAlias = Union[np.ndarray, csr_array]

GramBlocks = namedtuple(
    "GramBlocks", ["ref_vs_ref", "ref_vs_gen", "gen_vs_gen"]
)


class DescriptorKernel(ABC, Generic[GraphType]):
    """Abstract base class for kernel functions that operate on graph descriptors.

    This class defines the interface for kernels that compute similarity between graphs
    based on their descriptors. It handles both the featurization of graphs and the
    computation of kernel matrices.

    Args:
        descriptor_fn: Function that transforms graphs into descriptor vectors/matrices
    """

    def __init__(self, descriptor_fn: GraphDescriptor[GraphType]):
        self._descriptor_fn = descriptor_fn

    @abstractmethod
    def pre_gram_block(self, x: Any, y: Any) -> np.ndarray: ...

    @abstractmethod
    def get_subkernel(self, idx: int) -> "DescriptorKernel": ...

    @property
    def is_adaptive(self) -> bool:
        return type(self).adapt != DescriptorKernel.adapt

    @property
    @abstractmethod
    def num_kernels(self) -> int: ...

    def featurize(self, graphs: Iterable[GraphType]) -> Any:
        """Converts graphs into descriptor representations.

        Args:
            graphs: Collection of networkx graphs or rdkit molecules to featurize

        Returns:
            Descriptor representation of the graphs
        """
        return self._descriptor_fn(graphs)

    def pre_gram(self, ref: Any, gen: Any) -> GramBlocks:
        """Computes all kernel matrices between reference and generated samples.

        Args:
            ref: Descriptors of reference graphs
            gen: Descriptors of generated graphs

        Returns:
            Named tuple containing kernel matrices for ref-ref, ref-gen, and gen-gen comparisons
        """
        ref_vs_ref, ref_vs_gen, gen_vs_gen = (
            self.pre_gram_block(ref, ref),
            self.pre_gram_block(ref, gen),
            self.pre_gram_block(gen, gen),
        )
        assert (
            ref_vs_ref.ndim == ref_vs_gen.ndim
            and ref_vs_ref.ndim == gen_vs_gen.ndim
        )
        assert ref_vs_ref.shape[:2] == (ref.shape[0], ref.shape[0])
        assert ref_vs_gen.shape[:2] == (ref.shape[0], gen.shape[0])
        assert gen_vs_gen.shape[:2] == (gen.shape[0], gen.shape[0])
        return GramBlocks(ref_vs_ref, ref_vs_gen, gen_vs_gen)

    def adapt(self, blocks: GramBlocks) -> GramBlocks:
        """May adapt kernel parameters based on the computed kernel matrices.

        This method may, e.g., scale the bandwidth of a Gaussian kernel based on the
        typical distance between reference and generated graphs.

        Args:
            blocks: Pre-computed kernel matrices

        Returns:
            Adapted kernel matrices
        """
        return blocks

    def __call__(self, ref: Any, gen: Any) -> GramBlocks:
        """Computes the full kernel comparison between reference and generated graphs.

        Args:
            ref: Descriptors of reference graphs
            gen: Descriptors of generated graphs

        Returns:
            Named tuple containing all kernel matrices after adaptation
        """
        return self.adapt(self.pre_gram(ref, gen))


class LaplaceKernel(DescriptorKernel[GraphType], Generic[GraphType]):
    """Laplace kernel using L1 (Manhattan) distance.

    Computes similarity using the formula:

    $$
        k(x,y) = \\exp\\left(-\\lambda\\|x-y\\|_1\\right)
    $$

    where $\\lambda$ is the scale parameter.

    Args:
        descriptor_fn: Function that computes descriptors from graphs
        lbd: Scale parameter(s). Can be a single float or 1-dimensional numpy array of floats for multiple kernels
    """

    def __init__(
        self,
        descriptor_fn: GraphDescriptor[GraphType],
        lbd: Union[float, np.ndarray],
    ):
        super().__init__(descriptor_fn)
        self.lbd = lbd

    def get_subkernel(self, idx: int) -> DescriptorKernel[GraphType]:
        assert isinstance(self.lbd, np.ndarray)
        return LaplaceKernel(self._descriptor_fn, self.lbd[idx])

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
        assert comparison.shape[:2] == (x.shape[0], y.shape[0])  # pyright: ignore
        return comparison


class GaussianTV(DescriptorKernel[GraphType], Generic[GraphType]):
    """Gaussian kernel using L1 distance.

    Computes similarity using the formula:

    $$
        k(x,y) = \\exp\\left(-\\frac{(\\|x-y\\|_1/2)^2}{2\\sigma^2}\\right)
    $$

    where $\\sigma$ is the bandwidth parameter.

    Warning:
        This kernel is not positive definite.

    Args:
        descriptor_fn: Function that computes descriptors from graphs
        bw: Bandwidth parameter(s). Can be a single float or 1-dimensional numpy array of floats for multiple kernels
    """

    def __init__(
        self,
        descriptor_fn: GraphDescriptor[GraphType],
        bw: Union[float, np.ndarray],
    ):
        super().__init__(descriptor_fn)
        self.bw = bw

    def get_subkernel(self, idx: int) -> DescriptorKernel[GraphType]:
        assert isinstance(self.bw, np.ndarray)
        return GaussianTV(self._descriptor_fn, self.bw[idx])

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
        assert comparison.shape[:2] == (x.shape[0], y.shape[0])  # pyright: ignore
        return comparison


class RBFKernel(DescriptorKernel[GraphType], Generic[GraphType]):
    """Radial Basis Function (RBF) kernel, also known as Gaussian kernel.

    Computes similarity using the formula:

    $$
        k(x,y) = \\exp\\left(-\\frac{\\|x-y\\|^2}{2\\sigma^2}\\right)
    $$

    where $\\sigma$ is the bandwidth parameter.

    Args:
        descriptor_fn: Function that computes descriptors from graphs
        bw: Bandwidth parameter(s). Can be a single float or 1-dimensional numpy array of floats for multiple kernels
    """

    def __init__(
        self,
        descriptor_fn: GraphDescriptor[GraphType],
        bw: Union[float, np.ndarray],
    ) -> None:
        super().__init__(descriptor_fn)
        self.bw = bw

    def get_subkernel(self, idx: int) -> DescriptorKernel[GraphType]:
        assert isinstance(self.bw, np.ndarray)
        assert isinstance(idx, int), type(idx)
        return RBFKernel(self._descriptor_fn, self.bw[idx])

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
        assert comparison.shape[:2] == (x.shape[0], y.shape[0])  # pyright: ignore
        return comparison


class AdaptiveRBFKernel(DescriptorKernel[GraphType], Generic[GraphType]):
    """Adaptive RBF kernel with data-dependent bandwidth.

    Similar to the standard RBF kernel but adapts its bandwidth based on the data:

    $$
        k(x,y) = \\exp\\left(-\\frac{\\|x-y\\|^2}{2(c\\sigma)^2}\\right)
    $$

    where $\\sigma$ is the base bandwidth and $c$ is a scaling factor computed
    from the typical distance between reference and generated graphs.
    Specifically, $c$ is the square root of the mean or median of the squared $\\ell^2$ distance between reference and generated graphs.

    Args:
        descriptor_fn: Function that computes descriptors from graphs
        bw: Base bandwidth parameter(s). Can be a single float or 1-dimensional numpy array of floats for multiple kernels
        variant: Method for computing adaptive scaling. Either 'mean' or 'median' of the reference-generated distance.
    """

    _variant: Literal["mean", "median"]

    def __init__(
        self,
        descriptor_fn: GraphDescriptor[GraphType],
        bw: Union[float, np.ndarray],
        variant: Literal["mean", "median"] = "mean",
    ) -> None:
        super().__init__(descriptor_fn)
        self.bw = bw
        self._variant = variant

    def get_subkernel(self, idx: int) -> DescriptorKernel[GraphType]:
        assert isinstance(self.bw, np.ndarray)
        return AdaptiveRBFKernel(
            self._descriptor_fn, self.bw[idx], variant=self._variant
        )

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
                    f"The parameter `kernel_param` parameter must be a scalar or 1-dimensional. Got {bw.ndim} dimensions."
                )
            ref_ref = np.expand_dims(ref_ref, -1)
            ref_gen = np.expand_dims(ref_gen, -1)
            gen_gen = np.expand_dims(gen_gen, -1)

        ref_ref = np.exp(-ref_ref / (2 * bw**2))
        ref_gen = np.exp(-ref_gen / (2 * bw**2))
        gen_gen = np.exp(-gen_gen / (2 * bw**2))
        return GramBlocks(ref_ref, ref_gen, gen_gen)


class LinearKernel(DescriptorKernel[GraphType], Generic[GraphType]):
    """Simple linear kernel using dot product.

    Computes similarity using the formula:

    $$
        k(x,y) = x^\\top y
    $$

    Args:
        descriptor_fn: Function that computes descriptors from graphs
    """

    @property
    def num_kernels(self) -> int:
        return 1

    def get_subkernel(self, idx: int) -> DescriptorKernel[GraphType]:
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
