from typing import (
    Protocol,
    Iterable,
    Union,
    Generic,
)

import numpy as np
from scipy.sparse import csr_array

from polygraph import GraphType


class GraphDescriptor(Protocol, Generic[GraphType]):
    """Interface for graph descriptors.

    A graph descriptor is a callable that takes an iterable of graphs and returns a numpy array or a sparse matrix.
    Graphs must be of the type specified by the `GraphType` generic parameter. In practice, this may either be `nx.Graph` or `rdkit.Chem.Mol`.
    """

    def __call__(
        self, graphs: Iterable[GraphType]
    ) -> Union[np.ndarray, csr_array]:
        """Compute features of graphs.

        Args:
            graphs: Iterable of networkx graphs or rdkit molecules

        Returns:
            Features of graphs. Dense numpy array or sparse matrix of shape `(n_graphs, n_features)`.
        """
        ...
