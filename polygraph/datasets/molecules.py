# -*- coding: utf-8 -*-
"""qm9.py

QM9 dataset.
"""

from __future__ import annotations

import networkx as nx
from torch_geometric.utils import from_networkx

from polygraph.datasets.base.dataset import SplitGraphDataset


class MoleculeDataset(SplitGraphDataset):
    """QM9 dataset.
    The backbone of this implementation is adapted from the PyG implementation of
    the QM9 dataset and the implementation from the DiGress paper
    https://github.com/cvignac/DiGress. We have removed molecules with unclear
    stereochemistry.
    """

    _HASH_FOR_SPLIT = {
        "train": None,
        "val": None,
        "test": None,
    }

    def hash_for_split(self, split: str) -> str:
        return self._HASH_FOR_SPLIT[split]

    def is_valid(self, graph: nx.Graph) -> bool:
        """Convert PyG graph back to RDKit molecule and validate it."""
        from polygraph.datasets.base.molecules import graph2molecule

        graph = from_networkx(graph)
        # Convert nx Graph to PyG Batch
        mol = graph2molecule(
            node_labels=graph.atom_labels,
            edge_index=graph.edge_index,
            bond_types=graph.bond_types,
            charges=graph.charges,
            num_radical_electrons=graph.radical_electrons,
            pos=graph.pos,
        )
        return mol is not None


class QM9(MoleculeDataset):
    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/OcGj065QUzbIgvM/download",
        "val": "https://datashare.biochem.mpg.de/s/bzuxQ3uZnPSTjMZ/download",
        "test": "https://datashare.biochem.mpg.de/s/xMJYUrAjz4D1tIj/download",
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]


class MOSES(MoleculeDataset):
    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/uMkCFQwuKPqnEIq/download",
        "val": "https://datashare.biochem.mpg.de/s/HBEyI64aySBUlta/download",
        "test": "https://datashare.biochem.mpg.de/s/nX61TFqMgRMwuKf/download",
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]


class Guacamol(MoleculeDataset):
    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/zMifbN4VWFohbdm/download",
        "val": "https://datashare.biochem.mpg.de/s/p9ZccTox73lIVFw/download",
        "test": "https://datashare.biochem.mpg.de/s/z2MlUDLB0SzAxoA/download",
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]
