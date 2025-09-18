from importlib.metadata import version
from typing import TypeVar
import networkx as nx
from rdkit.Chem import rdchem

__version__ = version("polygraph")

GraphType = TypeVar("GraphType", nx.Graph, rdchem.Mol, contravariant=True)
