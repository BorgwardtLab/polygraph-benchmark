# -*- coding: utf-8 -*-
"""qm9.py

QM9 dataset.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional

import networkx as nx
import pandas as pd
import torch
from loguru import logger
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch, download_url, extract_zip
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from graph_gen_gym.datasets.base.caching import (
    identifier_to_path,
    to_list,
)
from graph_gen_gym.datasets.base.dataset import OnlineGraphDataset
from graph_gen_gym.datasets.base.graph import Graph
from graph_gen_gym.datasets.base.molecules import (
    EDGE_ATTRS,
    NODE_ATTRS,
    graph2molecule,
    molecule2graph,
)

# Those constants are used for the guidance attributes
HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414
conversion = torch.tensor(
    [
        1.0,
        1.0,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        1.0,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        1.0,
        KCALMOL2EV,
        KCALMOL2EV,
        KCALMOL2EV,
        KCALMOL2EV,
        1.0,
        1.0,
        1.0,
    ]
)

GUIDANCE_ATTRS = [
    "mu",
    "alpha",
    "eps_homo",
    "eps_lumo",
    "gap_eps_lumo_homo",
    "r2",
    "zpve",
    "u0",
    "u",
    "h",
    "g",
    "cv",
    "u0_atom",
    "u_atom",
    "h_atom",
    "g_atom",
    "a",
    "b",
    "c",
]


class QM9(OnlineGraphDataset):
    """QM9 dataset.
    The backbone of this implementation is taking from the PyG implementation of
    the QM9 dataset and the implementation from the DiGress paper
    https://github.com/cvignac/DiGress.
    """

    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/cnRpafWxInKGUWB",
        "val": "https://datashare.biochem.mpg.de/s/FO1jlHTDqQwvG9o",
        "test": "https://datashare.biochem.mpg.de/s/O6pyw8wo5SZ0zg0",
    }
    raw_url = (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
        "molnet_publish/qm9.zip"
    )
    raw_url2 = "https://ndownloader.figshare.com/files/3195404"
    processed_url = "https://data.pyg.org/datasets/qm9_v3.zip"

    def __init__(
        self,
        split: Optional[str] = None,
        use_precomputed: bool = True,
        data_store: Optional[Graph] = None,
    ):
        if not use_precomputed:
            self.download_raw_data()
            self.split_raw_data()
            self.process_split(split)
        super().__init__(data_store)

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]

    @property
    def raw_dir(self) -> str:
        return identifier_to_path(self.identifier)

    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading.
        """
        files = self.raw_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [
            os.path.join(identifier_to_path(self.identifier), f) for f in to_list(files)
        ]

    @property
    def raw_file_names(self) -> List[str]:
        return ["gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt"]

    def download_raw_data(self):
        # Check if raw files already exist
        if not all(os.path.exists(path) for path in self.raw_paths):
            # Download and extract first file
            file_path = download_url(self.raw_url, self.raw_dir)
            try:
                extract_zip(file_path, self.raw_dir)
            except Exception as e:
                logger.error(f"Error extracting zip file {file_path} because of {e}")
                raise e

            # Download and rename second file
            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(
                os.path.join(self.raw_dir, "3195404"),
                os.path.join(self.raw_dir, "uncharacterized.txt"),
            )

    def split_raw_data(self):
        dataset = pd.read_csv(self.raw_paths[1])

        n_samples = len(dataset)
        n_train = 100000
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)

        shuffled = dataset.sample(frac=1, random_state=42)
        train = shuffled.iloc[:n_train]
        val = shuffled.iloc[n_train : n_train + n_val]
        test = shuffled.iloc[n_train + n_val :]

        train.to_csv(os.path.join(self.raw_dir, "train.csv"))
        val.to_csv(os.path.join(self.raw_dir, "val.csv"))
        test.to_csv(os.path.join(self.raw_dir, "test.csv"))

    def process_split(self, split_name):
        RDLogger.DisableLog("rdApp.*")

        target_df = pd.read_csv(
            os.path.join(self.raw_dir, f"{split_name}.csv"), index_col=0
        )
        target_df.drop(columns=["mol_id"], inplace=True)
        with open(self.raw_paths[1]) as f:
            target = [
                [float(x) for x in line.split(",")[1:20]]
                for line in f.read().split("\n")[1:-1]
            ]
            y = torch.tensor(target, dtype=torch.float)
            y = torch.cat([y[:, 3:], y[:, :3]], dim=-1)
            y = y * conversion.view(1, -1)

        with open(self.raw_paths[2], "r") as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl, desc="Processing QM9 molecules")):
            if i in skip or i not in target_df.index:
                continue
            data = molecule2graph(mol)
            # Add node-level attributes
            for attr in GUIDANCE_ATTRS:
                setattr(data, attr, y[i, GUIDANCE_ATTRS.index(attr)])
            # Add num_nodes explicitly
            data.num_nodes = data.atom_labels.size(0)
            data_list.append(data)

        pyg_batch = Batch.from_data_list(data_list)

        graph_storage = Graph.from_pyg_batch(
            pyg_batch,
            node_attrs=NODE_ATTRS,
            edge_attrs=EDGE_ATTRS,
            # TODO: graph attrs cannot be added bc they are strings
            graph_attrs=GUIDANCE_ATTRS,
        )
        torch.save(
            graph_storage.model_dump(),
            os.path.join(self.raw_dir, f"{split_name}.pt"),
        )

    @staticmethod
    def is_valid(data: nx.Graph) -> bool:
        """Convert PyG graph back to RDKit molecule and validate it."""
        graph = from_networkx(data)
        # Convert nx Graph to PyG Batch
        mol = graph2molecule(
            node_labels=graph.atom_labels,
            edge_index=graph.edge_index,
            edge_labels=graph.bond_labels,
            explicit_hydrogens=graph.explicit_hydrogens,
            charges=graph.charges,
            num_radical_electrons=graph.radical_electrons,
            stereo=graph.stereo,
        )
        return mol is not None
