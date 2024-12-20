# -*- coding: utf-8 -*-
"""qm9.py

QM9 dataset.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Batch, Data, download_url, extract_zip
from tqdm import tqdm

from graph_gen_gym.datasets.base.molecules import molecule2graph
from graph_gen_gym.datasets.base.dataset import OnlineGraphDataset
from graph_gen_gym.datasets.base.graph import Graph
from graph_gen_gym.datasets.base.caching import (
    identifier_to_path,
    to_list,
)

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
ATOM_TYPES = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
BOND_TYPES = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}


class QM9(OnlineGraphDataset):
    """QM9 dataset.
    The backbone of this implementation is taking from the PyG implementation of
    the QM9 dataset and the implementation from the DiGress paper
    https://github.com/cvignac/DiGress.
    """

    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/O6pyw8wo5SZ0zg0",
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
        split: str = "train",
        use_precomputed: bool = True,
        data_store: Optional[Graph] = None,
        pre_filter: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        if not use_precomputed:
            self.download_raw_data()
            self.split_raw_data()
            self.process_split(
                split, pre_filter=pre_filter, pre_transform=pre_transform
            )
        super().__init__(
            split, data_store, pre_filter=pre_filter, pre_transform=pre_transform
        )

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

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(
            dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train]
        )

        train.to_csv(os.path.join(self.raw_dir, "train.csv"))
        val.to_csv(os.path.join(self.raw_dir, "val.csv"))
        test.to_csv(os.path.join(self.raw_dir, "test.csv"))

    def process_split(self, split_name, pre_filter, pre_transform):
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
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue
            data = molecule2graph(mol)
            # QM9-specific labels
            data_list.append(data)

        pyg_batch = Batch.from_data_list(data_list)

        # pyg_batch.smiles = [data.smiles for data in data_list]
        # pyg_batch.names = [data.name for data in data_list]
        # graph_attrs = ["smiles", "name"]

        for guidance_name in GUIDANCE_ATTRS:
            setattr(
                pyg_batch,
                guidance_name,
                y[:, GUIDANCE_ATTRS.index(guidance_name)],
            )

        graph_storage = Graph.from_pyg_batch(
            pyg_batch,
            node_attrs=[
                "atom_labels",
                "explicit_hydrogens",
                "implicit_hydrogens",
                "charges",
                "radical_electrons",
            ]
            + GUIDANCE_ATTRS,
            edge_attrs=["bond_labels"],
            # TODO: graph attrs cannot be added bc they are strings
            # graph_attrs=graph_attrs,
        )
        torch.save(
            graph_storage.model_dump(),
            os.path.join(self.raw_dir, f"{split_name}.pt"),
        )

    def is_valid(self, data: Data) -> bool:
        """Convert PyG graph back to RDKit molecule and validate it."""
        mol = Chem.RWMol()
        # Add atoms
        for atom_label in data.atom_labels:
            atom = Chem.Atom(int(atom_label.item()))
            mol.AddAtom(atom)
        logger.info(f"Added atoms to mol: {mol}")

        # Add bonds
        edge_index = data.edge_index
        bond_labels = data.bond_labels
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            bond_type = bond_labels[i].item()
            # Only add bond if source index is less than destination
            # to avoid adding same bond twice
            if src < dst:
                try:
                    mol.AddBond(int(src), int(dst), list(BOND_TYPES.keys())[bond_type])
                except Exception as e:
                    logger.error(f"Error adding bond: {e}")
                    return False
        logger.info(f"Added bonds to mol: {mol}")

        # Set atom properties
        for i, (exp_h, imp_h, charge, radical) in enumerate(zip(
            data.explicit_hydrogens,
            data.implicit_hydrogens,
            data.charges,
            data.radical_electrons
        )):
            try:
                atom = mol.GetAtomWithIdx(i)
                atom.SetNumExplicitHs(int(exp_h.item()))
                atom.SetFormalCharge(int(charge.item()))
                atom.SetNumRadicalElectrons(int(radical.item()))
            except Exception as e:
                # Plot networkx graph
                import networkx as nx
                import matplotlib.pyplot as plt
                G = nx.from_pyg_batch(data)
                nx.draw(G, with_labels=True)
                plt.savefig("invalid_mol.png")
                logger.error(f"Error setting atom properties: {e}")
                return False
        logger.info(f"Set atom properties: {mol}")

        try:
            Chem.SanitizeMol(mol)
            logger.success(f"Sanitized mol: {mol}")
            return True
        except Exception as e:
            logger.error(f"Invalid molecule: {e}")
            return False
