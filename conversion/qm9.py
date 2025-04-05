# -*- coding: utf-8 -*-
"""qm9.py

QM9 dataset.
"""

from __future__ import annotations

import os

import pandas as pd
import torch
from loguru import logger
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch, download_url, extract_zip
from tqdm.rich import tqdm

from polygrapher.datasets.base.graph import Graph
from polygrapher.datasets.base.molecules import (
    EDGE_ATTRS,
    NODE_ATTRS,
    add_hydrogens_and_stereochemistry,
    are_smiles_equivalent,
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

# Removed stereochemistry-dependent molecules.
UNCLEAR_STEREO_SPECS = [
    38359,
    75031,
    42056,
    43391,
    17957,
    39605,
    75045,
    93167,
    38365,
    39614,
    42059,
    42062,
    42068,
    42069,
    42117,
    45903,
    47263,
    47264,
    75406,
    93166,
    95588,
]

raw_url = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/" "molnet_publish/qm9.zip"
)
raw_url2 = "https://ndownloader.figshare.com/files/3195404"
raw_file_names = ["gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt"]


def download_raw_data(raw_paths, raw_url, raw_url2, raw_dir):
    if not all(os.path.exists(path) for path in raw_paths):
        file_path = download_url(raw_url, raw_dir)
        try:
            extract_zip(file_path, raw_dir)
        except Exception as e:
            logger.error(f"Error extracting zip file {file_path} because of {e}")
            raise e

        # Download and rename second file
        file_path = download_url(raw_url2, raw_dir)
        os.rename(
            os.path.join(raw_dir, "3195404"),
            os.path.join(raw_dir, "uncharacterized.txt"),
        )


def split_raw_data(raw_paths, raw_dir):
    dataset = pd.read_csv(raw_paths[1])

    n_samples = len(dataset)
    n_train = 100000
    n_test = int(0.1 * n_samples)
    n_val = n_samples - (n_train + n_test)

    shuffled = dataset.sample(frac=1, random_state=42)
    train = shuffled.iloc[:n_train]
    val = shuffled.iloc[n_train : n_train + n_val]
    test = shuffled.iloc[n_train + n_val :]

    train.to_csv(os.path.join(raw_dir, "train.csv"))
    val.to_csv(os.path.join(raw_dir, "val.csv"))
    test.to_csv(os.path.join(raw_dir, "test.csv"))

    return train, val, test


def process_split(raw_paths, raw_dir, split_name):
    # Distinguish between molecules that fail sanitization from input file, sanitization after reconstruction, and assert failure despite passing sanitization.
    RDLogger.DisableLog("rdApp.*")

    target_df = pd.read_csv(os.path.join(raw_dir, f"{split_name}.csv"), index_col=0)
    target_df.drop(columns=["mol_id"], inplace=True)
    with open(raw_paths[1]) as f:
        target = [
            [float(x) for x in line.split(",")[1:20]]
            for line in f.read().split("\n")[1:-1]
        ]
        y = torch.tensor(target, dtype=torch.float)
        y = torch.cat([y[:, 3:], y[:, :3]], dim=-1)
        y = y * conversion.view(1, -1)

    with open(raw_paths[2], "r") as f:
        skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

    skip = skip + UNCLEAR_STEREO_SPECS

    suppl = Chem.SDMolSupplier(
        raw_paths[0],
        removeHs=False,
        sanitize=True,
    )

    data_list = []
    for i, mol in enumerate(tqdm(suppl, desc="Processing QM9 molecules")):
        if i in skip or i not in target_df.index or mol is None:
            continue

        if len(Chem.GetMolFrags(mol)) > 1:
            logger.warning(f"Fragmented molecule {i}")
            continue

        mol = add_hydrogens_and_stereochemistry(mol)
        smiles_mol = Chem.MolToSmiles(mol, canonical=True)

        data = molecule2graph(mol)
        mol_reconstructed = graph2molecule(
            node_labels=data.atom_labels,
            edge_index=data.edge_index,
            bond_types=data.bond_types,
            charges=data.charges,
            num_radical_electrons=data.radical_electrons,
            pos=data.pos,
        )
        smiles_graph = Chem.MolToSmiles(mol_reconstructed)
        try:
            assert are_smiles_equivalent(
                smiles_mol,
                smiles_graph,
            ), f"Molecule {i} is not equivalent to its graph representation, mol: {smiles_mol}, from graph: {smiles_graph}"
        except AssertionError as e:
            logger.error(f"Error checking if molecules are equivalent: {e}")
            continue
        # Add graph-level attributes
        for attr in GUIDANCE_ATTRS:
            setattr(data, attr, y[i, GUIDANCE_ATTRS.index(attr)])

        # Add num_nodes explicitly
        data.num_nodes = data.atom_labels.size(0)
        data_list.append(data)
    logger.info(f"Processed {len(data_list)} molecules")
    pyg_batch = Batch.from_data_list(data_list)
    logger.info(f"Created PyG batch with {pyg_batch.num_graphs} graphs")
    graph_storage = Graph.from_pyg_batch(
        pyg_batch,
        node_attrs=NODE_ATTRS,
        edge_attrs=EDGE_ATTRS,
        # TODO: graph attrs cannot be added bc they are strings
        graph_attrs=GUIDANCE_ATTRS,
    )
    logger.info(f"Created Graph storage with {graph_storage.num_graphs} graphs")
    torch.save(
        graph_storage.model_dump(),
        os.path.join(raw_dir, f"{split_name}.pt"),
    )

    storage = torch.load(os.path.join(raw_dir, f"{split_name}.pt"), weights_only=True)
    data = Graph(**storage)
    logger.info(f"Saved Graph storage to {os.path.join(raw_dir, f'{split_name}.pt')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=str, required=True)
    args = parser.parse_args()

    raw_paths = [os.path.join(args.destination, f) for f in raw_file_names]
    download_raw_data(raw_paths, raw_url, raw_url2, args.destination)
    train, val, test = split_raw_data(raw_paths, args.destination)
    os.makedirs(args.destination, exist_ok=True)
    process_split(raw_paths, args.destination, "test")
    process_split(raw_paths, args.destination, "val")
    process_split(raw_paths, args.destination, "train")
