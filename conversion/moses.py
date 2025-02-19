"""moses.py"""

from __future__ import annotations

import os
import os.path as osp

import pandas as pd
import rdkit  # noqa
import torch
from loguru import logger
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Batch, download_url

from graph_gen_gym.datasets.base.graph import Graph
from graph_gen_gym.datasets.base.molecules import (
    NODE_ATTRS,
    add_hydrogens_and_stereochemistry,
    graph2molecule,
    molecule2graph,
)
from graph_gen_gym.utils.parallel import (
    distribute_function,
    flatten_lists,
    make_chunks,
    retry,
)

moses_url = "https://media.githubusercontent.com/media/molecularsets/moses/"

train_url = moses_url + "master/data/train.csv"
val_url = moses_url + "master/data/test.csv"
test_url = moses_url + "master/data/test_scaffolds.csv"

raw_file_names = ["train_moses.csv", "val_moses.csv", "test_moses.csv"]
processed_file_names = ["train.pt", "test.pt", "test_scaffold.pt"]
atom_decoder = ["C", "N", "S", "O", "F", "Cl", "Br", "H"]


def check_smiles_graph_mapping(smiles):
    data_list = []
    for smile_idx, smile in smiles:
        try:
            data = check_smiles_graph_mapping_worker(smile_idx, smile)
            data_list.append(data)
        except Exception as e:
            logger.error(f"Error processing smile: {smile}")
            logger.error(f"Error: {e}")
    return data_list


@retry(max_retries=3, delay=1.0)
def check_smiles_graph_mapping_worker(smile_idx, smile):
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smile)
    data = molecule2graph(mol)

    reconstructed = graph2molecule(
        node_labels=data.atom_labels,
        edge_index=data.edge_index,
        bond_types=data.bond_types,
        stereo_types=data.stereo_types,
        charges=data.charges,
        num_radical_electrons=data.radical_electrons,
        pos=data.pos,
    )
    reconstructed_smiles = Chem.MolToSmiles(reconstructed, canonical=True)
    smile2 = Chem.MolToSmiles(mol, canonical=True)
    mol2 = Chem.MolFromSmiles(smile2)
    mol2 = add_hydrogens_and_stereochemistry(mol2)
    smile2 = Chem.MolToSmiles(mol2, canonical=True)
    assert smile2 == reconstructed_smiles, (smile2, reconstructed_smiles)
    return data


def download(train_url, test_url, val_url, raw_dir):
    def download_file(url, raw_dir, filename, split_name):
        path = osp.join(raw_dir, filename)
        if not os.path.exists(path):
            downloaded_path = download_url(url, raw_dir)
            os.rename(downloaded_path, path)
        else:
            logger.info(
                f"Skipping download of {split_name} set, file already exists: {path}"
            )
        return path

    train_path = download_file(train_url, raw_dir, "train_moses.csv", "train")
    test_path = download_file(test_url, raw_dir, "val_moses.csv", "test")
    valid_path = download_file(val_url, raw_dir, "test_moses.csv", "valid")


def process(split, raw_dir, n_jobs, limit, chunk_size):
    RDLogger.DisableLog("rdApp.*")
    types = {atom: i for i, atom in enumerate(atom_decoder)}

    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    path = osp.join(raw_dir, raw_file_names[0])
    smiles_list = pd.read_csv(path)["SMILES"].values.tolist()

    data_list = []
    smiles_kept = []
    if limit is not None:
        smiles_list = smiles_list[:limit]

    chunks = make_chunks(
        [(idx, item) for idx, item in enumerate(smiles_list)], chunk_size
    )

    data_list = (
        flatten_lists(
            distribute_function(
                check_smiles_graph_mapping,
                chunks,
                n_jobs=n_jobs,
                description=f"Processing {split} smiles",
            ),
        ),
    )
    data_list = data_list[0]
    pyg_batch = Batch.from_data_list(data_list)
    logger.info(f"Created PyG batch with {pyg_batch.num_graphs} graphs")
    graph_storage = Graph.from_pyg_batch(
        pyg_batch,
        node_attrs=NODE_ATTRS,
    ).model_dump()

    torch.save(
        graph_storage,
        os.path.join(raw_dir, f"{split}.pt"),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=str, required=True)
    parser.add_argument("--n_jobs", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=100)
    args = parser.parse_args()

    download(train_url, test_url, val_url, args.destination)
    process("test", args.destination, args.n_jobs, args.limit, args.chunk_size)
    process("valid", args.destination, args.n_jobs, args.limit, args.chunk_size)
    process("train", args.destination, args.n_jobs, args.limit, args.chunk_size)
