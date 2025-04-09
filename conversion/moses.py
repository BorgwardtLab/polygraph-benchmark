"""moses.py"""

from __future__ import annotations

import os
import os.path as osp

import pandas as pd
import rdkit  # noqa
import torch
from loguru import logger
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch, download_url

from polygraph.datasets.base.graph_storage import GraphStorage
from polygraph.datasets.base.molecules import (
    EDGE_ATTRS,
    NODE_ATTRS,
    add_hydrogens_and_stereochemistry,
    graph2molecule,
    molecule2graph,
)
from polygraph.utils.parallel import (
    distribute_function,
    flatten_lists,
    make_batches,
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

    _ = download_file(train_url, raw_dir, "train_moses.csv", "train")
    _ = download_file(test_url, raw_dir, "val_moses.csv", "test")
    _ = download_file(val_url, raw_dir, "test_moses.csv", "val")


def process(split, out_dir, n_jobs, limit, chunk_size):
    RDLogger.DisableLog("rdApp.*")

    path = osp.join(out_dir, f"{split}_moses.csv")
    smiles_list = pd.read_csv(path)["SMILES"].values.tolist()

    data_list = []
    if limit is not None:
        smiles_list = smiles_list[:limit]

    chunks = make_batches(
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
    graph_storage = GraphStorage.from_pyg_batch(
        pyg_batch,
        node_attrs=NODE_ATTRS,
        edge_attrs=EDGE_ATTRS,
    ).model_dump()
    torch.save(
        graph_storage,
        os.path.join(out_dir, f"{split}.pt"),
    )
    logger.info(f"Saved graph storage to {os.path.join(out_dir, f'{split}.pt')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=str, required=True)
    parser.add_argument("--n_jobs", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=1000)
    args = parser.parse_args()

    download(train_url, test_url, val_url, args.destination)
    process("test", args.destination, args.n_jobs, args.limit, args.chunk_size)
    process("val", args.destination, args.n_jobs, args.limit, args.chunk_size)
    process("train", args.destination, args.n_jobs, args.limit, args.chunk_size)
