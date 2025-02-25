"""guacamol.py"""

from __future__ import annotations

import os

import torch
from loguru import logger
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch, download_url

from graph_gen_gym.datasets.base.graph import Graph
from graph_gen_gym.datasets.base.molecules import (
    EDGE_ATTRS,
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

TRAIN_URL = "https://figshare.com/ndownloader/files/13612760"
TEST_URL = "https://figshare.com/ndownloader/files/13612757"
VALID_URL = "https://figshare.com/ndownloader/files/13612766"
ALL_URL = "https://figshare.com/ndownloader/files/13612745"


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([os.path.exists(f) for f in files])


def download(raw_dir):
    train_path = os.path.join(raw_dir, "guacamol_v1_train.smiles")
    if not os.path.exists(train_path):
        train_path = download_url(TRAIN_URL, raw_dir)
        os.rename(train_path, os.path.join(raw_dir, "guacamol_v1_train.smiles"))
    else:
        logger.info(
            f"Skipping download of train set, file already exists: {train_path}"
        )

    test_path = os.path.join(raw_dir, "guacamol_v1_test.smiles")
    if not os.path.exists(test_path):
        test_path = download_url(TEST_URL, raw_dir)
        os.rename(test_path, os.path.join(raw_dir, "guacamol_v1_test.smiles"))
    else:
        logger.info(f"Skipping download of test set, file already exists: {test_path}")

    valid_path = os.path.join(raw_dir, "guacamol_v1_valid.smiles")
    if not os.path.exists(valid_path):
        valid_path = download_url(VALID_URL, raw_dir)
        os.rename(valid_path, os.path.join(raw_dir, "guacamol_v1_val.smiles"))
    else:
        logger.info(
            f"Skipping download of valid set, file already exists: {valid_path}"
        )


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


def process(
    split: str, raw_dir: str, n_jobs: int, limit: int | None, chunk_size: int
) -> None:
    path = os.path.join(raw_dir, f"guacamol_v1_{split}.smiles")
    smile_list = open(path).readlines()
    if limit is not None:
        smile_list = smile_list[:limit]

    chunks = make_chunks(
        [(idx, item) for idx, item in enumerate(smile_list)], chunk_size
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
        edge_attrs=EDGE_ATTRS,
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

    download(args.destination)
    process("test", args.destination, args.n_jobs, args.limit, args.chunk_size)
    process("val", args.destination, args.n_jobs, args.limit, args.chunk_size)
    process("train", args.destination, args.n_jobs, args.limit, args.chunk_size)
