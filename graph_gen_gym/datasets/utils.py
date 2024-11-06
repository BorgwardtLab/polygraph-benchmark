import os
from dataclasses import asdict
from typing import Dict

import joblib
import torch
from appdirs import user_cache_dir

from graph_gen_gym import __version__
from graph_gen_gym.datasets.graph_storage_dataset import GraphStorage


def _torch_to_hashable(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    if isinstance(data, dict):
        return {key: _torch_to_hashable(val) for key, val in data.items()}
    if isinstance(data, (tuple, list)):
        return tuple(_torch_to_hashable(item) for item in data)
    if data is None or isinstance(data, (int, float, str)):
        return data
    raise ValueError


def torch_hash(data):
    return joblib.hash(_torch_to_hashable(data))


def identifier_to_path(identifier: str):
    cache_dir = user_cache_dir(f"graph_gen_gym-{__version__}", "MPIB-MLSB")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, identifier)


def write_splits_to_cache(data: Dict[str, GraphStorage], identifier: str):
    path = identifier_to_path(identifier)
    if os.path.exists(path):
        raise FileExistsError(f"Tried to write data to {path}, but path already exists")
    os.makedirs(path, exist_ok=False)
    primitive_data = {key: value.model_dump() for key, value in data.items()}
    print(f"Saving data with hash {torch_hash(primitive_data)} to {path}")
    for key, tensors in primitive_data.items():
        torch.save(tensors, os.path.join(path, f"{key}.pt"))


def load_and_verify_splits(identifier: str, hash: str):
    path = identifier_to_path(identifier)
    fnames = list(os.listdir(path))
    if not os.path.exists(path) or len(fnames) == 0:
        raise FileNotFoundError
    result = {}
    for fname in fnames:
        split_name = ".".join(fname.split(".")[:-1])
        data = torch.load(os.path.join(path, fname), weights_only=True)
        result[split_name] = data
    return {key: GraphStorage(**val) for key, val in result.items()}
