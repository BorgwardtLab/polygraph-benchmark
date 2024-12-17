import os
import urllib

import torch
from appdirs import user_cache_dir

from graph_gen_gym import __version__
from graph_gen_gym.datasets.base.graph import Graph


def identifier_to_path(identifier: str):
    cache_dir = os.environ.get("GRAPH_GEN_GYM_CACHE_DIR")
    if cache_dir is None:
        cache_dir = user_cache_dir(f"graph_gen_gym-{__version__}", "MPIB-MLSB")
    else:
        cache_dir = os.path.join(cache_dir, str(__version__))
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, identifier)


def download_to_cache(url: str, identifier: str, split: str = "data"):
    path = identifier_to_path(identifier)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f"{split}.pt")
    if os.path.exists(path):
        raise FileExistsError(
            f"Tried to download data to {path}, but path already exists"
        )
    urllib.request.urlretrieve(url, path)


def write_to_cache(data: Graph, identifier: str, split: str = "data"):
    path = identifier_to_path(identifier)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(f"{split}.pt")
    if os.path.exists(path):
        raise FileExistsError(f"Tried to write data to {path}, but path already exists")
    primitive_data = data.model_dump()
    torch.save(primitive_data, path)


def load_from_cache(identifier: str, split: str = "data", mmap: bool = False) -> Graph:
    path = os.path.join(identifier_to_path(identifier), f"{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError
    data = torch.load(path, weights_only=True, mmap=mmap)
    return Graph(**data)
