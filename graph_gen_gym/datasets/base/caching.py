import os
import urllib
from typing import Any, Sequence, Optional

import torch
from appdirs import user_cache_dir
from loguru import logger

from graph_gen_gym import __version__
from graph_gen_gym.datasets.base.graph import Graph
import shutil
import hashlib
import filelock


def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        data_hash = hashlib.md5()
        while chunk := f.read(8192):
            data_hash.update(chunk)
    return data_hash.hexdigest()

def identifier_to_path(identifier: str):
    cache_dir = os.environ.get("GRAPH_GEN_GYM_CACHE_DIR")
    if cache_dir is None:
        cache_dir = user_cache_dir(f"graph_gen_gym-{__version__}", "MPIB-MLSB")
    else:
        cache_dir = os.path.join(cache_dir, str(__version__))
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, identifier)


def clear_cache(identifier: str):
    path = identifier_to_path(identifier)
    shutil.rmtree(path)


def download_to_cache(url: str, identifier: str, split: str = "data"):
    path = identifier_to_path(identifier)
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{split}.pt")
    lock_path = file_path + ".lock"

    with filelock.FileLock(lock_path):
        if os.path.exists(file_path):
            logger.debug(f"Couldn't download data to {file_path} because it already exists")
            raise FileExistsError(
                f"Tried to download data to {file_path}, but path already exists"
            )
        logger.debug(f"Downloading data to {file_path}")
        urllib.request.urlretrieve(url, file_path)


def write_to_cache(identifier: str, split: str, data: Graph):
    path = identifier_to_path(identifier)
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{split}.pt")
    lock_path = file_path + ".lock"

    with filelock.FileLock(lock_path):
        logger.debug(f"Writing data to {file_path}")
        torch.save(data.model_dump(), file_path)

def load_from_cache(identifier: str, split: str = "data", mmap: bool = False, data_hash: Optional[str] = None) -> Graph:
    file_path = os.path.join(identifier_to_path(identifier), f"{split}.pt")
    lock_path = file_path + ".lock"

    with filelock.FileLock(lock_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError
        if data_hash is not None and file_hash(file_path) != data_hash:
            raise ValueError(f"Hash mismatch for {file_path}. Expected {data_hash}, got {file_hash(file_path)}")

        logger.debug(f"Loading data from {file_path}")
        data = torch.load(file_path, weights_only=True, mmap=mmap)
        return Graph(**data)


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]
