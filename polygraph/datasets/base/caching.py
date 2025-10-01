import hashlib
import os
import shutil
import urllib.request
from typing import Any, Optional, Sequence

import filelock
import torch
from appdirs import user_cache_dir
from loguru import logger

from polygraph import __version__
from polygraph.datasets.base.graph_storage import GraphStorage


def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        data_hash = hashlib.md5()
        while chunk := f.read(8192):
            data_hash.update(chunk)
    return data_hash.hexdigest()


def identifier_to_path(identifier: str):
    cache_dir = os.environ.get("POLYGRAPH_CACHE_DIR")
    if cache_dir is None:
        cache_dir = user_cache_dir(f"polygraph-{__version__}", "MPIB-MLSB")
    else:
        cache_dir = os.path.join(cache_dir, str(__version__))
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, identifier)


def clear_cache(identifier: str):
    path = identifier_to_path(identifier)
    shutil.rmtree(path)


class CacheLock:
    def __init__(self, identifier: str):
        lock_path = identifier_to_path(identifier) + ".lock"
        self._lock = filelock.FileLock(lock_path)

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()


def download_to_cache(url: str, identifier: str, split: str = "data"):
    path = identifier_to_path(identifier)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f"{split}.pt")
    if os.path.exists(path):
        raise FileExistsError(
            f"Tried to download data to {path}, but path already exists"
        )
    logger.debug(f"Downloading data to {path}")
    urllib.request.urlretrieve(url, path)


def write_to_cache(identifier: str, split: str, data: GraphStorage):
    path = identifier_to_path(identifier)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f"{split}.pt")
    logger.debug(f"Writing data to {path}")
    torch.save(data.model_dump(), path)


def load_from_cache(
    identifier: str,
    split: str = "data",
    mmap: bool = False,
    data_hash: Optional[str] = None,
) -> GraphStorage:
    path = os.path.join(identifier_to_path(identifier), f"{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError
    computed_file_hash = file_hash(path)
    if data_hash is not None and computed_file_hash != data_hash:
        raise ValueError(
            f"Hash mismatch for {path}. Expected {data_hash}, got {computed_file_hash}"
        )

    logger.debug(f"Loading data from {path}")
    data = torch.load(path, weights_only=True, mmap=mmap)
    return GraphStorage(**data)


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]
