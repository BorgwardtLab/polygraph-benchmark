import os
import tempfile
import urllib
from collections import defaultdict

import torch
import torch_geometric
from torch_geometric.data import Batch, Data

from polygraph.datasets.base import GraphStorage


def _spectre_link_to_storage(url):
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "data.pt")
        urllib.request.urlretrieve(url, fpath)
        adjs, _, _, _, _, _, _, _ = torch.load(fpath, weights_only=True)
    assert isinstance(adjs, list)
    test_len = int(round(len(adjs) * 0.2))
    train_len = int(round((len(adjs) - test_len) * 0.8))
    val_len = len(adjs) - train_len - test_len
    train, val, test = torch.utils.data.random_split(
        adjs,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(1234),
    )
    split_adjs = {"train": train, "val": val, "test": test}
    data_lists = defaultdict(list)
    for split, adjs in split_adjs.items():
        for adj in adjs:
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            data_lists[split].append(Data(edge_index=edge_index, num_nodes=len(adj)))

    return {
        key: GraphStorage.from_pyg_batch(Batch.from_data_list(lst))
        for key, lst in data_lists.items()
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--destination", type=str, required=True)
    args = parser.parse_args()

    urls = {
        "planar": "https://github.com/KarolisMart/SPECTRE/raw/refs/heads/main/data/planar_64_200.pt",
        "sbm": "https://github.com/KarolisMart/SPECTRE/raw/refs/heads/main/data/sbm_200.pt",
    }

    url = urls[args.dataset]
    whole_data = _spectre_link_to_storage(url)
    for key, val in whole_data.items():
        torch.save(val.model_dump(), os.path.join(args.destination, f"{key}.pt"))
