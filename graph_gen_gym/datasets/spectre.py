import os
import tempfile
import urllib
import warnings
from collections import defaultdict
from typing import List

import networkx as nx
import numpy as np
import torch
import torch_geometric
from scipy.stats import chi2
from torch_geometric.data import Batch, Data

from graph_gen_gym.datasets.graph_storage_dataset import (
    GraphStorage,
    GraphStorageDataset,
)
from graph_gen_gym.datasets.utils import load_and_verify_splits, write_splits_to_cache


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
        key: GraphStorage.from_pyg_batch(
            Batch.from_data_list(lst), compute_indexing_info=True
        )
        for key, lst in data_lists.items()
    }


class _SpectreDataset(GraphStorageDataset):
    def __init__(self, split: str):
        try:
            whole_data = load_and_verify_splits(self.identifier, self.hash)
        except FileNotFoundError:
            warnings.warn("Downloading dataset...")
            whole_data = _spectre_link_to_storage(self.url)
            write_splits_to_cache(whole_data, self.identifier)

        super().__init__(data_store=whole_data[split])


class PlanarGraphDataset(_SpectreDataset):
    @property
    def url(self):
        return "https://github.com/KarolisMart/SPECTRE/raw/refs/heads/main/data/planar_64_200.pt"

    @property
    def hash(self):
        return "c9a96c31fb513c15bc42c6a0b01d830e"

    def is_valid(self, graph):
        if isinstance(graph, nx.Graph):
            return nx.is_connected(graph) and nx.is_planar(graph)
        raise TypeError

    def sample(self, n_samples: int, replace: bool = False) -> List[GraphStorage]:
        idx_to_sample = np.random.choice(len(self), n_samples, replace=replace)
        return self[idx_to_sample]


class SBMGraphDataset(_SpectreDataset):
    @property
    def url(self):
        return (
            "https://github.com/KarolisMart/SPECTRE/raw/refs/heads/main/data/sbm_200.pt"
        )

    @property
    def hash(self):
        return "2a1b79e12163d3ea26d49db41e7822ff"

    def is_valid(self, graph):
        import graph_tool.all as gt

        adj = nx.adjacency_matrix(graph).toarray()
        idx = adj.nonzero()
        g = gt.Graph()
        g.add_edge_list(np.transpose(idx))
        try:
            state = gt.minimize_blockmodel_dl(g)
        except ValueError:
            return False

        # Refine using merge-split MCMC
        for i in range(100):
            state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

        b = state.get_blocks()
        b = gt.contiguous_map(state.get_blocks())
        state = state.copy(b=b)
        e = state.get_matrix()
        n_blocks = state.get_nonempty_B()
        node_counts = state.get_nr().get_array()[:n_blocks]
        edge_counts = e.todense()[:n_blocks, :n_blocks]
        if (
            (node_counts > 40).sum() > 0
            or (node_counts < 20).sum() > 0
            or n_blocks > 5
            or n_blocks < 2
        ):
            return False

        max_intra_edges = node_counts * (node_counts - 1)
        est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

        max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
        np.fill_diagonal(edge_counts, 0)
        est_p_inter = edge_counts / (max_inter_edges + 1e-6)

        W_p_intra = (est_p_intra - 0.3) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
        W_p_inter = (est_p_inter - 0.005) ** 2 / (
            est_p_inter * (1 - est_p_inter) + 1e-6
        )

        W = W_p_inter.copy()
        np.fill_diagonal(W, W_p_intra)
        p = 1 - chi2.cdf(abs(W), 1)
        p = p.mean()
        return p > 0.9  # p value < 10 %


class ProteinGraphDataset(GraphStorageDataset):
    @property
    def url(self):
        pass
