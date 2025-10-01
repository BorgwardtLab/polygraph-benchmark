import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, LargestConnectedComponents
from torch_geometric.utils import coalesce
from torch_geometric.data import Data, Batch
import numpy as np
from tqdm import tqdm
import argparse
import os

from polygraph.datasets.base import GraphStorage


def meshes_to_graphs(ds, rng, k=4):
    graphs = []
    lcc_transform = LargestConnectedComponents()

    for data in tqdm(ds):
        transform = SamplePoints(
            num=min(data.face.shape[1], rng.integers(2000, 5000))
        )
        data = transform(data)

        dist = torch.cdist(data.pos, data.pos)
        dist.diagonal(dim1=0, dim2=1).fill_(torch.inf)
        result = torch.topk(dist, k=k, dim=1, largest=False)
        left = torch.repeat_interleave(torch.arange(dist.shape[0]), k)
        right = result.indices.flatten()
        edge_index = torch.stack([left, right], dim=0)
        edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
        edge_index = coalesce(edge_index)
        new_graph = Data(
            edge_index=edge_index,
            pos=data.pos,
            num_nodes=data.num_nodes,
            object_class=data.y,
        )
        new_graph = lcc_transform(new_graph)
        graphs.append(new_graph)
    return graphs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=str, required=True)
    parser.add_argument(
        "--dataset",
        choices=["modelnet10", "modelnet40"],
        default="modelnet10",
        help="Choose between ModelNet10 and ModelNet40 datasets",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    dataset_name = "10" if args.dataset == "modelnet10" else "40"
    data_root = f"./data/{args.dataset}"

    train_ds = ModelNet(
        root=data_root,
        name=dataset_name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        train=True,
    )
    test_ds = ModelNet(
        root=data_root,
        name=dataset_name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        train=False,
    )

    train_graphs = meshes_to_graphs(train_ds, rng)
    test_graphs = meshes_to_graphs(test_ds, rng)
    val_size = int(0.1 * len(train_graphs))
    val_indices = rng.choice(len(train_graphs), size=val_size, replace=False)
    train_indices = np.setdiff1d(np.arange(len(train_graphs)), val_indices)
    tmp = [train_graphs[i] for i in train_indices]
    val_graphs = [train_graphs[i] for i in val_indices]
    train_graphs = tmp

    train_storage = GraphStorage.from_pyg_batch(
        Batch.from_data_list(train_graphs),
        node_attrs=["pos"],
        graph_attrs=["object_class"],
    )
    val_storage = GraphStorage.from_pyg_batch(
        Batch.from_data_list(val_graphs),
        node_attrs=["pos"],
        graph_attrs=["object_class"],
    )
    test_storage = GraphStorage.from_pyg_batch(
        Batch.from_data_list(test_graphs),
        node_attrs=["pos"],
        graph_attrs=["object_class"],
    )

    os.makedirs(args.destination, exist_ok=True)
    torch.save(
        train_storage.model_dump(), os.path.join(args.destination, "train.pt")
    )
    torch.save(
        val_storage.model_dump(), os.path.join(args.destination, "val.pt")
    )
    torch.save(
        test_storage.model_dump(), os.path.join(args.destination, "test.pt")
    )
