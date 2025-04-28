import json
import torch
from polygraph.datasets.rna import dotbracket_to_graph
from polygraph.datasets.base import GraphStorage
from torch_geometric.data import Batch


def to_storage(structure_list):
    data_list = [
        dotbracket_to_graph(data["sequence"], data["structure"])
        for seq_id, data in structure_list
    ]
    for (id, data), g in zip(structure_list, data_list):
        g.free_energy = data["energy"]

    batch = Batch.from_data_list(data_list)
    batch.nucleotides = batch.x
    batch.bond_types = batch.edge_attr
    return GraphStorage.from_pyg_batch(
        batch,
        node_attrs=["nucleotides"],
        edge_attrs=["bond_types"],
        graph_attrs=["free_energy"],
    )


if __name__ == "__main__":
    with open("combined_structures.json", "r") as f:
        data = json.load(f)

    data = [(key, value) for key, value in data.items()]

    test_len = int(round(len(data) * 0.2))
    train_len = int(round((len(data) - test_len) * 0.8))
    val_len = len(data) - train_len - test_len
    train, val, test = torch.utils.data.random_split(
        data,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(1234),
    )

    for name, split in [("train", train), ("val", val), ("test", test)]:
        storage = to_storage(split)
        torch.save(storage.model_dump(), f"{name}.pt")
