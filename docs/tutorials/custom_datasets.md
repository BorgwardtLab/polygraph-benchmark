# Creating and Sharing Datasets

In this tutorial we discuss how you may create custom graph datasets and share them publicly.


## Creating Custom Datasets

While we provide the most commonly used datasets in graph generation (e.g., [`PlanarGraphDataset`][polygraph.datasets.PlanarGraphDataset]),
you may want to use your own custom data.

In this tutorial, we consider a toy example in which we create a dataset of 128 Erdos-Renyi graphs:
```python
import networkx as nx

er_samples = [nx.erdos_renyi_graph(64, 0.2) for _ in range(128)]
```
We may now construct a [`GraphStorage`][polygraph.datasets.GraphStorage] container from these NetworkX graphs. PolyGraph uses these containers
internally to store, serialize, and access graphs:
```python
from polygraph.datasets import GraphStorage
storage = GraphStorage.from_nx_graphs(er_samples)
```
If your graphs contain node, edge, or graph-level attributes, you may specify them by passing the keyword arguments `node_attrs`, `edge_attrs` and `graph_attrs` to [`GraphStorage.from_nx_graphs`][polygraph.datasets.GraphStorage.from_nx_graphs].

Alternatively, one may instantiate a [`GraphStorage`][polygraph.datasets.GraphStorage] from a PyTorch geometric `Batch` object like this:
```python
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx

storage = GraphStorage.from_pyg_batch(Batch.from_data_list([from_networkx(g) for g in er_samples]))
```

We may then wrap this `storage` object in a dataset which exposes the same indexing and `.to_nx()` functionalities as the datasets we have seen in previous tutorials:
```python
from polygraph.datasets import GraphDataset

ds = GraphDataset(storage)
print(ds[0])                # PyG Data(edge_index=[2, 808], num_nodes=64)
print(ds.to_nx()[0])        # Networkx Graph with 64 nodes and 404 edges
```

To store and later load the same data, the [`GraphDataset`][polygraph.datasets.GraphDataset] class provides the `dump_data` and `load_data` methods:
```python
ds.dump_data("/tmp/my_dataset.pt")
ds2 = GraphDataset.load_data("/tmp/my_dataset.pt", memmap=True)
```

## Sharing Datasets

Datasets can be shared conveniently via the data that is produced by [`GraphDataset.dump_data`][polygraph.datasets.GraphDataset.dump_data].
Simply upload the file to a digital repository like [Zenodo](https://zenodo.org/) and share the link with other users.
They may then instantiate the dataset like this:
```python
from polygraph.datasets import URLGraphDataset

ds = URLGraphDataset(
    url="https://sandbox.zenodo.org/records/309573/files/planar_train.pt?download=1",
    file_hash="edc2630954a23b1cf6a549d43a95e359"
)
```
If no md5 hash of the file is available, it may be set to `None`.

!!!warning
    Only download datasets from trusted sources!
