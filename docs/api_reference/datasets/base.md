# Dataset Abstractions

## Storing and Serializing Graphs
::: polygraph.datasets.base.graph_storage
    options:
        show_root_heading: false
        show_root_toc_entry: false
        show_source: false
        show_full_path: false
        heading_level: 3
        members: []


::: polygraph.datasets.base.GraphStorage
    options:
        show_root_heading: true
        show_root_toc_entry: false
        show_source: false
        heading_level: 3
        members: [get_example, __len__]

## Dataset Base Class
::: polygraph.datasets.base.dataset
    options:
        show_root_heading: false
        show_root_toc_entry: false
        show_source: false
        show_full_path: false
        heading_level: 3
        members: []

::: polygraph.datasets.base.AbstractDataset
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
        members: [__getitem__, __len__, to_nx, is_valid]

::: polygraph.datasets.base.GraphDataset
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3

::: polygraph.datasets.base.OnlineGraphDataset
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
        members: [url_for_split, hash_for_split]

::: polygraph.datasets.base.ProceduralGraphDataset
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3

## Accessing Graphs as NetworkX Objects

The datasets discussed above can be converted into datasets of NetworkX graphs
by calling the [`to_nx`][polygraph.datasets.base.AbstractDataset.to_nx] method.
This method returns a [`NetworkXView`][polygraph.datasets.base.NetworkXView] object.

::: polygraph.datasets.base.NetworkXView
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3