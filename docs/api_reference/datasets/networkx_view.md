# Accessing Graphs as NetworkX Objects

The datasets discussed above can be converted into datasets of NetworkX graphs
by calling the [`to_nx`][polygraph.datasets.base.AbstractDataset.to_nx] method.
This method returns a [`NetworkXView`][polygraph.datasets.base.NetworkXView] object.

::: polygraph.datasets.base.NetworkXView
    options:
        show_root_heading: true
        show_source: false
        heading_level: 2
        members: [__len__, __getitem__]
