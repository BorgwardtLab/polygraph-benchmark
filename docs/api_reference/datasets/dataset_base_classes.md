# Dataset Base Class
::: polygraph.datasets.base.dataset
    options:
        show_root_heading: false
        show_root_toc_entry: false
        show_source: false
        show_full_path: false
        heading_level: 2
        members: []

::: polygraph.datasets.base.AbstractDataset
    options:
        show_root_heading: true
        show_source: false
        heading_level: 2
        members: [__getitem__, __len__, to_nx, is_valid]

::: polygraph.datasets.base.GraphDataset
    options:
        show_root_heading: true
        show_source: false
        heading_level: 2

::: polygraph.datasets.base.URLGraphDataset
    options:
        show_root_heading: true
        show_source: false
        heading_level: 2
        members: [url_for_split, hash_for_split]

::: polygraph.datasets.base.SplitGraphDataset
    options:
        show_root_heading: true
        show_source: false
        heading_level: 2
        members: [url_for_split, hash_for_split]

::: polygraph.datasets.base.ProceduralGraphDataset
    options:
        show_root_heading: true
        show_source: false
        heading_level: 2
