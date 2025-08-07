# Metrics API

::: polygraph.metrics.base.interface
    options:
        show_root_heading: false
        show_root_toc_entry: false
        show_source: false
        show_full_path: false
        heading_level: 2
        members: []

We implement the following metrics:

- [MMD](mmd.md) - Classical Maximum Mean Discrepancy
- [PolyGraphScore](polygraphscore.md) - Lower bounds on probability metrics via classification
- [VUN](../../metrics/vun.md) - Validity, Uniqueness, Novelty
- [Fréchet Distance](frechet.md) - Optimal transport distance between fitted Gaussians

## Interface Protocols

::: polygraph.metrics.base.GenerationMetric
    options:
        show_root_heading: true
        show_full_path: true
        show_source: false
        heading_level: 3


## Metric Collections

::: polygraph.metrics.base.MetricCollection
    options:
        show_root_heading: true
        show_full_path: true
        show_source: false
        heading_level: 3
