"""Metrics based on embeddings of random GIN models, as proposed by Thompson et al. [1].

This module provides implementations of metrics for evaluating graph generative models using embeddings from untrained (random) Graph Isomorphism Networks (GINs).


Available Metrics:
    - [`RBFGraphNeuralNetworkMMD2`][polygrapher.metrics.gin.RBFGraphNeuralNetworkMMD2]: MMD using an adaptive RBF kernel with multiple bandwidths
    - [`LinearGraphNeuralNetworkMMD2`][polygrapher.metrics.gin.LinearGraphNeuralNetworkMMD2]: MMD using a linear kernel
    - [`GraphNeuralNetworkFrechetDistance`][polygrapher.metrics.gin.GraphNeuralNetworkFrechetDistance]: Fréchet Distance in GIN embedding space

References:
    [1] Thompson, R., Knyazev, B., Ghalebi, E., Kim, J., & Taylor, G. W. (2022). [On Evaluation Metrics for Graph Generative Models](https://arxiv.org/abs/2201.09871). In International Conference on Learning Representations (ICLR).
"""

from polygrapher.metrics.gin.frechet_distance import *  # noqa
from polygrapher.metrics.gin.mmd import *  # noqa
