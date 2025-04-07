"""MMD metrics based on graph descriptors introduced by You et al. [1] and Liao et al. [2].

We provide both point estimates of MMD and uncertainty quantifications. Combinations of the following graph descriptors and kernels are available:

Graph Descriptors:
    - [`OrbitCounts`][polygrapher.utils.graph_descriptors.OrbitCounts]: Counts of different graphlet orbits
    - [`ClusteringHistogram`][polygrapher.utils.graph_descriptors.ClusteringHistogram]: Distribution of clustering coefficients
    - [`SparseDegreeHistogram`][polygrapher.utils.graph_descriptors.SparseDegreeHistogram]: Distribution of node degrees
    - [`EigenvalueHistogram`][polygrapher.utils.graph_descriptors.EigenvalueHistogram]: Distribution of graph Laplacian eigenvalues

Available Kernels:
    - [`GaussianTV`][polygrapher.utils.kernels.GaussianTV]: Gaussian Total Variation kernel with fixed bandwidth
    - [`LinearKernel`][polygrapher.utils.kernels.LinearKernel]: Simple linear (dot product) kernel
    - [`AdaptiveRBFKernel`][polygrapher.utils.kernels.AdaptiveRBFKernel]: Radial Basis Function kernel with multiple bandwidths, as proposed by Thompson et al. [4]


The Gaussian TV kernel, introduced by Liao et al. [2], is most widely used in the literature. Here, we refer to the resulting MMD metrics as GRAN-MMD (e.g. [`GRANOrbitMMD2`][polygrapher.metrics.gran.GRANOrbitMMD2]).

Warning:
    The Gaussian TV kernel is not positive definite, as shown by O'Bray et al. [3]. While it is most widely used in the literature, consider also evaluating the linear and RBF kernels.
    

References:
    [1] You, J., Ying, R., Ren, X., Hamilton, W., & Leskovec, J. (2018). [GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models](https://arxiv.org/abs/1802.08773). In International Conference on Machine Learning (ICML).

    [2] Liao, R., Li, Y., Song, Y., Wang, S., Hamilton, W., Duvenaud, D., Urtasun, R., & Zemel, R. (2019). [Efficient Graph Generation with Graph Recurrent Attention Networks](https://arxiv.org/abs/1910.00760). In Advances in Neural Information Processing Systems (NeurIPS).

    [3] O'Bray, L., Horn, M., Rieck, B., & Borgwardt, K. (2022). [Evaluation Metrics for Graph Generative Models: Problems, Pitfalls, and Practical Solutions](https://arxiv.org/abs/2106.01098). In International Conference on Learning Representations (ICLR).

    [4] Thompson, R., Knyazev, B., Ghalebi, E., Kim, J., & Taylor, G. W. (2022). [On Evaluation Metrics for Graph Generative Models](https://arxiv.org/abs/2201.09871). In International Conference on Learning Representations (ICLR).
"""

from polygrapher.metrics.gran.gaussian_tv_mmd import *  # noqa
from polygrapher.metrics.gran.rbf_mmd import *  # noqa
from polygrapher.metrics.gran.linear_mmd import *  # noqa