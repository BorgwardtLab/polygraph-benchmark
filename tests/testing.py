import numpy as np
from graph_gen_gym.datasets.spectre import PlanarGraphDataset, SBMGraphDataset
from graph_gen_gym.metrics.mmd import DescriptorMMD2, MaxDescriptorMMD2
from graph_gen_gym.metrics.mmd import OrbitCounts, DegreeHistogram, ClusteringHistogram
from graph_gen_gym.metrics.mmd import RBFKernel, LinearKernel, LaplaceKernel, StackedKernel
from graph_gen_gym.metrics.mmd import BootStrapMMDTest
from graph_gen_gym.metrics.mmd import ClassifierTest


planar_train = PlanarGraphDataset("train")        # Indexable and iterable, gives PyG graphs
sbm_train = SBMGraphDataset("train")

# Stack multiple kernels to get one `combined_kernel` covering 201 different descriptor x kernel combinations
kernel1 = RBFKernel(OrbitCounts(), bw=np.linspace(0.01, 20, 100))
kernel2 = LinearKernel(DegreeHistogram(max_degree=200))
kernel3 = LaplaceKernel(ClusteringHistogram(bins=100), lbd=np.linspace(0.01, 20, 100))
combined_kernel = StackedKernel([kernel1, kernel2, kernel3])
print(combined_kernel.num_kernels)

# Compute MMD for all combinations in parallel
mmd = DescriptorMMD2(sbm_train.to_nx(), combined_kernel, variant="umve")
print(mmd.compute(planar_train.to_nx()))

# Compute maximal MMD across combinations (and the corresponding kernel)
max_mmd = MaxDescriptorMMD2(sbm_train.to_nx(), combined_kernel, "umve")
metric, kernel = max_mmd.compute(planar_train.to_nx())

# Perform a two-sample test with linear kernel (compute p-value)
tst = BootStrapMMDTest(sbm_train.to_nx(), kernel2)
print(tst.compute(planar_train.to_nx()))

# Compute the classifier statistic (with some uncertainty quantification and p-value)
tst = ClassifierTest(sbm_train.to_nx(), combined_kernel)
print(tst.compute(planar_train.to_nx()))