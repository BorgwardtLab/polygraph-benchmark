# Polygraph
# Installation
For now, you can install the package in editable mode
```bash
mamba create -n polygrapher python=3.12
pip install -e ".[dev]"
mamba activate polygrapher
```

This will also install orca and (unpinned) dependencies.

# Loading Datasets
Currently, we can load planar and SBM graphs. Data is stored in your `.cache` folder.
```python
from polygrapher.datasets.spectre import PlanarGraphDataset, SBMGraphDataset


ds_planar = PlanarGraphDataset("train")        # Indexable and iterable, gives PyG graphs
ds_sbm = SBMGraphDataset("train")
nx_view = ds.to_nx()                    # Indexable and iterable, gives networkx graphs
```


# Computing MMDs
The standard MMDs can be computed like this:

```python
from polygrapher.metrics.mmd.mmd import OrbitMM2

mmd = OrbitMM2(ds_planar)
print(mmd.compute(ds_sbm.to_nx()))
```

Note that we provide the reference graphs (`ds_planar`) as a dataset while we provide the samples of the generative model as an iterable of nx graphs (we might want to change that but not sure).

We can compute arbitrary MMDs on graph descriptors via the `DescriptorMMD2` class:

```python
from polygrapher.metrics.mmd.mmd import DescriptorMMD2
from polygrapher.metrics.graph_descriptors import orbit_descriptor
from polygrapher.metrics.mmd.kernels import LaplaceKernel

mmd = DescriptorMMD2(ds_planar, descriptor_fn=orbit_descriptor, kernel=LaplaceKernel(lbd=0.2), variant="umve")
print(mmd.compute(ds_sbm.to_nx()))
```

You may also compute the MMD for different kernel hyper-parameter choices in a vectorized fashion by passing an array of hyperparameters to `LaplaceKernel`:

```python
import numpy as np
mmd = DescriptorMMD2(ds_planar, descriptor_fn=orbit_descriptor, kernel=LaplaceKernel(lbd=np.linspace(0.1, 5, 10)), variant="umve")
print(mmd.compute(ds_sbm.to_nx()))     # Gives a numpy array
```


## Computing p-Values

```python
from polygrapher.metrics.mmd.tests import OptimizedPValue
from polygrapher.metrics.graph_descriptors import clustering_descriptor
from functools import partial

planar_val = PlanarGraphDataset("val")
sbm_val = SBMGraphDataset("val")

tst = OptimizedPValue(ds_planar, planar_val, descriptor_fn=partial(clustering_descriptor, bins=100), kernel=LaplaceKernel(lbd=np.linspace(0.05, 5, 100)))
print(tst.compute(ds_sbm.to_nx(), sbm_val.to_nx()))
```
