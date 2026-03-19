<p align="center">
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo/logo_icon_Dark_NordDark.png">
  <source media="(prefers-color-scheme: light)" srcset="logo/logo_icon_Light_NordLight.png">
  <img src="https://raw.githubusercontent.com/BorgwardtLab/polygraph-benchmark/refs/heads/master/logo/logo_Light_NordLight.png" alt="PolyGraph icon" height="128">
  </picture>
  <br>
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo/logo_Dark_NordDark.png">
  <source media="(prefers-color-scheme: light)" srcset="logo/logo_Light_NordLight.png">
  <img src="https://raw.githubusercontent.com/BorgwardtLab/polygraph-benchmark/refs/heads/master/logo/logo_icon_Light_NordLight.png" alt="PolyGraph logo" height="100">
  </picture>
</p>

PolyGraph is a Python library for evaluating graph generative models by providing standardized datasets and metrics
(including PolyGraph Discrepancy). Full documentation for this library can be found [here](https://polygraph-benchmark.readthedocs.io/).

PolyGraph discrepancy is a new metric we introduced, which provides the following advantages over maxmimum mean discrepancy (MMD):

<div align="center">
<table>
<thead>
<tr>
  <th>Property</th>
  <th>MMD</th>
  <th>PGD</th>
</tr>
</thead>
<tbody>
<tr>
  <td>Range</td>
  <td>[0, ∞)</td>
  <td>[0, 1]</td>
</tr>
<tr>
  <td>Intrinsic Scale</td>
  <td style="color:red;">❌</td>
  <td style="color:green;">✅</td>
</tr>
<tr>
  <td>Descriptor Comparison</td>
  <td style="color:red;">❌</td>
  <td style="color:green;">✅</td>
</tr>
<tr>
  <td>Multi-Descriptor Aggregation</td>
  <td style="color:red;">❌</td>
  <td style="color:green;">✅</td>
</tr>
<tr>
  <td>Single Ranking</td>
  <td style="color:red;">❌</td>
  <td style="color:green;">✅</td>
</tr>
</tbody>
</table>
</div>

It also provides a number of other advantages over MMD which we discuss in [our paper](https://arxiv.org/abs/2510.06122).

## Installation

```bash
pip install polygraph-benchmark
```

No manual compilation of ORCA is required. For details on interaction with `graph_tool`, see the more detailed installation instructions in the docs.

If you'd like to use SBM graph dataset validation with graph tools, use a mamba or pixi environment. More information is available in the documentation.

## At a glance

Here are a set of datasets and metrics this library provides:
- 🗂️ **Datasets**: ready-to-use splits for procedural and real-world graphs
  - Procedural datasets: `PlanarLGraphDataset`, `SBMLGraphDataset`, `LobsterLGraphDataset`
  - Real-world: `QM9`, `MOSES`, `Guacamol`, `DobsonDoigGraphDataset`, `ModelNet10GraphDataset`
  - Also: `EgoGraphDataset`, `PointCloudGraphDataset`
- 📊 **Metrics**: unified, fit-once/compute-many interface with convenience wrappers, avoiding redundant computations.
  - MMD<sup>2</sup>: `GaussianTVMMD2Benchmark`, `RBFMMD2Benchmark`
  - Kernel hyperparameter optimization with `MaxDescriptorMMD2`.
  - PolyGraphDiscrepancy: `StandardPGD`, `MolecularPGD` (for molecule descriptors).
  - Validation/Uniqueness/Novelty: `VUN`.
  - Uncertainty quantification for benchmarking (`GaussianTVMMD2BenchmarkInterval`, `RBFMMD2Benchmark`, `PGD5Interval`)
- 🧩 **Extendable**: Users can instantiate custom metrics by specifying descriptors, kernels, or classifiers (`PolyGraphDiscrepancy`, `DescriptorMMD2`). PolyGraph defines all necessary interfaces but imposes no requirements on the data type of graph objects.
- ⚙️ **Interoperability**: Works on Apple Silicon Macs and Linux.
- ✅ **Tested, type checked and documented**

<details>
<summary><strong>⚠️ Important - Dataset Usage Warning</strong></summary>

**To help reproduce previous results, we provide the following datasets:**
- `PlanarGraphDataset`
- `SBMGraphDataset`
- `LobsterGraphDataset`

But they should not be used for benchmarking, due to unreliable metric estimates (see [our paper](https://arxiv.org/abs/2510.06122) for more details).

We provide larger datasets that should be used instead:
- `PlanarLGraphDataset`
- `SBMLGraphDataset`
- `LobsterLGraphDataset`

</details>

## Tutorial

Our [demo script](polygraph_demo.py) showcases some features of our library in action.

### Datasets
Instantiate a benchmark dataset as follows:
```python
import networkx as nx
from polygraph.datasets import PlanarGraphDataset

reference = PlanarGraphDataset("test").to_nx()

# Let's also generate some graphs coming from another distribution.
generated = [nx.erdos_renyi_graph(64, 0.1) for _ in range(40)]
```

### Metrics

#### Maximum Mean Discrepancy
To compute existing MMD2 formulations (e.g. based on the TV pseudokernel), one can use the following:
```python
from polygraph.metrics import GaussianTVMMD2Benchmark # Can also be RBFMMD2Benchmark

gtv_benchmark = GaussianTVMMD2Benchmark(reference)

print(gtv_benchmark.compute(generated))  # {'orbit': ..., 'clustering': ..., 'degree': ..., 'spectral': ...}
```

#### PolyGraphDiscrepancy
Similarly, you can compute our proposed PolyGraphDiscrepancy, like so:

```python
from polygraph.metrics import StandardPGD

pgd = StandardPGD(reference)
print(pgd.compute(generated)) # {'pgd': ..., 'pgd_descriptor': ..., 'subscores': {'orbit': ..., }}
```

`pgd_descriptor` provides the best descriptor used to report the final score.

#### Validity, uniqueness and novelty
VUN values follow a similar interface:
```python
from polygraph.metrics import VUN
reference_ds = PlanarGraphDataset("test")
pgd = VUN(reference, validity_fn=reference_ds.is_valid, confidence_level=0.95) # if applicable, validity functions are defined as a dataset attribute
print(pgd.compute(generated))  # {'valid': ..., 'valid_unique_novel': ..., 'valid_novel': ..., 'valid_unique': ...}
```

#### Metric uncertainty quantification

For MMD and PGD, uncertainty quantifiation for the metrics are obtained through subsampling. For VUN, a confidence interval is obtained with a binomial test.

For `VUN`, the results can be obtained by specifying a confidence level when instantiating the metric.

For the others, the `Interval` suffix references the class that implements subsampling.

```python
from polygraph.metrics import GaussianTVMMD2BenchmarkInterval, RBFMMD2BenchmarkInterval, StandardPGDInterval
from tqdm import tqdm

metrics = [
  GaussianTVMMD2BenchmarkInterval(reference, subsample_size=8, num_samples=10), # specify size of each subsample, and the number of samples
  RBFMMD2BenchmarkInterval(reference, subsample_size=8, num_samples=10),
  StandardPGDInterval(reference, subsample_size=8, num_samples=10)
]

for metric in tqdm(metrics):
	metric_results = metric.compute(
    generated,
  )
```
## Example Benchmark

The following results mirror the tables from [our paper](https://arxiv.org/abs/2510.06122). Bold indicates best, and underlined indicates second-best. Values are multiplied by 100 for legibility. Standard deviations are obtained with subsampling using `StandardPGDInterval` and `MoleculePGDInterval`. Specific parameters are discussed in [the paper](https://arxiv.org/abs/2510.06122).

<div align="center">
<table>
  <thead>
    <tr>
      <th>Method</th>
      <th style="text-align:right;">Planar-L</th>
      <th style="text-align:right;">Lobster-L</th>
      <th style="text-align:right;">SBM-L</th>
      <th style="text-align:right;">Proteins</th>
      <th style="text-align:right;">Guacamol</th>
      <th style="text-align:right;">Moses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AutoGraph</td>
      <td style="text-align:right;"><strong>34.0 ± 1.8</strong></td>
      <td style="text-align:right;"><u>18.0 ± 1.6</u></td>
      <td style="text-align:right;"><strong>5.6 ± 1.5</strong></td>
      <td style="text-align:right;"><strong>67.7 ± 7.4</strong></td>
      <td style="text-align:right;"><u>22.9 ± 0.5</u></td>
      <td style="text-align:right;"><strong>29.6 ± 0.4</strong></td>
    </tr>
    <tr>
      <td>AutoGraph*</td>
      <td style="text-align:right;">—</td>
      <td style="text-align:right;">—</td>
      <td style="text-align:right;">—</td>
      <td style="text-align:right;">—</td>
      <td style="text-align:right;"><strong>10.4 ± 1.2</strong></td>
      <td style="text-align:right;">—</td>
    </tr>
    <tr>
      <td>DiGress</td>
      <td style="text-align:right;">45.2 ± 1.8</td>
      <td style="text-align:right;"><strong>3.2 ± 2.6</strong></td>
      <td style="text-align:right;"><u>17.4 ± 2.3</u></td>
      <td style="text-align:right;">88.1 ± 3.1</td>
      <td style="text-align:right;">32.7 ± 0.5</td>
      <td style="text-align:right;"><u>33.4 ± 0.5</u></td>
    </tr>
    <tr>
      <td>GRAN</td>
      <td style="text-align:right;">99.7 ± 0.2</td>
      <td style="text-align:right;">85.4 ± 0.5</td>
      <td style="text-align:right;">69.1 ± 1.4</td>
      <td style="text-align:right;">89.7 ± 2.7</td>
      <td style="text-align:right;">—</td>
      <td style="text-align:right;">—</td>
    </tr>
    <tr>
      <td>ESGG</td>
      <td style="text-align:right;"><u>45.0 ± 1.4</u></td>
      <td style="text-align:right;">69.9 ± 0.6</td>
      <td style="text-align:right;">99.4 ± 0.2</td>
      <td style="text-align:right;"><u>79.2 ± 4.3</u></td>
      <td style="text-align:right;">—</td>
      <td style="text-align:right;">—</td>
    </tr>
  </tbody>
  </table>
</div>

<sub>* AutoGraph* denotes a variant that leverages additional training heuristics as described in the [paper](https://arxiv.org/abs/2510.06122).</sub>

## Reproducibility

The [`reproducibility/`](reproducibility/) directory contains scripts to reproduce all tables and figures from the paper.

### Quick Start

```bash
# 1. Install dependencies
pixi install

# 2. Download the graph data (~3GB)
cd reproducibility
python download_data.py

# 3. Generate all tables and figures
make all
```

### Data Download

The generated graph data (~3GB) is hosted on [Proton Drive](https://drive.proton.me/urls/VM4NWYBQD0#3sqmZtmSgWTB). After downloading, extract to `data/polygraph_graphs/` in the repository root.

```bash
# Full dataset (required for complete reproducibility)
python download_data.py

# Small subset for testing/CI (~50 graphs per model)
python download_data.py --subset
```

Expected data structure after extraction:

```
data/polygraph_graphs/
├── AUTOGRAPH/
│   ├── planar.pkl
│   ├── lobster.pkl
│   ├── sbm.pkl
│   └── proteins.pkl
├── DIGRESS/
│   ├── planar.pkl
│   ├── lobster.pkl
│   ├── sbm.pkl
│   ├── proteins.pkl
│   ├── denoising-iterations/
│   │   └── {15,30,45,60,75,90}_steps.pkl
│   └── training-iterations/
│       └── {119,209,...,3479}_steps.pkl
├── ESGG/
│   └── *.pkl
├── GRAN/
│   └── *.pkl
└── molecule_eval/
    └── *.smiles
```

### Scripts Overview

#### Table Generation

| Script | Output | Description |
|--------|--------|-------------|
| `generate_benchmark_tables.py` | `tables/benchmark_results.tex` | Main PGD benchmark (Table 1) comparing AUTOGRAPH, DiGress, GRAN, ESGG |
| `generate_mmd_tables.py` | `tables/mmd_gtv.tex`, `tables/mmd_rbf_biased.tex` | MMD² metrics with GTV and RBF kernels |
| `generate_gklr_tables.py` | `tables/gklr.tex` | PGD with Kernel Logistic Regression using WL and SP kernels |
| `generate_concatenation_tables.py` | `tables/concatenation.tex` | Ablation comparing individual vs concatenated descriptors |

#### Figure Generation

| Script | Output | Description |
|--------|--------|-------------|
| `generate_subsampling_figures.py` | `figures/subsampling/` | Bias-variance tradeoff as function of sample size |
| `generate_perturbation_figures.py` | `figures/perturbation/` | Metric sensitivity to edge perturbations |
| `generate_model_quality_figures.py` | `figures/model_quality/` | PGD vs training/denoising steps for DiGress |
| `generate_phase_plot.py` | `figures/phase_plot/` | Training dynamics showing PGD vs VUN |

Each script can be run independently with `--subset` for quick testing:

```bash
# Tables (full computation)
python generate_benchmark_tables.py
python generate_mmd_tables.py
python generate_gklr_tables.py
python generate_concatenation_tables.py

# Tables (quick testing with --subset)
python generate_benchmark_tables.py --subset
python generate_mmd_tables.py --subset

# Figures (full computation)
python generate_subsampling_figures.py
python generate_perturbation_figures.py
python generate_model_quality_figures.py
python generate_phase_plot.py

# Figures (quick testing)
python generate_subsampling_figures.py --subset
python generate_perturbation_figures.py --subset
```

### Make Targets

```bash
make download        # Download full dataset (manual step required)
make download-subset # Create small subset for CI testing
make tables          # Generate all LaTeX tables
make figures         # Generate all figures
make all             # Generate everything
make tables-submit   # Submit table jobs to SLURM cluster
make tables-collect  # Collect results from completed SLURM jobs
make clean           # Remove generated outputs
make help            # Show available targets
```

### Hardware Requirements

- **Memory:** 16GB RAM recommended for full dataset
- **Storage:** ~4GB for data + outputs
- **Time:** Full generation takes ~2-4 hours on a modern CPU

The `--subset` flag uses ~50 graphs per model, runs in minutes, and verifies code correctness (results are not publication-quality).

### Cluster Submission

Table generation scripts support SLURM cluster submission via [submitit](https://github.com/facebookincubator/submitit). Install the cluster extras first:

```bash
pip install -e ".[cluster]"
```

SLURM parameters are configured in YAML files (see `reproducibility/configs/slurm_default.yaml`):

```yaml
slurm:
  partition: "cpu"
  timeout_min: 360
  cpus_per_task: 8
  mem_gb: 32
```

Submit jobs, then collect results after completion:

```bash
cd reproducibility

# Submit all table jobs to SLURM
python generate_benchmark_tables.py --slurm-config configs/slurm_default.yaml

# After jobs complete, collect results and generate tables
python generate_benchmark_tables.py --collect

# Or use Make targets
make tables-submit                                        # submit all
make tables-submit SLURM_CONFIG=configs/my_cluster.yaml   # custom config
make tables-collect                                       # collect all
```

Use `--local` with `--slurm-config` to test the submission pipeline in-process without SLURM.

### Troubleshooting

**Memory issues:** Use `--subset` flag for testing, process one dataset at a time, or increase system swap space.

**Missing data:** Verify `data/polygraph_graphs/` exists in repo root, run `python download_data.py` to check data status, or download manually from Proton Drive.

**TabPFN issues:** TabPFN is pinned to v2.0.0 for reproducibility: `pip install tabpfn==2.0.0`.

## Citing

To cite our paper:

```latex
@misc{krimmel2025polygraph,
  title={PolyGraph Discrepancy: a classifier-based metric for graph generation},
  author={Markus Krimmel and Philip Hartout and Karsten Borgwardt and Dexiong Chen},
  year={2025},
  eprint={2510.06122},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2510.06122},
}
```
