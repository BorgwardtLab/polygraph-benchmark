# Reproducibility Package for ICLR 2026 Paper

This directory contains scripts to reproduce all tables and figures from the paper:

**"PolyGraph Discrepancy: a classifier-based metric for graph generation"**

## Quick Start

```bash
# 1. Install dependencies
pip install -e ..[reproducibility]

# 2. Download the graph data (~3GB)
python download_data.py

# 3. Generate all tables and figures
make all
```

## Data Download

The generated graph data is hosted on Proton Drive due to its size (~3GB).

```bash
# Full dataset (required for complete reproducibility)
python download_data.py

# Small subset for testing/CI (~50 graphs per model)
python download_data.py --subset
```

**Manual download:** https://drive.proton.me/urls/VM4NWYBQD0#3sqmZtmSgWTB

After downloading, extract to `polygraph_graphs/` in the repository root.

### Data Structure

```
polygraph_graphs/
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

## Directory Structure

```
reproducibility/
├── README.md                         # This file
├── requirements.txt                  # Dependencies
├── Makefile                          # Automation targets
├── download_data.py                  # Data download script
├── generate_benchmark_tables.py      # Main PGD benchmark (Table 1)
├── generate_mmd_tables.py            # MMD tables (Appendix)
├── generate_gklr_tables.py           # GKLR kernel tables (Appendix)
├── generate_concatenation_tables.py  # Feature concatenation (Appendix)
├── generate_subsampling_figures.py   # Bias/variance figures (Fig 2)
├── generate_perturbation_figures.py  # Perturbation experiments (Fig 3)
├── generate_model_quality_figures.py # Training/denoising (Fig 4-5)
├── generate_phase_plot.py            # Phase plot (Appendix)
├── figures/                          # Generated figure outputs
└── tables/                           # Generated LaTeX table outputs
```

## Scripts Overview

### Table Generation

| Script | Output | Description |
|--------|--------|-------------|
| `generate_benchmark_tables.py` | `tables/benchmark_results.tex` | Main PGD benchmark (Table 1) comparing AUTOGRAPH, DiGress, GRAN, ESGG |
| `generate_mmd_tables.py` | `tables/mmd_gtv.tex`, `tables/mmd_rbf_biased.tex` | MMD² metrics with GTV and RBF kernels |
| `generate_gklr_tables.py` | `tables/gklr.tex` | PGD with Kernel Logistic Regression using WL and SP kernels |
| `generate_concatenation_tables.py` | `tables/concatenation.tex` | Ablation comparing individual vs concatenated descriptors |

### Figure Generation

| Script | Output | Description |
|--------|--------|-------------|
| `generate_subsampling_figures.py` | `figures/subsampling/` | Bias-variance tradeoff as function of sample size |
| `generate_perturbation_figures.py` | `figures/perturbation/` | Metric sensitivity to edge perturbations |
| `generate_model_quality_figures.py` | `figures/model_quality/` | PGD vs training/denoising steps for DiGress |
| `generate_phase_plot.py` | `figures/phase_plot/` | Training dynamics showing PGD vs VUN |

## Individual Scripts

Each script can be run independently with `--subset` flag for quick testing:

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

## Make Targets

```bash
make download        # Download full dataset (manual step required)
make download-subset # Create small subset for CI testing
make tables          # Generate all LaTeX tables
make figures         # Generate all figures
make all             # Generate everything
make clean           # Remove generated outputs
make help            # Show available targets
```

## Hardware Requirements

- **Memory:** 16GB RAM recommended for full dataset
- **Storage:** ~4GB for data + outputs
- **Time:** Full generation takes ~2-4 hours on a modern CPU

### Subset Mode

All scripts support `--subset` flag for quick testing:
- Uses ~50 graphs per model instead of full dataset
- Runs in minutes instead of hours
- Results are not publication-quality but verify code correctness

## Dependencies

See `requirements.txt`. Key dependencies:
- `polygraph-benchmark` (this library, install from repo root)
- `tabpfn==2.0.0` (pinned for reproducibility - paper version)
- `matplotlib`, `seaborn` for plotting
- `typer` for CLI
- `networkx>=3.1` for graph handling

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Reproducing Specific Results

### Table 1: Main Benchmark

```bash
python generate_benchmark_tables.py
# Output: tables/benchmark_results.tex
```

Computes PGD metrics for AUTOGRAPH, DiGress, GRAN, and ESGG on Planar-L, Lobster-L, SBM-L, and Proteins datasets.

### Figure 4-5: Model Quality

```bash
python generate_model_quality_figures.py
# Output: figures/model_quality/training_curve.pdf
#         figures/model_quality/denoising_curve.pdf
```

Shows how PGD improves as DiGress is trained longer or uses more denoising steps.

### Appendix: MMD Comparison

```bash
python generate_mmd_tables.py
# Output: tables/mmd_gtv.tex
#         tables/mmd_rbf_biased.tex
```

Compares GTV and RBF kernel MMD² metrics across all models and datasets.

## Troubleshooting

### Memory Issues
If you encounter memory errors, try:
1. Use `--subset` flag for testing
2. Process one dataset at a time
3. Increase system swap space

### Missing Data
If scripts report missing data:
1. Verify `polygraph_graphs/` exists in repo root
2. Run `python download_data.py` to check data status
3. Download manually from Proton Drive if needed

### TabPFN Issues
TabPFN is pinned to v2.0.0 for reproducibility. If you encounter issues:
```bash
pip install tabpfn==2.0.0
```

## Citation

```bibtex
@inproceedings{krimmel2026polygraph,
  title={PolyGraph Discrepancy: a classifier-based metric for graph generation},
  author={Krimmel, Markus and Hartout, Philip and Borgwardt, Karsten and Chen, Dexiong},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```
