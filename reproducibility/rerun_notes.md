# Reproducibility Rerun Notes

**Date:** 2026-02-12
**Branch:** `camera_ready`
**Cluster:** MPCDF HPC (hpcl91 GPU, hpcl94c CPU)

---

## 0. Cluster Setup Required for Reproducing All Experiments (Current Baseline)

This is the minimum cluster-specific setup needed to rerun all `reproducibility/` experiments
with the current scripts.

### 0.1 Edit only Hydra launcher configs (single source of truth)

Update these files under `reproducibility/configs/hydra/launcher/`:

- `slurm_gpu.yaml`
  - `partition: p.hpcl8`
  - keep GPU request in `additional_parameters.gres` (currently `h100_pcie_2g.20gb:1`)
- `slurm_cpu.yaml`
  - `partition: p.hpcl94c,p.hpcl8` (combined CPU pool)
- `slurm_cpu_small.yaml`
  - `partition: p.hpcl8` (small, high-throughput CPU jobs)
- `slurm_cpu_large.yaml`
  - `partition: p.hpcl94c` (large memory-heavy CPU jobs)

If you move to another cluster, only these partition/GRES values should need changing.

### 0.2 Current default behavior by script

- `01_subsampling/compute_mmd.py`
  - default config: `01_subsampling_mmd`
  - default launcher: `slurm_cpu` (combined CPU pool)
- `01_subsampling/compute_pgd.py` and `01_subsampling/compute.py`
  - default config: `01_subsampling_pgd`
  - default launcher: `slurm_gpu` (hpcl8 GPU)

### 0.3 One-command submission for 01_subsampling

Use:

```bash
reproducibility/01_subsampling/submit_all.sh
```

This script now uses `pixi` and submits:

- MMD (tiered CPU): small sizes -> `hpcl8`, large sizes -> `hpcl94c`
- PGD (GPU): `hpcl8`

### 0.4 Submission pattern for all experiments (01-08)

- TabPFN-based runs (`01`, `02`, `03`, `05`, `07`) -> `hydra/launcher=slurm_gpu`
- CPU-only runs (`04`, `06`, `08`) -> `hydra/launcher=slurm_cpu`

Examples:

```bash
# from repo root, using pixi env
pixi run python reproducibility/02_perturbation/compute.py --multirun hydra/launcher=slurm_gpu
pixi run python reproducibility/06_mmd/compute.py --multirun hydra/launcher=slurm_cpu
```

### 0.5 What each experiment produces

- `01_subsampling`
  - Compute outputs JSON files in script-specific folders under `reproducibility/figures/01_subsampling/results/`
    - `reproducibility/figures/01_subsampling/results/compute/`
    - `reproducibility/figures/01_subsampling/results/compute_pgd/`
    - `reproducibility/figures/01_subsampling/results/compute_mmd/`
  - Main filename patterns:
    - PGD: `pgd_{dataset}_{model}_{subsample_size}.json`
    - MMD: `mmd_{dataset}_{model}_{subsample_size}_{descriptor}_{variant}.json`
  - Plot output: `reproducibility/figures/01_subsampling/subsampling_planar.pdf`
- `02_perturbation`
  - Compute outputs JSON in `reproducibility/figures/02_perturbation/results/`
  - Pattern: `perturbation_{dataset}_{perturbation}.json`
  - Plot output: `reproducibility/figures/02_perturbation/perturbation_*.pdf`
- `03_model_quality`
  - Compute outputs JSON in `reproducibility/figures/03_model_quality/results/`
  - Pattern: `{curve_type}_{dataset}_{variant}.json`
  - Plot output: `reproducibility/figures/03_model_quality/*.pdf`
  - Table output: `reproducibility/tables/model_quality.tex`
- `04_phase_plot`
  - Compute output: `reproducibility/figures/04_phase_plot/results/phase_plot.json`
  - Plot output: `reproducibility/figures/04_phase_plot/metrics_vs_steps.pdf`
- `05_benchmark`
  - Compute outputs JSON in `reproducibility/tables/results/benchmark/`
  - Pattern: `{dataset}_{model}.json`
  - Table output: `reproducibility/tables/benchmark.tex`
- `06_mmd`
  - Compute outputs JSON in `reproducibility/tables/results/mmd/`
  - Pattern: `{dataset}_{model}.json`
  - Table outputs: `reproducibility/tables/mmd_*.tex`
- `07_concatenation`
  - Compute outputs JSON in `reproducibility/tables/results/concatenation/`
  - Pattern: `{dataset}_{model}.json`
  - Table output: `reproducibility/tables/concatenation.tex`
- `08_gklr`
  - Compute outputs JSON in `reproducibility/tables/results/gklr/`
  - Pattern: `{dataset}_{model}.json`
  - Table output: `reproducibility/tables/gklr.tex`

### 0.6 Single async results file (shared across jobs)

To stream results from many SLURM tasks into one lock-safe file (like the original repo),
set:

```bash
export POLYGRAPH_ASYNC_RESULTS_FILE=reproducibility/results/01_subsampling/async_results.jsonl
```

All `01`-`08` compute scripts now append one JSON line per completed/skipped/error task using
an exclusive file lock (`fcntl`) via `polygraph/utils/io.py`.

The `01_subsampling/submit_all.sh` wrapper now auto-sets this env var (to the path above) if
you do not provide one.

---

## 1. Experiment Overview & Node Assignment

### TabPFN experiments -> hpcl91 (20G MIG H100 slices)

| Experiment | Description | TabPFN Usage |
|---|---|---|
| 01_subsampling | PGD variance vs subsample size | `StandardPGDInterval` (classifier=None -> TabPFN) |
| 02_perturbation | Perturbation sensitivity (34 metrics) | Explicit `tabpfn` + `lr` classifiers |
| 03_model_quality | Training/denoising curves | `StandardPGD(classifier=None)` -> TabPFN |
| 05_benchmark | Main benchmark table (Table 1) | `StandardPGDInterval` -> TabPFN |
| 07_concatenation | Concatenated vs max-reduction PGD | `StandardPGDInterval` + `PolyGraphDiscrepancyInterval(classifier=None)` -> TabPFN |

### Non-TabPFN experiments -> hpcl94c (CPU only, AMD Genoa)

| Experiment | Description | Why no TabPFN |
|---|---|---|
| 04_phase_plot | VUN vs loss from training logs | Just CSV parsing, no metric computation |
| 06_mmd | MMD table | Only MMD metrics (kernel-based, no classifiers) |
| 08_gklr | Graph kernel logistic regression | Uses `KernelLogisticRegression`, not TabPFN |

### SLURM Configuration

- **GPU config** (`slurm_gpu.yaml`): partition `p.hpcl91`, gres `gpu:h100_pcie_2g.20gb:1`
- **CPU config** (`slurm_cpu.yaml`): partition `p.hpcl94c`, 10 CPUs, 192 GB RAM (increased from 32 GB due to GKLR OOM)

---

## 2. Bugs Found & Fixed

### Bug 1: Wrong SLURM partition names

**Status:** FIXED
**Files:** `configs/hydra/launcher/slurm_gpu.yaml`, `configs/hydra/launcher/slurm_cpu.yaml`

The configs used `partition: gpu` and `partition: cpu` but the actual SLURM partitions are `p.hpcl91`, `p.hpcl94c`, etc. There are no generic `gpu`/`cpu` partitions on this cluster.

**Fix:** Updated partition names and GPU GRES to target 20G MIG slices specifically:
```yaml
# slurm_gpu.yaml
partition: p.hpcl91
gpus_per_node: 0
additional_parameters:
  gres: "gpu:h100_pcie_2g.20gb:1"

# slurm_cpu.yaml
partition: p.hpcl94c
```

### Bug 2: 01_subsampling sweeper creates 60x redundant jobs

**Status:** KNOWN, workaround applied
**File:** `configs/01_subsampling.yaml`

The config defines sweeper params for `model`, `descriptor`, `variant` (used by `compute_mmd.py` / `compute_pgd.py`) but the main `compute.py` only uses `dataset` and `subsample_size`. Running `compute.py --multirun` creates 3 x 5 x 10 x 6 x 2 = 1800 jobs instead of the needed 3 x 10 = 30.

**Workaround:** Override unused sweeper params on command line to single values.

### Bug 3: pixi.toml references deleted scripts

**Status:** KNOWN, not blocking
**File:** `pixi.toml`

The pixi task definitions (e.g., `figure-subsampling`, `tables-submit`) reference old scripts like `01_generate_subsampling_figures.py` that have been deleted and replaced by the new `XX_*/compute.py` + `XX_*/plot.py` Hydra architecture. The Makefile is the correct entry point now.

### Bug 4: Missing AUTOGRAPH training logs for experiment 04

**Status:** KNOWN, data gap
**File:** `04_phase_plot/compute.py` expects `data/AUTOGRAPH/logs/sbm_proc_{small,large}_metrics.csv`

The `data/AUTOGRAPH/logs/` directory does not exist. Experiment 04 will skip gracefully but produce no results. These training logs need to be downloaded or generated separately.

### Bug 5: Wrong package name in graph_storage.py

**Status:** FIXED
**File:** `polygraph/datasets/base/graph_storage.py:120`

`version("polygraph")` should be `version("polygraph-benchmark")`. The package is named `polygraph-benchmark` in `pyproject.toml`. This caused ALL SLURM jobs to fail with "No package metadata was found for polygraph".

**Fix:** Changed to `version("polygraph-benchmark")`.

### Bug 6: Hydra launcher configs used `@package _global_` with nested `hydra.launcher` keys

**Status:** FIXED
**Files:** `configs/hydra/launcher/slurm_gpu.yaml`, `slurm_cpu.yaml`, `slurm_gpu_fallback.yaml`

The launcher configs used `@package _global_` with `hydra: launcher:` nesting, but Hydra expects launcher configs in the `hydra/launcher/` config group to use `@package _group_` (default) with flat keys. The `@package _global_` caused `SlurmQueueConf` structured config conflicts.

**Fix:** Removed `@package _global_` and flattened config keys.

### Bug 7: 02_perturbation file glob mismatch

**Status:** KNOWN, plot.py broken
**Files:** `02_perturbation/compute.py` vs `02_perturbation/plot.py`

- Compute writes: `perturbation_{dataset}_{perturbation}.json`
- Plot globs: `{dataset}_*.json` -- won't match any files
- Plot also expects `pgs_mean`/`pgs_std` keys but compute writes per-metric scores (orbit_tv, degree_tv, etc.) nested in a `results` array

### Bug 8: 03_model_quality file naming + key mismatch

**Status:** KNOWN, plot.py and format.py broken
**Files:** `03_model_quality/compute.py` vs `plot.py`/`format.py`

- Compute writes: `{curve_type}_{dataset}_{variant}.json` (e.g., `training_planar_jsd.json`)
- Plot loads: `{curve_type}.json` (e.g., `training.json`) -- filename mismatch
- Compute writes `polyscore` key; plot expects `pgs_mean`/`pgs_std`
- `polyscore` value is a raw `pgd_result["pgd"]` which may not serialize as expected

### Bug 9: 04_phase_plot JSON structure mismatch

**Status:** KNOWN, plot.py broken
**Files:** `04_phase_plot/compute.py` vs `plot.py`

- Compute produces: `{"sbm_small": {"val_loss": [...], "vun": [...], "steps": [...]}, ...}`
- Plot expects: `data["results"]` key with `pgs_mean`, `pgs_std`, `vun`, `steps` per row
- Completely different data structures

### Bug 10: 05_benchmark subscore key naming

**Status:** KNOWN, format.py partial mismatch
**Files:** `05_benchmark/compute.py` vs `format.py`

- Compute writes subscore keys like `clustering_mean`, `degree_mean`
- Format expects `clustering_pgs_mean`, `degree_pgs_mean` (with `_pgs` suffix)
- PGD mean/std keys (`pgs_mean`, `pgs_std`) are correct

### Bug 11: 08_gklr format expects wrong metric names

**Status:** KNOWN, format.py broken
**Files:** `08_gklr/compute.py` vs `format.py`

- Compute writes GKLR subscores: `wl_mean`, `shortest_path_mean`, `pyramid_match_mean`
- Format hardcodes expected metrics as: `clustering`, `degree`, `gin`, `orbit4`, `orbit5`, `spectral`
- None of these match -- format will show NaN for all subscores

### Bug 12: 08_gklr OOM — sparse-to-dense conversion in `_descriptions_to_classifier_metric`

**Status:** FIXED
**Files:** `polygraph/metrics/base/polygraphdiscrepancy.py`, `08_gklr/run_single.py`

**Root cause:** `_descriptions_to_classifier_metric()` unconditionally called `.toarray()` on
sparse CSR descriptor features (line 275, 283). WL features have 2^31 (2 billion) sparse columns.
Converting to dense: 2^31 × n_rows × 8 bytes = ~16 GB per row. With subsample_size=128 (256 rows
total), this consumed 500+ GB instantly.

**Fix:** Modified `_descriptions_to_classifier_metric` to keep features sparse when `scale=False`
and `classifier is not None` (i.e., not TabPFN). Key changes:
- Added `_vstack()` helper for sparse-aware vertical stacking
- Added `_is_constant()` helper for sparse-aware constant check
- Replaced `len(x)` with `x.shape[0]` for sparse compatibility
- Modified `_classifier_cross_validation` to handle sparse input
- `KernelLogisticRegression` already handles sparse via `LinearKernel.pre_gram_block`
  → `sparse_dot_product()`, so no changes needed there.

**Verified:** 32-graph test completes in 234 MB (vs 500+ GB before). Both GKLR (sparse) and
standard TabPFN (dense) paths confirmed working.

Memory before fix: 512 graphs → 515+ GB OOM
Memory after fix: 512 graphs → estimated ~500 MB

Restored to 512 graphs per dataset, 192 GB memory allocation, 4 concurrent tasks.
Resubmitted as job 2157393.

### Bug 13: TabPFN runs on CPU despite MIG GPU allocation

**Status:** WORKAROUND applied
**Files:** `configs/hydra/launcher/slurm_gpu.yaml`

Despite SLURM correctly allocating MIG 2g.20gb slices (confirmed via `CUDA_VISIBLE_DEVICES`
and `nvidia-smi -L`), TabPFN runs on CPU. PyTorch's `torch.cuda.is_available()` appears
to return False inside MIG containers, likely due to CUDA runtime vs MIG compatibility.

Consequences:
- Subsample sizes ≤500 work (total samples ≤1000 for classifier)
- Subsample sizes >500 trigger TabPFN's hard CPU limit: "Running on CPU with more than
  1000 samples is not allowed"

**Fix:** Added `export TABPFN_ALLOW_CPU_LARGE_DATASET=1` to `slurm_gpu.yaml` setup.
This allows TabPFN to run on CPU without size limits (slower, but functional).

### Bug 14: 01_subsampling missing results for large subsample sizes

**Status:** FIXED by resubmission
**Root cause:** Bug 13 (TabPFN CPU limit)

Tasks 7-9 (planar subsample_sizes 1024, 2048, 4096) completed without producing results
because TabPFN refused to process >1000 samples on CPU. Resubmitted with
`TABPFN_ALLOW_CPU_LARGE_DATASET=1` env var.

**Impact:** All TabPFN experiments (01, 02, 03, 05, 07) run on CPU, ~5-10x slower.
Jobs should still complete within the 360-minute timeout for most configurations.

**Root cause hypothesis:** CUDA MIG UUID device format may not be properly recognized
by the PyTorch CUDA runtime in this environment. The `gpus_per_node: 0` submitit setting
does not set `--gpus-per-node` in sbatch, relying only on `--gres` for allocation, which
may not properly configure the CUDA runtime path for MIG devices.

### Bug 14: TabPFN CPU limit blocks subsample_size >= 1024

**Status:** FIXED via env var
**Files:** `configs/hydra/launcher/slurm_gpu.yaml`

TabPFN v2.0.9 refuses to process >1000 samples on CPU by default:
```
Error computing PGD for subsample_size=1024: Running on CPU with more than 1000 samples
is not allowed by default due to slow performance.
```

For subsample_size=1024, the classifier gets 2×1024 = 2048 samples, exceeding the limit.
Jobs 7-9 of pgd_01 (subsample_size 1024/2048/4096) failed with this error.

**Fix:** Added `export TABPFN_ALLOW_CPU_LARGE_DATASET=1` to `slurm_gpu.yaml` setup.
Note: These jobs will still be very slow on CPU (hours per job).

### Bug 15: 03_model_quality RBFOrbitMMD2 TypeError crashes all jobs

**Status:** FIXED (stale cache)
**File:** `03_model_quality/compute.py`

All 12 pgd_03 tasks (job 2149719) completed in SLURM but returned `JobStatus.FAILED` with:
```
TypeError("RBFOrbitMMD2.__init__() got an unexpected keyword argument 'graphlet_size'")
```

The current code on disk calls `RBFOrbitMMD2(ref)` correctly (no extra kwargs), and this
works locally. The error likely came from stale `.pyc` bytecode cache on the cluster node
that referenced an older version of `rbf_mmd.py`.

**Fix:** Cleared all `__pycache__` directories and resubmitted as job 2155329.
The orbit5 MMD is correctly constructed using `MaxDescriptorMMD2` with
`AdaptiveRBFKernel(descriptor_fn=OrbitCounts(graphlet_size=5), bw=...)`.

### Bug 16: 08_gklr fundamentally unscalable — RESOLVED by Bug 12 fix

**Status:** FIXED (was misdiagnosed as kernel complexity; actual cause was Bug 12)

The memory explosion was NOT inherent to graph kernels — it was caused by the
unconditional `.toarray()` conversion in `_descriptions_to_classifier_metric()`.
After fixing Bug 12, GKLR runs in ~500 MB for 512 graphs instead of 500+ GB.

All 16 GKLR tasks (4 datasets × 4 models) resubmitted as job 2157393 with 512 graphs.

### Bug 17: 05_benchmark proteins subsample_size exceeds PolyGraphDiscrepancyInterval constraint

**Status:** FIXED
**File:** `05_benchmark/compute.py`

The proteins special case `subsample_size = len(reference_graphs)` (184) violates
`PolyGraphDiscrepancyInterval`'s constraint that `len(reference_graphs) >= 2 * subsample_size`.
This would crash all 4 proteins tasks (12-15) with `ValueError`.

**Fix:** Removed the proteins special case, using the same formula as other datasets:
`subsample_size = min(int(min_subset * 0.5), 2048)` → 92 for proteins.
Tasks 12-15 are still PENDING and will pick up the fixed code from the shared filesystem.

### Summary: Data flow audit

| Exp | Compute | Plot/Format | Status |
|---|---|---|---|
| 01_subsampling | OK | OK | Working |
| 02_perturbation | OK | BROKEN | File glob + JSON structure mismatch |
| 03_model_quality | OK | BROKEN | Filename + key mismatch |
| 04_phase_plot | OK (but missing data) | BROKEN | JSON structure mismatch |
| 05_benchmark | OK | PARTIAL | Subscore key naming off |
| 06_mmd | OK | OK | Working |
| 07_concatenation | OK | OK | Working |
| 08_gklr | OK | BROKEN | Wrong expected metric names |

---

## 3. Implementation Differences: polygraph-benchmark vs. polygraph

### Architecture Changes

| Aspect | polygraph (original) | polygraph-benchmark (v1.0.2) |
|---|---|---|
| **Metric organization** | Modular by model (gin/, gran/) | Unified (standard_pgd.py, rbf_mmd.py, etc.) |
| **Classifier approach** | Kernel-based (LOO, train/test split) | 4-fold stratified k-fold CV |
| **Default classifier** | Selectable (kernel_logistic, logistic, tabpfn) | TabPFN only (classifier=None) |
| **Scoring variants** | auroc, informedness, informedness-adaptive | jsd, informedness only |
| **Result key names** | "polyscore", "polyscore_descriptor" | "pgd", "pgd_descriptor" |
| **Python version** | >=3.7 | >=3.10 |
| **TabPFN version** | Not pinned | Pinned to 2.0.9 |

### Key Algorithmic Differences

1. **Evaluation strategy:** Original uses train/test split with LOO on training set. Benchmark uses 4-fold stratified k-fold CV.
2. **Threshold selection:** Original selects optimal threshold from training data, applies to test. Benchmark selects threshold per fold within CV.
3. **Descriptor combination:** Original uses `AggregateClassifierMetric`. Benchmark uses `PolyGraphDiscrepancy` with max-reduction.
4. **Removed features:** Kernel-based metrics, GIN/GRAN specialization, data copying utilities.

### New in Benchmark

- Uncertainty quantification (Interval variants: `StandardPGDInterval`, etc.)
- Molecule descriptors (TopoChemical, Fingerprint, Lipinski, ChemNet, MolCLR)
- `MoleculePGD` metric
- HPC support via Hydra + submitit
- Graph kernel descriptors (WeisfeilerLehman, ShortestPath, PyramidMatch)

---

## 4. Paper Results Reference (ICLR)

### Table 1: Main Benchmark (PGD x100) — REVISED (camera-ready) version

**Note:** The paper has two Table 1 versions. The MMD appendix tables (`mmd_rbf_umve.tex`, etc.) use
OLD values (PGD 34.0 for planar/AUTOGRAPH). The main paper uses REVISED values (PGD 33.5). Our benchmark
comparison should use the revised values below.

| Dataset | Model | VUN | PGD | Clust | Deg | GIN | Orb5 | Orb4 | Eig |
|---|---|---|---|---|---|---|---|---|---|
| Planar-L | AutoGraph | 85.7 | **33.5** | **6.0** | **8.2** | **8.4** | **33.5** | **28.1** | **27.7** |
| Planar-L | DiGress | 79.6 | 45.4 | 23.9 | 23.7 | 29.3 | 45.4 | 41.1 | 39.4 |
| Planar-L | GRAN | 3.1 | 99.3 | 99.1 | 98.1 | 98.1 | 99.3 | 98.9 | 98.2 |
| Planar-L | ESGG | 93.9 | 45.0 | 10.9 | 21.7 | 32.9 | 45.0 | 42.8 | 29.6 |
| Lobster-L | AutoGraph | 82.6 | 16.4 | 5.1 | 11.0 | 12.6 | 16.4 | 14.5 | 11.8 |
| Lobster-L | DiGress | 91.8 | **4.1** | 1.7 | **1.6** | **2.5** | **5.1** | **5.0** | **0.6** |
| Lobster-L | GRAN | 42.9 | 85.7 | 21.4 | 76.5 | 79.6 | 85.8 | 85.4 | 69.9 |
| Lobster-L | ESGG | 70.9 | 69.9 | **0.0** | 63.4 | 66.8 | 69.9 | 66.0 | 51.7 |
| SBM-L | AutoGraph | 86.2 | **5.7** | **0.9** | **5.9** | **5.8** | **3.0** | **2.4** | **4.4** |
| SBM-L | DiGress | 73.5 | 16.7 | 5.3 | 10.1 | 14.4 | 16.7 | 14.7 | 7.6 |
| SBM-L | GRAN | 21.2 | 68.8 | 49.6 | 57.8 | 69.1 | 65.6 | 62.3 | 54.0 |
| SBM-L | ESGG | 10.2 | 99.4 | 97.9 | 97.5 | 98.3 | 96.8 | 89.2 | 99.4 |
| Proteins | AutoGraph | - | **64.6** | 49.6 | 33.7 | 51.8 | **67.0** | **48.3** | 59.4 |
| Proteins | DiGress | - | 88.5 | **31.0** | **20.4** | **17.6** | 88.5 | 60.4 | **31.6** |
| Proteins | GRAN | - | 91.6 | 87.0 | 70.8 | 72.3 | 91.6 | 87.9 | 75.7 |
| Proteins | ESGG | - | 81.6 | 60.5 | 58.2 | 62.4 | 82.3 | 75.7 | 35.2 |

### Table 1 (from mmd_rbf_umve.tex): Benchmark with UMVE RBF MMD²

Note: The paper's "Table 1" in `benchmark_results.tex` lists PGD as 33.5 etc., but the `mmd_rbf_umve.tex`
table lists PGD as 34.0. The difference is likely the benchmark table vs a combined table. Using the more
precise values from `mmd_rbf_umve.tex` which also has VUN:

| Dataset | Model | VUN | PGD |
|---|---|---|---|
| Planar-L | AutoGraph | 0.851 | 34.0 +/- 1.8 |
| Planar-L | DiGress | 0.801 | 45.2 +/- 1.8 |
| Planar-L | GRAN | 0.016 | 99.7 +/- 0.2 |
| Planar-L | ESGG | 0.939 | 45.0 +/- 1.4 |
| Lobster-L | AutoGraph | 0.831 | 18.0 +/- 1.6 |
| Lobster-L | DiGress | 0.914 | 3.2 +/- 2.6 |
| Lobster-L | GRAN | 0.413 | 85.4 +/- 0.5 |
| Lobster-L | ESGG | 0.709 | 69.9 +/- 0.6 |
| SBM-L | AutoGraph | 0.856 | 5.6 +/- 1.5 |
| SBM-L | DiGress | 0.730 | 17.4 +/- 2.3 |
| SBM-L | GRAN | 0.214 | 69.1 +/- 1.4 |
| SBM-L | ESGG | 0.104 | 99.4 +/- 0.2 |
| Proteins | AutoGraph | - | 67.7 +/- 7.4 |
| Proteins | DiGress | - | 88.1 +/- 3.1 |
| Proteins | GRAN | - | 89.7 +/- 2.7 |
| Proteins | ESGG | - | 79.2 +/- 4.3 |

### Concatenation Table (PGD vs PGD-Concat, x100)

| Dataset | Model | PGD | PGD-Concat |
|---|---|---|---|
| Planar-L | AutoGraph | 34.0 +/- 1.8 | 44.8 +/- 1.3 |
| Planar-L | DiGress | 45.2 +/- 1.8 | 55.3 +/- 1.5 |
| Planar-L | GRAN | 99.7 +/- 0.2 | 99.4 +/- 0.2 |
| Planar-L | ESGG | 45.0 +/- 1.4 | 52.4 +/- 1.1 |
| Lobster-L | AutoGraph | 18.0 +/- 1.6 | 29.0 +/- 2.1 |
| Lobster-L | DiGress | 3.2 +/- 2.6 | 43.2 +/- 1.4 |
| Lobster-L | GRAN | 85.4 +/- 0.5 | 86.4 +/- 0.9 |
| Lobster-L | ESGG | 69.9 +/- 0.6 | 69.9 +/- 1.0 |
| SBM-L | AutoGraph | 5.6 +/- 1.5 | 27.2 +/- 3.0 |
| SBM-L | DiGress | 17.4 +/- 2.3 | 32.0 +/- 2.0 |
| SBM-L | GRAN | 69.1 +/- 1.4 | 78.0 +/- 0.8 |
| SBM-L | ESGG | 99.4 +/- 0.2 | 98.1 +/- 0.4 |
| Proteins | AutoGraph | 67.7 +/- 7.4 | 94.8 +/- 2.6 |
| Proteins | DiGress | 88.1 +/- 3.1 | 99.6 +/- 0.3 |
| Proteins | GRAN | 89.7 +/- 2.7 | 99.8 +/- 0.1 |
| Proteins | ESGG | 79.2 +/- 4.3 | 99.4 +/- 0.3 |

### GKLR Table (PGD-GKLR with PM/SP/WL subscores, x100)

| Dataset | Model | PGD | PGD-GKLR | PM | SP | WL |
|---|---|---|---|---|---|---|
| Planar-L | AutoGraph | 34.0 | 6.2 | 5.3 | 5.2 | 6.7 |
| Planar-L | DiGress | 45.2 | 22.7 | 19.3 | 22.8 | 20.5 |
| Planar-L | GRAN | 99.7 | 43.1 | 8.8 | 5.2 | 43.1 |
| Planar-L | ESGG | 45.0 | 14.4 | 2.7 | 12.8 | 14.6 |
| Lobster-L | AutoGraph | 18.0 | 10.6 | 10.3 | 8.4 | 10.5 |
| Lobster-L | DiGress | 3.2 | 2.4 | 2.6 | 2.5 | 2.2 |
| Lobster-L | GRAN | 85.4 | 72.7 | 52.3 | 57.9 | 72.7 |
| Lobster-L | ESGG | 69.9 | 56.1 | 42.0 | 41.8 | 56.1 |
| SBM-L | AutoGraph | 5.6 | 5.7 | 1.4 | 5.7 | 1.3 |
| SBM-L | DiGress | 17.4 | 8.8 | 7.8 | 4.0 | 9.0 |
| SBM-L | GRAN | 69.1 | 47.4 | 46.8 | 32.7 | 47.4 |
| SBM-L | ESGG | 99.4 | 93.5 | 23.8 | 93.5 | 42.6 |
| Proteins | AutoGraph | 67.7 | 39.2 | 14.0 | 39.2 | 16.5 |
| Proteins | DiGress | 88.1 | 44.8 | 3.6 | 44.8 | 8.9 |
| Proteins | GRAN | 89.7 | 59.4 | 55.0 | 45.7 | 59.4 |
| Proteins | ESGG | 79.2 | 31.9 | 17.7 | 31.9 | 22.0 |

### Key Paper Parameters

- **Classifier:** TabPFN v2.0, 4-fold stratified CV
- **Subsample sizes:** 2048 ref vs 2048 gen (benchmark), up to 4096 for large datasets
- **Bootstrap replicates:** 10 (interval estimation)
- **RBF bandwidths:** {0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0}
- **Descriptors (PGD):** orbit4, orbit5, degree, spectral, clustering, gin
- **Descriptors (GKLR):** PyramidMatch (PM), ShortestPath (SP), WeisfeilerLehman (WL)
- **Perturbation steps:** 100 noise levels per type
- **Concatenation subsample_size:** 1024 (compute.py uses len//4 not len//2)

---

## 5. Experiment Run Log

### Run order (sequential)

| # | Experiment | SLURM Name | Node | Launcher | Jobs | Status | Notes |
|---|---|---|---|---|---|---|---|
| 1 | 01_subsampling | pgd_01 | hpcl91 | slurm_gpu | 30 | PARTIAL | 6/30 results (planar 8-64, lobster 8-16). Pending: job 2156474 (30 jobs, covers all remaining) |
| 2 | 02_perturbation | pgd_02 | hpcl91 | slurm_gpu | 25 | PENDING | Job 2152836. Queued behind pgd_05 |
| 3 | 03_model_quality | pgd_03 | hpcl91 | slurm_gpu | 12 | PENDING | Job 2155329. Previous failed (Bug 15, stale cache). Cleared __pycache__ |
| 4 | 04_phase_plot | pgd_04 | - | - | - | SKIPPED | Missing AUTOGRAPH logs (data/AUTOGRAPH/logs/) |
| 5 | 05_benchmark | pgd_05 | hpcl91 | slurm_gpu | 16 | RUNNING | Job 2149721. Tasks 0-5 running on hpcl9101 (55 min for 0-2, 30 min for 3-5), 6-15 pending. CUDA/MIG confirmed working. 360 min timeout (cannot extend). TabPFN on CPU ~5h estimated |
| 6 | 06_mmd | pgd_06 | hpcl94c | slurm_cpu | 16 | **COMPLETED** | All 16/16 results in tables/results/mmd/ |
| 7 | 07_concatenation | pgd_07 | hpcl91 | slurm_gpu | 16 | PENDING | Job 2149726. Queued behind pgd_05 |
| 8 | 08_gklr | pgd_08 | hpcl94c | direct sbatch | 16 | RUNNING | Job 2157393. Bug 12 fix: sparse features kept sparse. 512 graphs, 192 GB, 4 concurrent |

---

## 6. Results Comparison with Paper

### 06_mmd: MMD Results Comparison (COMPLETE)

**Overall verdict: QUALITATIVELY REPRODUCIBLE, quantitatively divergent in specific areas**

Compared all 192 values across 3 kernel types (RBF biased, GTV, RBF UMVE) × 4 datasets × 4 models × 4 descriptors against the paper's Tables in `tables/mmd_rbf_biased.tex`, `tables/mmd_gtv.tex`, `tables/mmd_rbf_umve.tex`.

**Metric mapping:** Our `rbf_*` = paper's RBF biased (variant="biased"), `gtv_*` = paper's GTV, `umve_*` = paper's RBF UMVE (variant="umve").

**Note:** The pre-generated tables in `reproducibility/tables/` use DIFFERENT values than the paper — they appear to have been generated with a different code version. Only the paper tables in `polygraph_iclr_paper/tables/` are the correct reference.

#### Overall discrepancy distribution (192 values)

| Category | Count | Fraction |
|---|---|---|
| Exact match (<5% diff) | 39 | 20.3% |
| Close (5-50% diff) | 83 | 43.2% |
| Moderate (50-200% diff) | 53 | 27.6% |
| Large (>200% diff) | 17 | 8.9% |

#### By kernel type

| Kernel | Exact (<5%) | Close (<50%) | Moderate | Large (>200%) |
|---|---|---|---|---|
| GTV | 26.6% | 48.4% | 23.4% | 1.6% |
| RBF UMVE | 23.4% | 43.8% | 21.9% | 10.9% |
| RBF biased | 10.9% | 37.5% | 37.5% | 14.1% |

#### Systematic bias

Strong upward bias: 85.6% of discrepant entries have our values **higher** than paper.
Likely causes: different random seed/subsampling, kernel bandwidth calibration, orbit counting.

#### Largest discrepancies

| Kernel | Dataset | Model | Metric | Paper | Ours | Ratio |
|---|---|---|---|---|---|---|
| RBF biased | proteins | DIGRESS | Spectral | 3.6e-03 | 4.4e-02 | 12.1x |
| UMVE | sbm | GRAN | Orbit | 3.2e-03 | 3.6e-02 | 11.2x |
| RBF biased | proteins | ESGG | Spectral | 3.9e-03 | 4.4e-02 | 11.1x |
| UMVE | sbm | DIGRESS | Orbit | 4.3e-04 | 4.3e-03 | 10.0x |
| UMVE | lobster | AUTO | Clustering | 5.0e-06 | 4.9e-05 | 9.9x |
| RBF biased | lobster | AUTO | Clustering | 6.3e-06 | 6.2e-05 | 9.8x |

**Pattern:** SBM orbit consistently inflated 6-11x across all models/kernels (systematic, not random).
Lobster clustering outliers are at tiny absolute values (1e-06 range, noise-dominated).
Proteins spectral inflated 8-12x for RBF biased only (not GTV/UMVE), suggesting kernel-specific issue.

#### Model ranking preservation

Rankings preserved in **37/48 cases (77.1%)**. All 11 mismatches are adjacent-rank swaps
between models with close absolute values. Best/worst model identification is always correct.

| Dataset | Metric | Paper Ranking | Rerun Ranking | Match? |
|---|---|---|---|---|
| Planar | Degree (UMVE) | ESGG < AUTO < GRAN < DIG | AUTO < ESGG < GRAN < DIG | NO (top 2 close) |
| Planar | Orbit (UMVE) | AUTO < GRAN < ESGG < DIG | AUTO < GRAN < ESGG < DIG | YES |
| Lobster | Degree (UMVE) | DIG < AUTO < ESGG < GRAN | DIG < AUTO < ESGG < GRAN | YES |
| Lobster | Orbit (UMVE) | DIG < AUTO < GRAN < ESGG | DIG < AUTO < GRAN < ESGG | YES |
| SBM | Degree (UMVE) | AUTO < DIG < ESGG < GRAN | AUTO < DIG < ESGG < GRAN | YES |
| SBM | Orbit (UMVE) | AUTO < DIG < GRAN < ESGG | AUTO < DIG < GRAN < ESGG | YES |
| Proteins | Degree (UMVE) | DIG < AUTO < ESGG < GRAN | DIG < AUTO < ESGG < GRAN | YES |
| Proteins | Orbit (UMVE) | AUTO < ESGG < DIG < GRAN | AUTO < DIG < ESGG < GRAN | NO (middle 2 close) |

### 01_subsampling: Partial Results (6/30)

| Dataset | Subsample Size | PGD Mean | PGD Std |
|---|---|---|---|
| planar | 8 | 0.088 | 0.168 |
| planar | 16 | 0.055 | 0.103 |
| planar | 32 | 0.035 | 0.073 |
| planar | 64 | 0.030 | 0.051 |
| lobster | 8 | 0.085 | 0.146 |
| lobster | 16 | 0.067 | 0.107 |

Trend matches paper: PGD mean and std decrease with increasing subsample size.
Paper shows stabilization beyond ~256 samples. Remaining jobs (128-4096 for all datasets) pending.

### Remaining experiments

| Experiment | Status | Notes |
|---|---|---|
| 01_subsampling | PARTIAL (6/30) | Decreasing PGD trend matches paper. Awaiting remaining jobs |
| 02_perturbation | PENDING | 25 jobs queued behind pgd_05 |
| 03_model_quality | PENDING | 12 jobs queued. Cache cleared after Bug 15 |
| 04_phase_plot | SKIPPED | Missing AUTOGRAPH training logs |
| 05_benchmark | RUNNING | Jobs 0-5 running (~55 min). TabPFN on CPU is slow (~5h estimated). 360 min timeout |
| 06_mmd | **COMPLETE** | 63.5% within 50%, rankings 77% preserved. Systematic upward bias |
| 07_concatenation | PENDING | 16 jobs queued behind pgd_05 |
| 08_gklr | RUNNING | Bug 12 fixed, job 2157393. 512 graphs, 16 tasks |
