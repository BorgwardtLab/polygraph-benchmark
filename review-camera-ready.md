# Code Review: `camera_ready` Branch

**Date:** 2026-03-03
**Branch:** `camera_ready` (19 commits, 116 files, ~15K lines vs `master`)

**Review agents:** security-sentinel, performance-oracle, architecture-strategist, pattern-recognition-specialist, kieran-python-reviewer, code-simplicity-reviewer, git-history-analyzer

## Findings Summary

- **Total Findings:** 26
- **P1 CRITICAL (Bugs):** 3
- **P2 IMPORTANT:** 12
- **P3 NICE-TO-HAVE:** 11

---

## P1 - CRITICAL (Bugs)

### 1. Missing f-string prefix

**File:** `polygraph/metrics/base/polygraphdiscrepancy.py:159`

The error message prints the literal string `{ref_scores.shape}` instead of the actual value.

```python
# CURRENT (broken):
raise RuntimeError("ref_scores must be 1-dimensional, got shape {ref_scores.shape}...")
# SHOULD BE:
raise RuntimeError(f"ref_scores must be 1-dimensional, got shape {ref_scores.shape}...")
```

### 2. `X.shape[0]` crashes for list input

**File:** `polygraph/metrics/base/kernel_lr.py:218`

The type signature declares `X: Union[List[nx.Graph], np.ndarray, csr_array]`, but `.shape[0]` only works for arrays. Should be `np.zeros(len(X))`.

```python
alpha_init = np.zeros(X.shape[0])  # AttributeError when X is list[nx.Graph]
```

### 3. `predict` returns wrong label space

**File:** `polygraph/metrics/base/kernel_lr.py:283`

`fit()` accepts `{0, 1}` labels and converts to `{-1, 1}` internally, but `predict()` returns `np.sign()` output which includes 0 for ties and -1 for the negative class. Should map back to the original label space stored in `self.classes_`.

```python
return np.sign(f).astype(int)  # Returns {-1, 0, 1} instead of {0, 1}
```

---

## P2 - IMPORTANT

### 4. TabPFN classifier factory duplicated 5 times

**Source:** pattern, architecture, simplicity agents

The identical `version_map` + `create_default_for_version` block exists in 5 compute scripts:

- `reproducibility/01_subsampling/compute_pgd.py`
- `reproducibility/02_perturbation/compute.py`
- `reproducibility/03_model_quality/compute.py`
- `reproducibility/05_benchmark/compute.py`
- `reproducibility/07_concatenation/compute.py`

Extract to `reproducibility/utils/data.py` as a single `make_tabpfn_classifier(weights_version: str)` function.

### 5. PGS/PGD naming inconsistency

**Source:** pattern agent

The rename is half-complete. The library API returns `pgd`, commit messages say PGD, but JSON keys use `pgs_mean`, function names use `compute_pgs_metrics`, and format scripts read `pgs_*`. The `08_gklr/compute.py` even translates `pgd_mean` back to `pgs_mean`. Either complete the rename or revert consistently.

### 6. `load_graphs()` duplicated in 8 files, `get_reference_dataset()` in 7

**Source:** pattern, architecture agents

Scripts `01_subsampling/compute_pgd.py`, `01_subsampling/compute_mmd.py`, and `03_model_quality/compute.py` have full reimplementations instead of using the shared `utils/data.py` versions. Subtle behavioral differences risk inconsistent results.

### 7. Unsafe pickle deserialization (9 locations)

**Source:** security agent

`pickle.load()` on data files without validation. Combined with no hash verification on `download_data.py`, this is a supply-chain attack vector. Add SHA-256 hash verification to `download_data.py` as an immediate mitigation.

Affected files:
- `reproducibility/utils/data.py:35`
- `reproducibility/01_subsampling/compute_pgd.py:70`
- `reproducibility/01_subsampling/compute_mmd.py:68`
- `reproducibility/08_gklr/run_single.py:33`
- `reproducibility/03_model_quality/compute.py:40`
- `reproducibility/recompute_training_pgd.py:31`
- `reproducibility/recompute_training_pgd_single.py:30`
- `reproducibility/check_pkl.py:7`
- `reproducibility/05_benchmark/compute.py` (via `utils/data.py`)

### 8. TabPFN classifier re-created on every call

**Source:** performance agent

**File:** `polygraph/metrics/base/polygraphdiscrepancy.py:352-360`

When `classifier=None` in the library's fallback path, `TabPFNClassifier()` is instantiated fresh each call. With `StandardPGDInterval` (6 descriptors x 10 samples), this means 60 redundant instantiations per benchmark run. Cache the classifier instance in `__init__` instead.

### 9. EigenvalueHistogram `.todense()` is O(n^3) per graph

**Source:** performance agent

**File:** `polygraph/utils/descriptors/generic_descriptors.py:232-236`

Converts sparse Laplacian to dense, then computes all eigenvalues. Use `scipy.sparse.linalg.eigsh(L, k=min(n-1, n_bins))` to avoid densification for large graphs.

### 10. Unused `torch` import

**Source:** Python reviewer

**File:** `polygraph/metrics/base/polygraphdiscrepancy.py:58`

`torch` is imported at module level but never used. This adds heavy import overhead to every user of the module.

### 11. KernelLogisticRegression computes `K @ alpha` 3x per L-BFGS iteration

**Source:** performance agent

**File:** `polygraph/metrics/base/kernel_lr.py`

Separate `_objective` and `_gradient` methods each compute `K @ alpha`. L-BFGS-B supports `jac=True` to return both from one function, reducing the dominant O(n^2) operation from 3x to 1x.

### 12. `_create_sparse_matrix` triplicated in 3 descriptor classes

**Source:** Python reviewer

**File:** `polygraph/utils/descriptors/generic_descriptors.py`

Identical method in `WeisfeilerLehmanDescriptor`, `ShortestPathHistogramDescriptor`, and `PyramidMatchDescriptor`. Extract to a shared utility function.

### 13. Hardcoded absolute filesystem paths

**Source:** security agent

7+ files contain paths like `/fs/gpfs41/.../pool-hartout/...` that expose infrastructure and won't work for other researchers. Replace with `pyprojroot.here()` or `$(dirname "$0")`.

Affected files:
- `reproducibility/05_benchmark/submit_benchmark.sh`
- `reproducibility/08_gklr/submit_gklr.sh`
- `reproducibility/compare_figures.py`
- `reproducibility/compare_pgd_v2_vs_v25.py`
- `reproducibility/slurm_recompute_training.sh`
- `reproducibility/05_benchmark/submit_vun.sh`
- `reproducibility/_check_pgd_diffs.py`

### 14. Archive extraction without path traversal protection

**Source:** security agent

**File:** `reproducibility/download_data.py:55`

`shutil.unpack_archive()` without validating archive contents. Add path validation after extraction.

### 15. `global RESULTS_DIR` mutation

**Source:** pattern agent

**File:** `reproducibility/03_model_quality/format.py:312`

Mutates a module-level global from within a function. Refactor to pass as function argument.

---

## P3 - NICE-TO-HAVE

### 16. Remove `compare_*.py` scripts

**Source:** simplicity agent

1,196 LOC of dev-only tools with hardcoded local paths (`compare_figures.py`, `compare_tables.py`, `compare_pgd_v2_vs_v25.py`). Not referenced by Makefile, README, or pipeline. Not needed for camera-ready.

### 17. Legacy typing imports

**Source:** Python reviewer

Replace `Dict`, `Tuple`, `Optional`, `Union` from `typing` with built-in generics (`dict`, `tuple`, `X | None`) throughout library files. Project targets Python 3.10+.

### 18. `assert` in production code

**Source:** pattern agent

**File:** `reproducibility/02_perturbation/compute.py:165-168`

`edge_swapping` uses `assert` for data integrity check. Replace with proper `if/raise ValueError` since `assert` is stripped with `python -O`.

### 19. Missing type hints

**Source:** Python reviewer

Multiple private helpers (`_vstack`, `_is_constant`, `_classifier_cross_validation`) and public APIs (`StandardPGD.classifier`, `compute_pgs_metrics`) lack type annotations.

### 20. `sys.path.insert` hack (9 files)

**Source:** architecture, pattern agents

Nine files use `sys.path.insert(0, str(here() / "reproducibility"))` to import shared utils. Make `reproducibility/` a proper package or use editable install.

### 21. Descriptor dict duplication in `StandardPGD`/`StandardPGDInterval`

**Source:** Python reviewer

**File:** `polygraph/metrics/standard_pgd.py`

The same 6-descriptor dictionary is copy-pasted between both classes. Extract to a module-level factory function `_standard_descriptors()`.

### 22. Pointless function aliases

**Source:** simplicity agent

Several format scripts create aliases like `_fmt_pgs = fmt_pgs` and `_best_two = best_two` that add no value. Remove and use imports directly.

### 23. `io.py` belongs in reproducibility, not library

**Source:** architecture agent

**File:** `polygraph/utils/io.py`

`maybe_append_reproducibility_jsonl` is reproducibility infrastructure that ships with the core library to all pip users. Move to `reproducibility/utils/io.py`.

### 24. Stateful `_fitted` flag in descriptors

**Source:** Python reviewer

**File:** `polygraph/utils/descriptors/generic_descriptors.py:597-627`

`ShortestPathHistogramDescriptor.__call__` has side effects (`self._fitted = True`) and behaves differently on first vs subsequent calls. This surprising statefulness should be documented prominently or refactored.

### 25. `eval` in shell script

**Source:** security agent

**File:** `reproducibility/slurm_recompute_v2.sh:24`

`eval "$@"` in `run_cmd()` function. Replace with direct execution (`"$@"` without eval).

### 26. No download integrity verification

**Source:** security agent

**File:** `reproducibility/download_data.py`

Downloads from MPCDF DataShare without verifying SHA-256 hash. Add hash verification before extraction.

---

## Git History Notes

- Two massive commits (2000+ files each) mix code changes with regenerated binary outputs, making review difficult
- The `"orbig5"` typo fix (a real bug in `StandardPGDInterval`) was silently buried in a 2091-file commit
- 5 rapid bug-fix commits immediately after the big restructure suggest insufficient pre-commit testing
- Working tree has significant uncommitted/untracked changes suggesting work is still in progress
- 35% of commits are bug fixes, which is high for a branch preparing camera-ready submission

---

## Recommended Priority Order

1. **Fix P1 bugs** -- the 3 bugs are straightforward fixes (f-string, `len(X)`, label mapping)
2. **Extract TabPFN factory** -- consolidate the 5 duplicated blocks into one shared function
3. **Decide on PGS vs PGD naming** -- the current split state is the worst of both worlds
4. **Add hash verification to `download_data.py`** -- closes the supply-chain vector cheaply
5. **Remove unused `torch` import** -- free performance win
6. **Cache TabPFN classifier in library fallback** -- significant performance improvement
