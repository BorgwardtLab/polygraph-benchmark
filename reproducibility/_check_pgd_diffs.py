#!/usr/bin/env python3
"""Quantify PGD v2 vs v2.5 differences."""
import json
from pathlib import Path
import numpy as np

BASE = Path("/fs/gpfs41/lv11/fileset01/pool/pool-hartout/Documents/Git/polygraph-benchmark")
v2_dir = BASE / "reproducibility/figures/01_subsampling/results/compute_pgd_tabpfn_weights_v2"
v25_dir = BASE / "reproducibility/figures/01_subsampling/results/compute_pgd_tabpfn_weights_v2.5"

diffs = []
for f in sorted(v2_dir.glob("*.json")):
    v25_f = v25_dir / f.name
    if not v25_f.exists():
        continue
    v2 = json.loads(f.read_text())
    v25 = json.loads(v25_f.read_text())
    for k in v2:
        if "_mean" in k:
            d = abs(v2[k] - v25[k])
            diffs.append((f.stem, k, v2[k], v25[k], d))

diffs.sort(key=lambda x: -x[4])
arr = np.array([d[4] for d in diffs])
print(f"Total comparisons: {len(diffs)}")
print(f"Mean abs diff: {arr.mean():.6f}")
print(f"Max abs diff:  {arr.max():.6f}")
print(f"Median abs diff: {np.median(arr):.6f}")
print(f"> 0.01: {(arr > 0.01).sum()}/{len(arr)}")
print(f"> 0.05: {(arr > 0.05).sum()}/{len(arr)}")
print()
print("Top 20 largest differences:")
for name, key, v2_val, v25_val, diff in diffs[:20]:
    print(f"  {name} / {key}: v2={v2_val:.4f}  v25={v25_val:.4f}  diff={diff:.4f}")
