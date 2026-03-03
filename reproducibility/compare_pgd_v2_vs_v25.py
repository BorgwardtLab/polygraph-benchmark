#!/usr/bin/env python3
"""Generate HTML comparison of PGD figures: TabPFN weights v2 vs v2.5.

Finds all PDF figures with _tabpfn_weights_v2.pdf and _tabpfn_weights_v2.5.pdf
suffixes, pairs them, and generates a side-by-side HTML comparison with a diff
heatmap highlighting pixel differences.
"""

import base64
import io
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

FIGURES_DIR = Path("/fs/gpfs41/lv11/fileset01/pool/pool-hartout/Documents/Git/polygraph-benchmark/reproducibility/figures")
OUTPUT_DIR = Path("/fs/gpfs41/lv11/fileset01/pool/pool-hartout/Documents/Git/polygraph-benchmark/reproducibility/figure_comparison/pgd_v2_vs_v25")

SECTION_LABELS = {
    "01_subsampling": "Subsampling Experiments",
    "02_perturbation": "Perturbation Experiments",
    "03_model_quality": "Model Quality",
}


def pdf_to_png_bytes(pdf_path: Path, dpi: int = 150) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [
                "gs", "-dNOPAUSE", "-dBATCH", "-dSAFER",
                "-sDEVICE=png16m", f"-r{dpi}",
                "-dFirstPage=1", "-dLastPage=1",
                f"-sOutputFile={tmp_path}", str(pdf_path),
            ],
            capture_output=True, check=True,
        )
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def png_bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def compute_diff_image(png_bytes_a: bytes, png_bytes_b: bytes) -> tuple[str, float, int]:
    """Compute a diff heatmap image. Returns (b64_png, pct_diff, max_diff)."""
    img_a = np.array(Image.open(io.BytesIO(png_bytes_a)))
    img_b = np.array(Image.open(io.BytesIO(png_bytes_b)))

    # Handle size mismatch
    min_h = min(img_a.shape[0], img_b.shape[0])
    min_w = min(img_a.shape[1], img_b.shape[1])
    img_a = img_a[:min_h, :min_w]
    img_b = img_b[:min_h, :min_w]

    # Compute per-pixel absolute difference (max across RGB channels)
    diff = np.abs(img_a.astype(int) - img_b.astype(int))
    diff_max_channel = diff.max(axis=-1) if diff.ndim == 3 else diff

    n_diff_pixels = (diff_max_channel > 5).sum()
    total_pixels = diff_max_channel.size
    pct_diff = 100.0 * n_diff_pixels / total_pixels
    max_diff = int(diff_max_channel.max())

    # Create heatmap: red channel = diff intensity, amplified for visibility
    amplified = np.clip(diff_max_channel * 3, 0, 255).astype(np.uint8)
    heatmap = np.zeros((min_h, min_w, 3), dtype=np.uint8)
    heatmap[:, :, 0] = amplified  # red
    # Overlay faint original in grayscale for context
    gray = np.mean(img_a[:, :, :3].astype(float), axis=-1).astype(np.uint8)
    mask = amplified < 10
    for c in range(3):
        heatmap[:, :, c] = np.where(mask, gray // 3, heatmap[:, :, c])

    img_out = Image.fromarray(heatmap)
    buf = io.BytesIO()
    img_out.save(buf, format="PNG")
    return png_bytes_to_b64(buf.getvalue()), pct_diff, max_diff


def find_pairs():
    """Find all v2/v2.5 figure pairs, grouped by section."""
    v2_files = {}
    v25_files = {}

    for pdf in FIGURES_DIR.rglob("*_tabpfn_weights_v2.pdf"):
        section = pdf.parent.name
        base = pdf.name.replace("_tabpfn_weights_v2.pdf", "")
        v2_files[(section, base)] = pdf

    for pdf in FIGURES_DIR.rglob("*_tabpfn_weights_v2.5.pdf"):
        section = pdf.parent.name
        base = pdf.name.replace("_tabpfn_weights_v2.5.pdf", "")
        v25_files[(section, base)] = pdf

    pairs = []
    v25_only = []

    all_keys = sorted(set(v2_files.keys()) | set(v25_files.keys()))
    for key in all_keys:
        section, base = key
        if key in v2_files and key in v25_files:
            pairs.append({
                "section": section,
                "base": base,
                "v2_path": v2_files[key],
                "v25_path": v25_files[key],
            })
        elif key in v25_files:
            v25_only.append({
                "section": section,
                "base": base,
                "v25_path": v25_files[key],
            })

    return pairs, v25_only


def generate_html(pairs, v25_only):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, p in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] Converting {p['base']}...")
        try:
            v25_bytes = pdf_to_png_bytes(p["v25_path"])
            v2_bytes = pdf_to_png_bytes(p["v2_path"])
            diff_b64, pct_diff, max_diff = compute_diff_image(v25_bytes, v2_bytes)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        rows.append({
            "section": p["section"],
            "base": p["base"],
            "v2_b64": png_bytes_to_b64(v2_bytes),
            "v25_b64": png_bytes_to_b64(v25_bytes),
            "diff_b64": diff_b64,
            "pct_diff": pct_diff,
            "max_diff": max_diff,
            "v2_path": str(p["v2_path"]),
            "v25_path": str(p["v25_path"]),
        })

    # Sort by diff percentage descending so biggest changes are first within each section
    rows.sort(key=lambda r: (r["section"], -r["pct_diff"]))

    # Build TOC and comparisons
    toc_items = []
    comparisons = []
    current_section = None
    section_counts = {}

    for r in rows:
        section_counts[r["section"]] = section_counts.get(r["section"], 0) + 1

    for r in rows:
        if r["section"] != current_section:
            current_section = r["section"]
            label = SECTION_LABELS.get(current_section, current_section)
            count = section_counts[current_section]
            comparisons.append(
                f'<div class="section-header" data-section="{current_section}">'
                f'{label} ({count} figures)</div>'
            )

        fig_id = f"{r['section']}_{r['base']}".replace(".", "_").replace(" ", "_")
        diff_color = "#4cc9f0" if r["pct_diff"] < 1 else "#ffa500" if r["pct_diff"] < 5 else "#e94560"
        diff_badge = f'<span class="diff-badge" style="color:{diff_color}">{r["pct_diff"]:.1f}% pixels differ</span>'

        toc_items.append(
            f'<li><a href="#{fig_id}">{r["base"]}</a> '
            f'<span style="color:{diff_color}">[{r["pct_diff"]:.1f}%]</span> '
            f'<span style="color:#666">({r["section"]})</span></li>'
        )

        comparisons.append(f"""
<div class="comparison" data-section="{r['section']}" id="{fig_id}">
  <div class="title">{r['base']} {diff_badge}</div>
  <div class="file-paths">
    <span class="path-label v25">v2.5:</span> <code>{r['v25_path']}</code><br>
    <span class="path-label v2">v2:</span> <code>{r['v2_path']}</code>
  </div>
  <div class="panels three-panel">
    <div class="panel">
      <div class="panel-label v25">TabPFN Weights v2.5</div>
      <img src="data:image/png;base64,{r['v25_b64']}" loading="lazy">
    </div>
    <div class="panel">
      <div class="panel-label v2">TabPFN Weights v2</div>
      <img src="data:image/png;base64,{r['v2_b64']}" loading="lazy">
    </div>
    <div class="panel diff-panel">
      <div class="panel-label diff">Diff (red = change, 3x amplified)</div>
      <img src="data:image/png;base64,{r['diff_b64']}" loading="lazy">
    </div>
  </div>
</div>""")

    # v2.5-only section
    v25_only_html = ""
    if v25_only:
        v25_only_items = []
        for item in v25_only:
            v25_only_items.append(f"<li><code>{item['base']}</code> ({item['section']})</li>")
        v25_only_html = f"""
<div class="section-header" style="background:#4a1942;">v2.5-only figures (no v2 counterpart) &mdash; {len(v25_only)}</div>
<div class="comparison" style="padding:15px;">
  <ul style="columns:2;">{"".join(v25_only_items)}</ul>
</div>"""

    # Filter buttons
    filter_buttons = ['<button class="active" onclick="filterSection(\'all\', this)">All</button>']
    for section in sorted(section_counts.keys()):
        label = SECTION_LABELS.get(section, section)
        count = section_counts[section]
        filter_buttons.append(
            f'<button onclick="filterSection(\'{section}\', this)">{label} ({count})</button>'
        )
    filter_buttons.append('<button onclick="filterDiff(5, this)">Diff > 5%</button>')
    filter_buttons.append('<button onclick="filterDiff(1, this)">Diff > 1%</button>')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PGD Comparison: TabPFN Weights v2 vs v2.5</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 10px; font-size: 1.8em; }}
  .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
  .section-header {{ background: #0f3460; padding: 12px 20px; border-radius: 8px; margin: 30px 0 15px; font-size: 1.3em; }}
  .comparison {{ background: #16213e; border-radius: 8px; margin-bottom: 20px; overflow: hidden; }}
  .comparison .title {{ padding: 10px 15px; font-weight: 600; font-size: 0.95em; background: #1a1a40; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }}
  .comparison .file-paths {{ padding: 6px 15px; font-size: 0.75em; color: #999; background: #12122a; border-bottom: 1px solid #222; }}
  .comparison .file-paths code {{ color: #7ec8e3; font-size: 0.95em; }}
  .path-label {{ font-weight: bold; display: inline-block; width: 35px; }}
  .path-label.v25 {{ color: #e94560; }}
  .path-label.v2 {{ color: #4cc9f0; }}
  .diff-badge {{ font-size: 0.8em; font-weight: bold; padding: 2px 8px; border-radius: 4px; background: #1a1a2e; }}
  .comparison .panels {{ display: flex; align-items: flex-start; }}
  .comparison .three-panel .panel {{ flex: 1; padding: 8px; text-align: center; }}
  .comparison .panel:not(:last-child) {{ border-right: 1px solid #333; }}
  .comparison .panel-label {{ font-size: 0.7em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }}
  .comparison .panel-label.v25 {{ color: #e94560; }}
  .comparison .panel-label.v2 {{ color: #4cc9f0; }}
  .comparison .panel-label.diff {{ color: #ffa500; }}
  .comparison img {{ max-width: 100%; height: auto; border-radius: 4px; background: white; }}
  .comparison.matched-height .panels {{ align-items: stretch; }}
  .comparison.matched-height .panel {{ display: flex; flex-direction: column; }}
  .comparison.matched-height img {{ flex: 1; object-fit: contain; object-position: top center; }}
  .filter-btns {{ display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; margin-bottom: 15px; }}
  .filter-btns button {{ background: #16213e; color: #eee; border: 1px solid #333; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 0.85em; }}
  .filter-btns button.active {{ background: #0f3460; border-color: #4cc9f0; }}
  .toc {{ background: #16213e; border-radius: 8px; padding: 15px 20px; margin-bottom: 20px; }}
  .toc summary {{ cursor: pointer; font-weight: 600; }}
  .toc ul {{ margin-top: 10px; columns: 2; }}
  .toc li {{ margin: 4px 0; font-size: 0.85em; }}
  .toc a {{ color: #4cc9f0; text-decoration: none; }}
  .diff-panel img {{ background: #000 !important; }}
</style>
</head>
<body>
<h1>PGD Comparison: TabPFN Weights v2 vs v2.5</h1>
<p class="subtitle">{len(rows)} figure pairs &mdash; sorted by diff % within each section &mdash; red heatmap shows pixel differences (3x amplified)</p>

<div class="filter-btns">
  {"".join(filter_buttons)}
</div>

<details class="toc">
<summary>Table of Contents ({len(rows)} figures)</summary>
<ul>
{"".join(toc_items)}
</ul>
</details>

{"".join(comparisons)}

{v25_only_html}

<script>
document.querySelectorAll('.comparison').forEach(comp => {{
  comp.classList.add('matched-height');
}});

function filterSection(section, btn) {{
  document.querySelectorAll('.filter-btns button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.comparison, .section-header').forEach(el => {{
    if (section === 'all' || el.dataset.section === section) {{
      el.style.display = '';
    }} else {{
      el.style.display = 'none';
    }}
  }});
}}

function filterDiff(threshold, btn) {{
  document.querySelectorAll('.filter-btns button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.section-header').forEach(el => el.style.display = '');
  document.querySelectorAll('.comparison').forEach(el => {{
    const badge = el.querySelector('.diff-badge');
    if (badge) {{
      const pct = parseFloat(badge.textContent);
      el.style.display = pct >= threshold ? '' : 'none';
    }}
  }});
}}
</script>
</body>
</html>"""

    output_path = OUTPUT_DIR / "pgd_v2_vs_v25.html"
    output_path.write_text(html)
    print(f"\nDone! Wrote {output_path}")
    print(f"  {len(rows)} matched pairs, {len(v25_only)} v2.5-only")

    # Summary stats
    diffs = [r["pct_diff"] for r in rows]
    print(f"  Diff stats: mean={np.mean(diffs):.1f}%, max={np.max(diffs):.1f}%, median={np.median(diffs):.1f}%")
    print(f"  > 1%: {sum(1 for d in diffs if d >= 1)}, > 5%: {sum(1 for d in diffs if d >= 5)}")

    return output_path


if __name__ == "__main__":
    print("Finding v2 vs v2.5 figure pairs...")
    pairs, v25_only = find_pairs()
    print(f"Found {len(pairs)} pairs, {len(v25_only)} v2.5-only\n")

    print("Generating comparison HTML with diff heatmaps...")
    path = generate_html(pairs, v25_only)
    print(f"\nOpen in browser: file://{path}")
