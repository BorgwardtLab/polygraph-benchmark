#!/usr/bin/env python3
"""Generate an HTML page for side-by-side visual comparison of old (paper) vs new (repo) figures."""

import base64
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_DIR = Path(
    "/fs/pool/pool-hartout/Documents/papers/polygraph_iclr_paper/figures"
)
REPO_DIR = Path(
    "/fs/gpfs41/lv11/fileset01/pool/pool-hartout/Documents/Git/polygraph-benchmark/reproducibility/figures"
)

# Mapping: (repo_subdir, paper_subdir)
SUBDIR_MAP = {
    "01_subsampling": "subsampling",
    "02_perturbation": "perturbation_experiments",
    "03_model_quality": "model_quality",
    "04_phase_plot": "phase_plot",
}

OUTPUT_DIR = Path(
    "/fs/gpfs41/lv11/fileset01/pool/pool-hartout/Documents/Git/polygraph-benchmark/reproducibility/figure_comparison"
)


def pdf_to_png_base64(pdf_path: Path, dpi: int = 150) -> str:
    """Convert first page of a PDF to a base64-encoded PNG string."""
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis("off")

    # Use matplotlib's PDF renderer
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Use ghostscript for conversion
        subprocess.run(
            [
                "gs",
                "-dNOPAUSE",
                "-dBATCH",
                "-dSAFER",
                "-sDEVICE=png16m",
                f"-r{dpi}",
                "-dFirstPage=1",
                "-dLastPage=1",
                f"-sOutputFile={tmp_path}",
                str(pdf_path),
            ],
            capture_output=True,
            check=True,
        )
        with open(tmp_path, "rb") as f:
            png_data = f.read()
    finally:
        os.unlink(tmp_path)
        plt.close(fig)

    return base64.b64encode(png_data).decode("ascii")


def find_matching_figures():
    """Find figures that exist in both paper and repo directories."""
    matches = []

    for repo_subdir, paper_subdir in SUBDIR_MAP.items():
        repo_path = REPO_DIR / repo_subdir
        paper_path = PAPER_DIR / paper_subdir

        if not repo_path.exists() or not paper_path.exists():
            continue

        repo_pdfs = {f.name: f for f in repo_path.glob("*.pdf")}
        paper_pdfs = {f.name: f for f in paper_path.glob("*.pdf")}

        common = sorted(set(repo_pdfs.keys()) & set(paper_pdfs.keys()))
        for name in common:
            matches.append(
                {
                    "name": name,
                    "section": repo_subdir,
                    "paper_path": paper_pdfs[name],
                    "repo_path": repo_pdfs[name],
                }
            )

    return matches


def find_tabpfn_matches():
    """Find model quality figures that have both default and _tabpfn_v6 variants."""
    matches = []
    repo_path = REPO_DIR / "03_model_quality"
    if not repo_path.exists():
        return matches

    default_pdfs = {}
    tabpfn_pdfs = {}
    for f in repo_path.glob("*.pdf"):
        if "_tabpfn_v6" in f.name:
            tabpfn_pdfs[f.name] = f
        else:
            default_pdfs[f.name] = f

    for tabpfn_name, tabpfn_path in sorted(tabpfn_pdfs.items()):
        # Derive the default name by removing the suffix
        default_name = tabpfn_name.replace("_tabpfn_v6", "")
        if default_name in default_pdfs:
            matches.append(
                {
                    "name": default_name,
                    "section": "03_model_quality",
                    "paper_path": default_pdfs[
                        default_name
                    ],  # "left" = default
                    "repo_path": tabpfn_path,  # "right" = tabpfn_v6
                }
            )

    return matches


def generate_html(
    matches,
    output_name="comparison.html",
    title="Figure Comparison",
    subtitle_template="Paper (Old) vs Reproduced (New) &mdash; {n_total} matching figures",
    left_label="Paper (Old)",
    right_label="Repo (New)",
    left_class="old",
    right_class="new",
):
    """Generate an HTML comparison page."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, m in enumerate(matches):
        print(f"  [{i + 1}/{len(matches)}] Converting {m['name']}...")
        try:
            paper_b64 = pdf_to_png_base64(m["paper_path"])
            repo_b64 = pdf_to_png_base64(m["repo_path"])
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        rows.append(
            {
                "name": m["name"],
                "section": m["section"],
                "paper_b64": paper_b64,
                "repo_b64": repo_b64,
            }
        )

    # Also check for repo-only figures (not in paper)
    new_only = []
    for repo_subdir in SUBDIR_MAP:
        repo_path = REPO_DIR / repo_subdir
        paper_subdir = SUBDIR_MAP[repo_subdir]
        paper_path = PAPER_DIR / paper_subdir
        if not repo_path.exists():
            continue
        paper_names = (
            {f.name for f in paper_path.glob("*.pdf")}
            if paper_path.exists()
            else set()
        )
        for f in sorted(repo_path.glob("*.pdf")):
            if f.name not in paper_names and "tabpfn_v6" not in f.name:
                new_only.append(
                    {"name": f.name, "section": repo_subdir, "path": f}
                )

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
  h1 { text-align: center; margin-bottom: 10px; font-size: 1.8em; }
  .subtitle { text-align: center; color: #888; margin-bottom: 30px; }
  .nav { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-bottom: 30px; padding: 15px; background: #16213e; border-radius: 8px; }
  .nav a { color: #4cc9f0; text-decoration: none; padding: 4px 10px; border-radius: 4px; font-size: 0.85em; }
  .nav a:hover { background: #0f3460; }
  .section-header { background: #0f3460; padding: 12px 20px; border-radius: 8px; margin: 30px 0 15px; font-size: 1.3em; }
  .comparison { background: #16213e; border-radius: 8px; margin-bottom: 20px; overflow: hidden; }
  .comparison .title { padding: 10px 15px; font-weight: 600; font-size: 0.95em; background: #1a1a40; border-bottom: 1px solid #333; }
  .comparison .panels { display: flex; align-items: flex-start; }
  .comparison .panel { flex: 1; padding: 10px; text-align: center; position: relative; }
  .comparison .panel:first-child { border-right: 2px solid #e94560; }
  .comparison .panel-label { font-size: 0.75em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
  .comparison .panel-label.old { color: #e94560; }
  .comparison .panel-label.new { color: #4cc9f0; }
  .comparison img { max-width: 100%; height: auto; border-radius: 4px; background: white; }
  .comparison.matched-height .panels { align-items: stretch; }
  .comparison.matched-height .panel { display: flex; flex-direction: column; }
  .comparison.matched-height img { flex: 1; object-fit: contain; object-position: top center; }
  .controls { position: sticky; top: 0; z-index: 100; background: #1a1a2e; padding: 10px 0; border-bottom: 1px solid #333; margin-bottom: 20px; display: flex; gap: 15px; justify-content: center; align-items: center; }
  .controls label { font-size: 0.9em; }
  .controls input[type=range] { width: 200px; }
  .overlay-mode .panels { position: relative; }
  .overlay-mode .panel { position: absolute; top: 0; left: 0; width: 100%; }
  .overlay-mode .panel:first-child { z-index: 1; }
  .overlay-mode .panel:last-child { z-index: 2; }
  .filter-btns { display: flex; gap: 6px; justify-content: center; margin-bottom: 15px; }
  .filter-btns button { background: #16213e; color: #eee; border: 1px solid #333; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 0.85em; }
  .filter-btns button.active { background: #0f3460; border-color: #4cc9f0; }
  .toc { background: #16213e; border-radius: 8px; padding: 15px 20px; margin-bottom: 20px; }
  .toc summary { cursor: pointer; font-weight: 600; }
  .toc ul { margin-top: 10px; columns: 2; }
  .toc li { margin: 4px 0; font-size: 0.85em; }
  .toc a { color: #4cc9f0; text-decoration: none; }
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">{subtitle}</p>

<div class="filter-btns">
  <button class="active" onclick="filterSection('all', this)">All</button>
  <button onclick="filterSection('01_subsampling', this)">Subsampling</button>
  <button onclick="filterSection('02_perturbation', this)">Perturbation</button>
  <button onclick="filterSection('03_model_quality', this)">Model Quality</button>
  <button onclick="filterSection('04_phase_plot', this)">Phase Plot</button>
</div>

<details class="toc">
<summary>Table of Contents ({n_total} figures)</summary>
<ul>
{toc_items}
</ul>
</details>

{comparisons}

<script>
// Match image heights within each comparison for fair visual comparison
document.querySelectorAll('.comparison').forEach(comp => {{
  comp.classList.add('matched-height');
}});

function filterSection(section, btn) {{
  document.querySelectorAll('.filter-btns button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.comparison').forEach(el => {{
    if (section === 'all' || el.dataset.section === section) {{
      el.style.display = '';
    }} else {{
      el.style.display = 'none';
    }}
  }});
  document.querySelectorAll('.section-header').forEach(el => {{
    if (section === 'all' || el.dataset.section === section) {{
      el.style.display = '';
    }} else {{
      el.style.display = 'none';
    }}
  }});
}}
</script>
</body>
</html>"""

    toc_items = []
    comparisons = []
    current_section = None

    section_labels = {
        "01_subsampling": "Subsampling Experiments",
        "02_perturbation": "Perturbation Experiments",
        "03_model_quality": "Model Quality",
        "04_phase_plot": "Phase Plot",
    }

    for r in rows:
        if r["section"] != current_section:
            current_section = r["section"]
            label = section_labels.get(current_section, current_section)
            comparisons.append(
                f'<div class="section-header" data-section="{current_section}">{label}</div>'
            )

        fig_id = r["name"].replace(".", "_").replace(" ", "_")
        toc_items.append(
            f'<li><a href="#{fig_id}">{r["name"]}</a> <span style="color:#666">({current_section})</span></li>'
        )

        comparisons.append(f"""
<div class="comparison" data-section="{r["section"]}" id="{fig_id}">
  <div class="title">{r["name"]}</div>
  <div class="panels">
    <div class="panel">
      <div class="panel-label {left_class}">{left_label}</div>
      <img src="data:image/png;base64,{r["paper_b64"]}" loading="lazy">
    </div>
    <div class="panel">
      <div class="panel-label {right_class}">{right_label}</div>
      <img src="data:image/png;base64,{r["repo_b64"]}" loading="lazy">
    </div>
  </div>
</div>""")

    html = html.replace("{title}", title)
    html = html.replace(
        "{subtitle}", subtitle_template.replace("{n_total}", str(len(rows)))
    )
    html = html.replace("{left_label}", left_label)
    html = html.replace("{right_label}", right_label)
    html = html.replace("{left_class}", left_class)
    html = html.replace("{right_class}", right_class)
    html = html.replace("{n_total}", str(len(rows)))
    html = html.replace("{toc_items}", "\n".join(toc_items))
    html = html.replace("{comparisons}", "\n".join(comparisons))

    output_path = OUTPUT_DIR / output_name
    with open(output_path, "w") as f:
        f.write(html)

    print(f"\nDone! Wrote {output_path}")
    print(f"  {len(rows)} matching figures compared")
    if new_only:
        print(
            f"  {len(new_only)} repo-only figures (not in paper): {[x['name'] for x in new_only]}"
        )

    return output_path


if __name__ == "__main__":
    # 1. Paper vs Repo comparison
    print("Finding matching figures (paper vs repo)...")
    matches = find_matching_figures()
    print(f"Found {len(matches)} matching figures\n")

    print("Generating paper vs repo comparison HTML...")
    path = generate_html(matches)
    print(f"\nOpen in browser: file://{path}")

    # 2. Default vs TabPFN v6 comparison
    print("\nFinding TabPFN v6 matches...")
    tabpfn_matches = find_tabpfn_matches()
    if tabpfn_matches:
        print(f"Found {len(tabpfn_matches)} matching figures\n")
        print("Generating default vs TabPFN v6 comparison HTML...")
        tabpfn_path = generate_html(
            tabpfn_matches,
            output_name="comparison_tabpfn.html",
            title="Default vs TabPFN v6",
            subtitle_template="Default (JSD) vs TabPFN v6 &mdash; {n_total} figures",
            left_label="Default",
            right_label="TabPFN v6",
        )
        print(f"\nOpen in browser: file://{tabpfn_path}")
    else:
        print("No TabPFN v6 matches found.")
