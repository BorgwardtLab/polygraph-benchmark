#!/usr/bin/env python3
"""Generate HTML comparison of old vs new LaTeX tables with change quantification."""

import re
import subprocess
from pathlib import Path

TABLES_DIR = Path(__file__).parent / "tables"

CHANGED_TABLES = [
    "benchmark_results.tex",
    "concatenation.tex",
    "correlation-across-training-informedness.tex",
    "correlation-across-training-jsd.tex",
    "digress-denoising-iters-pgs-jsd.tex",
    "digress-pearson-correlation-informedness.tex",
    "digress-pearson-correlation-jsd.tex",
]

UNCHANGED_TABLES = [
    "denoising_quality.tex",
    "digress-denoising-iters-mmd.tex",
    "gklr.tex",
    "mmd_gtv.tex",
    "mmd_rbf_biased.tex",
    "mmd_rbf_umve.tex",
    "training_quality.tex",
]

TABLE_NAMES = {
    "benchmark_results.tex": "Benchmark Results (Table 1)",
    "concatenation.tex": "Concatenation Experiment (Table 2)",
    "correlation-across-training-informedness.tex": "Correlation Across Training (Informedness)",
    "correlation-across-training-jsd.tex": "Correlation Across Training (JSD)",
    "digress-denoising-iters-pgs-jsd.tex": "DiGress Denoising Iterations PGS (JSD)",
    "digress-pearson-correlation-informedness.tex": "DiGress Pearson Correlation (Informedness)",
    "digress-pearson-correlation-jsd.tex": "DiGress Pearson Correlation (JSD)",
    "denoising_quality.tex": "Denoising Quality",
    "digress-denoising-iters-mmd.tex": "DiGress Denoising Iterations MMD",
    "gklr.tex": "GKLR Results",
    "mmd_gtv.tex": "MMD GTV",
    "mmd_rbf_biased.tex": "MMD RBF Biased",
    "mmd_rbf_umve.tex": "MMD RBF UMVE",
    "training_quality.tex": "Training Quality",
}


def get_old_version(filename: str) -> str:
    """Get file content from git HEAD."""
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:reproducibility/tables/{filename}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return ""


def get_new_version(filename: str) -> str:
    """Get current working directory version."""
    return (TABLES_DIR / filename).read_text()


def latex_to_html_cell(cell: str) -> str:
    """Convert a single LaTeX cell to HTML."""
    cell = cell.strip()
    # Remove \textbf, \underline, \textsc
    cell = re.sub(r"\\textbf\{([^}]*)\}", r"<b>\1</b>", cell)
    cell = re.sub(r"\\underline\{([^}]*)\}", r"<u>\1</u>", cell)
    cell = re.sub(
        r"\\textsc\{([^}]*)\}",
        r'<span style="font-variant:small-caps">\1</span>',
        cell,
    )
    # Convert \pm\,\scriptstyle{X} to ± X
    cell = re.sub(r"\$\\pm\\,\\scriptstyle\{([^}]*)\}\$", r"± \1", cell)
    # Convert $\uparrow$ etc.
    cell = cell.replace(r"$\uparrow$", "↑").replace(r"$\downarrow$", "↓")
    # Remove remaining $ signs
    cell = cell.replace("$", "")
    # Remove \multicolumn but keep text
    cell = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{([^}]*)\}", r"\1", cell)
    return cell


def extract_numeric(cell: str) -> float | None:
    """Extract the main numeric value from a cell."""
    cell = cell.strip()
    if cell == "-" or cell == "":
        return None
    # Remove LaTeX formatting
    clean = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell)
    clean = re.sub(r"\\underline\{([^}]*)\}", r"\1", clean)
    clean = re.sub(r"\$\\pm\\,\\scriptstyle\{[^}]*\}\$", "", clean)
    clean = clean.replace("$", "").strip()
    # Try to extract number
    match = re.search(r"-?\d+\.?\d*", clean)
    if match:
        return float(match.group())
    return None


def parse_latex_table(latex: str) -> list[list[str]]:
    """Parse LaTeX tabular into list of rows of cells."""
    rows = []
    for line in latex.split("\n"):
        line = line.strip()
        if (
            not line
            or line.startswith("\\begin")
            or line.startswith("\\end")
            or line.startswith("\\toprule")
            or line.startswith("\\midrule")
            or line.startswith("\\bottomrule")
            or line.startswith("\\cmidrule")
        ):
            continue
        # Remove trailing \\
        line = re.sub(r"\s*\\\\$", "", line)
        cells = line.split("&")
        rows.append([c.strip() for c in cells])
    return rows


def compute_changes(
    old_rows: list[list[str]], new_rows: list[list[str]]
) -> dict:
    """Compute numerical changes between old and new tables."""
    changes = []
    total_cells = 0
    changed_cells = 0
    new_cells = 0  # cells that went from "-" to a value
    max_abs_change = 0.0

    for i, (old_row, new_row) in enumerate(zip(old_rows, new_rows)):
        for j, (old_cell, new_cell) in enumerate(zip(old_row, new_row)):
            old_val = extract_numeric(old_cell)
            new_val = extract_numeric(new_cell)

            if old_val is not None or new_val is not None:
                total_cells += 1

            if old_val is None and new_val is not None:
                new_cells += 1
                changes.append((i, j, None, new_val, None))
            elif old_val is not None and new_val is not None:
                diff = new_val - old_val
                if abs(diff) > 0.001:
                    changed_cells += 1
                    max_abs_change = max(max_abs_change, abs(diff))
                    changes.append((i, j, old_val, new_val, diff))

    return {
        "changes": changes,
        "total_numeric_cells": total_cells,
        "changed_cells": changed_cells,
        "new_cells": new_cells,
        "max_abs_change": max_abs_change,
    }


def render_table_html(
    rows: list[list[str]], is_header_row: set[int] = None
) -> str:
    """Render parsed rows to an HTML table."""
    if is_header_row is None:
        is_header_row = {0}
    html = "<table>\n"
    for i, row in enumerate(rows):
        html += "  <tr>\n"
        tag = "th" if i in is_header_row else "td"
        for cell in row:
            content = latex_to_html_cell(cell)
            html += f"    <{tag}>{content}</{tag}>\n"
        html += "  </tr>\n"
    html += "</table>\n"
    return html


def render_diff_table(
    old_rows: list[list[str]], new_rows: list[list[str]]
) -> str:
    """Render a diff table showing old→new with color coding."""
    html = '<table class="diff-table">\n'

    # Use new_rows as primary (may have more columns)
    max_rows = max(len(old_rows), len(new_rows))

    for i in range(max_rows):
        old_row = old_rows[i] if i < len(old_rows) else []
        new_row = new_rows[i] if i < len(new_rows) else []

        is_header = i == 0
        tag = "th" if is_header else "td"

        html += "  <tr>\n"
        max_cols = max(len(old_row), len(new_row))
        for j in range(max_cols):
            old_cell = old_row[j] if j < len(old_row) else ""
            new_cell = new_row[j] if j < len(new_row) else ""

            old_val = extract_numeric(old_cell)
            new_val = extract_numeric(new_cell)

            new_content = latex_to_html_cell(new_cell)

            css_class = ""
            annotation = ""
            if not is_header:
                if old_val is None and new_val is not None:
                    css_class = ' class="cell-new"'
                    annotation = ' <span class="annotation">NEW</span>'
                elif old_val is not None and new_val is not None:
                    diff = new_val - old_val
                    if abs(diff) > 0.001:
                        if diff > 0:
                            css_class = ' class="cell-increased"'
                            annotation = (
                                f' <span class="annotation">+{diff:.1f}</span>'
                            )
                        else:
                            css_class = ' class="cell-decreased"'
                            annotation = (
                                f' <span class="annotation">{diff:.1f}</span>'
                            )
                elif (
                    old_cell.strip() != new_cell.strip()
                    and old_cell.strip()
                    and new_cell.strip()
                ):
                    # Text changed
                    css_class = ' class="cell-text-changed"'

            html += f"    <{tag}{css_class}>{new_content}{annotation}</{tag}>\n"
        html += "  </tr>\n"
    html += "</table>\n"
    return html


def generate_html():
    """Generate the full HTML comparison page."""

    sections = []

    # Changed tables
    for filename in CHANGED_TABLES:
        old_latex = get_old_version(filename)
        new_latex = get_new_version(filename)

        old_rows = parse_latex_table(old_latex)
        new_rows = parse_latex_table(new_latex)

        # Compute changes
        min_rows = min(len(old_rows), len(new_rows))
        # Only compare rows that exist in both
        comparable_old = old_rows[:min_rows]
        comparable_new = new_rows[:min_rows]
        stats = compute_changes(comparable_old, comparable_new)

        old_html = render_table_html(old_rows)
        new_html = render_table_html(new_rows)
        diff_html = render_diff_table(old_rows, new_rows)

        summary_parts = []
        if stats["new_cells"] > 0:
            summary_parts.append(
                f'<span class="stat-new">{stats["new_cells"]} new values</span>'
            )
        if stats["changed_cells"] > 0:
            summary_parts.append(
                f'<span class="stat-changed">{stats["changed_cells"]} changed values</span>'
            )
        if stats["max_abs_change"] > 0:
            summary_parts.append(f"Max |Δ|: {stats['max_abs_change']:.1f}")
        summary = (
            " | ".join(summary_parts) if summary_parts else "No numeric changes"
        )

        sections.append(
            {
                "name": TABLE_NAMES.get(filename, filename),
                "filename": filename,
                "old_html": old_html,
                "new_html": new_html,
                "diff_html": diff_html,
                "summary": summary,
                "changed": True,
            }
        )

    # Unchanged tables
    for filename in UNCHANGED_TABLES:
        new_latex = get_new_version(filename)
        new_rows = parse_latex_table(new_latex)
        new_html = render_table_html(new_rows)

        sections.append(
            {
                "name": TABLE_NAMES.get(filename, filename),
                "filename": filename,
                "new_html": new_html,
                "changed": False,
            }
        )

    # Build HTML
    toc_items = []
    for i, s in enumerate(sections):
        status = "🔄" if s["changed"] else "✅"
        toc_items.append(
            f'<li><a href="#table-{i}">{status} {s["name"]}</a></li>'
        )

    content_sections = []
    for i, s in enumerate(sections):
        if s["changed"]:
            content_sections.append(f"""
<div class="table-section" id="table-{i}">
    <h2>{s["name"]} <span class="badge badge-changed">CHANGED</span></h2>
    <p class="filename">{s["filename"]}</p>
    <p class="summary">{s["summary"]}</p>

    <h3>Diff View (new values with change annotations)</h3>
    <div class="table-container">{s["diff_html"]}</div>

    <div class="side-by-side">
        <div class="side">
            <h3>Old (TabPFN ~2.0.9)</h3>
            <div class="table-container">{s["old_html"]}</div>
        </div>
        <div class="side">
            <h3>New (TabPFN 6.4.1)</h3>
            <div class="table-container">{s["new_html"]}</div>
        </div>
    </div>
</div>
""")
        else:
            content_sections.append(f"""
<div class="table-section" id="table-{i}">
    <h2>{s["name"]} <span class="badge badge-unchanged">UNCHANGED</span></h2>
    <p class="filename">{s["filename"]}</p>
    <div class="table-container">{s["new_html"]}</div>
</div>
""")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Table Comparison: Old (TabPFN 2.0.9) vs New (TabPFN 6.4.1)</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        max-width: 1800px;
        margin: 0 auto;
        padding: 20px;
        background: #f8f9fa;
        color: #333;
    }}
    h1 {{
        border-bottom: 2px solid #333;
        padding-bottom: 10px;
    }}
    .toc {{
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px 25px;
        margin-bottom: 30px;
    }}
    .toc ul {{
        list-style: none;
        padding: 0;
        columns: 2;
    }}
    .toc li {{
        padding: 3px 0;
    }}
    .toc a {{
        text-decoration: none;
        color: #0366d6;
    }}
    .toc a:hover {{
        text-decoration: underline;
    }}
    .table-section {{
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 25px;
    }}
    .filename {{
        color: #666;
        font-family: monospace;
        font-size: 0.9em;
    }}
    .summary {{
        font-size: 1.1em;
        font-weight: 500;
        margin: 10px 0;
    }}
    .badge {{
        font-size: 0.6em;
        padding: 3px 8px;
        border-radius: 4px;
        vertical-align: middle;
    }}
    .badge-changed {{
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffc107;
    }}
    .badge-unchanged {{
        background: #d4edda;
        color: #155724;
        border: 1px solid #28a745;
    }}
    .stat-new {{
        color: #0366d6;
        font-weight: bold;
    }}
    .stat-changed {{
        color: #e36209;
        font-weight: bold;
    }}
    .table-container {{
        overflow-x: auto;
        margin: 10px 0;
    }}
    table {{
        border-collapse: collapse;
        font-size: 0.85em;
        width: auto;
    }}
    th, td {{
        border: 1px solid #ddd;
        padding: 6px 10px;
        text-align: center;
        white-space: nowrap;
    }}
    th {{
        background: #f1f3f5;
        font-weight: bold;
    }}
    .diff-table .cell-new {{
        background: #dbeafe;
    }}
    .diff-table .cell-increased {{
        background: #fee2e2;
    }}
    .diff-table .cell-decreased {{
        background: #dcfce7;
    }}
    .diff-table .cell-text-changed {{
        background: #fef3c7;
    }}
    .annotation {{
        font-size: 0.75em;
        font-weight: bold;
        display: block;
        margin-top: 2px;
    }}
    .cell-new .annotation {{
        color: #1d4ed8;
    }}
    .cell-increased .annotation {{
        color: #dc2626;
    }}
    .cell-decreased .annotation {{
        color: #16a34a;
    }}
    .side-by-side {{
        display: flex;
        gap: 20px;
        margin-top: 15px;
    }}
    .side {{
        flex: 1;
        min-width: 0;
    }}
    .side h3 {{
        margin-top: 0;
    }}
    .legend {{
        display: flex;
        gap: 15px;
        margin: 15px 0;
        flex-wrap: wrap;
    }}
    .legend-item {{
        display: flex;
        align-items: center;
        gap: 5px;
        font-size: 0.9em;
    }}
    .legend-swatch {{
        width: 20px;
        height: 20px;
        border: 1px solid #999;
        border-radius: 3px;
    }}
    .overview-table {{
        margin: 15px 0;
    }}
    .overview-table td:first-child {{
        text-align: left;
        font-weight: 500;
    }}
</style>
</head>
<body>
<h1>Table Comparison: TabPFN 2.0.9 → TabPFN 6.4.1</h1>
<p>This page shows how all reproducibility tables changed after updating to TabPFN v6.4.1 results.
For PGD-based metrics (which depend on TabPFN as classifier), values change slightly.
VUN (Valid Unique Novel) and MMD values are TabPFN-independent.</p>

<div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:#dbeafe"></div> New value (was "-")</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#fee2e2"></div> Increased</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#dcfce7"></div> Decreased</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#fef3c7"></div> Text changed</div>
</div>

<div class="toc">
    <h3>Table of Contents</h3>
    <p><b>{len(CHANGED_TABLES)}</b> tables changed, <b>{len(UNCHANGED_TABLES)}</b> unchanged</p>
    <ul>
        {"".join(toc_items)}
    </ul>
</div>

{"".join(content_sections)}

<footer style="text-align:center;color:#666;padding:20px;font-size:0.9em;">
    Generated by compare_tables.py
</footer>
</body>
</html>
"""
    return html


if __name__ == "__main__":
    html = generate_html()
    output_dir = Path(__file__).parent / "figure_comparison"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "table_comparison.html"
    output_path.write_text(html)
    print(f"Written to {output_path}")
