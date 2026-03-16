#!/usr/bin/env python3
"""Generate HTML diff report: rebuttal_iclr vs current camera-ready (v2.5).

Compares all matching figures (PDF→PNG with red overlay) and tables
(parsed LaTeX → side-by-side HTML table with per-cell % change).
"""

import base64
import html as html_mod
import io
import re
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

REBUTTAL = Path("/fs/pool/pool-hartout/Documents/papers/rebuttal_iclr")
CAMERA_READY = Path(
    "/fs/pool/pool-hartout/Documents/papers/polygraph_iclr_paper"
)
REPO = Path(
    "/fs/gpfs41/lv11/fileset01/pool/pool-hartout/Documents/Git/polygraph-benchmark/reproducibility"
)
OUT_HTML = REPO / "rebuttal_vs_camera_ready_diff.html"

DPI = 150

# ═══════════════════════════════════════════════════════════════════════════
# Figure helpers
# ═══════════════════════════════════════════════════════════════════════════


def find_figure_pairs():
    pairs = []
    rebuttal_figs = REBUTTAL / "figures"
    camera_figs = CAMERA_READY / "figures"
    if not rebuttal_figs.exists():
        return pairs
    for section_dir in sorted(rebuttal_figs.iterdir()):
        if not section_dir.is_dir():
            continue
        section_name = section_dir.name
        # Camera-ready uses the same section directory names as rebuttal
        cr_dir = camera_figs / section_name
        for pdf in sorted(section_dir.glob("*.pdf")):
            cr_path = cr_dir / pdf.name
            if cr_path.exists():
                pairs.append((pdf, cr_path, f"{section_name}/{pdf.name}"))
    return pairs


def find_table_pairs():
    pairs = []
    rebuttal_tables = REBUTTAL / "tables"
    camera_tables = CAMERA_READY / "tables"
    if not rebuttal_tables.exists():
        return pairs
    for tex in sorted(rebuttal_tables.glob("*.tex")):
        cr_path = camera_tables / tex.name
        if cr_path.exists():
            pairs.append((tex, cr_path, tex.name))
    return pairs


def pdf_to_image(pdf_path: Path) -> Image.Image:
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    zoom = DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def compute_diff_image(old_img, new_img):
    w, h = (
        max(old_img.width, new_img.width),
        max(old_img.height, new_img.height),
    )
    old_r = Image.new("RGB", (w, h), (255, 255, 255))
    new_r = Image.new("RGB", (w, h), (255, 255, 255))
    old_r.paste(old_img, (0, 0))
    new_r.paste(new_img, (0, 0))
    old_arr = np.array(old_r, dtype=np.float32)
    new_arr = np.array(new_r, dtype=np.float32)
    diff = np.sqrt(np.sum((old_arr - new_arr) ** 2, axis=2))
    changed = diff > 15.0
    overlay = new_arr.copy()
    overlay[changed, 0] = np.clip(overlay[changed, 0] * 0.3 + 255 * 0.7, 0, 255)
    overlay[changed, 1] *= 0.3
    overlay[changed, 2] *= 0.3
    # Also mark changed pixels on the OLD side (for the rebuttal column)
    overlay[~changed] = overlay[~changed]  # no-op, but keeps logic clear
    pct = changed.sum() / changed.size * 100
    return Image.fromarray(overlay.astype(np.uint8)), pct


def img_to_uri(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ═══════════════════════════════════════════════════════════════════════════
# LaTeX table parser → structured cells with numeric extraction
# ═══════════════════════════════════════════════════════════════════════════


def _strip_latex(s: str) -> str:
    """Strip LaTeX formatting commands, return cleaned text."""
    s = s.strip()
    # Remove comments
    if s.startswith("%"):
        return ""
    # Remove common wrappers
    for cmd in [
        r"\textbf",
        r"\textsc",
        r"\bfseries",
        r"\underline",
        r"\multirow",
        r"\multicolumn",
        r"\scriptstyle",
        r"\scalebox",
        r"\resizebox",
    ]:
        s = s.replace(cmd, "")
    # Remove \# (LaTeX escaped hash)
    s = s.replace(r"\#", "#")
    # Remove \pm etc for display but keep structure
    s = re.sub(r"\\[a-zA-Z]+", " ", s)  # remove remaining commands
    s = s.replace("\\,", " ")  # remove \, (thin space)
    s = re.sub(r"[{}\$#]", "", s)  # remove braces, $, and #
    s = re.sub(r"\s+", " ", s).strip()
    return s


_NUM_RE = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")


def _extract_number(cell_text: str) -> float | None:
    """Extract the primary numeric value from a cell (first number found)."""
    clean = _strip_latex(cell_text)
    m = _NUM_RE.search(clean)
    if m:
        try:
            return float(m.group())
        except ValueError:
            return None
    return None


def _extract_mean_std(cell_text: str) -> tuple[float | None, float | None]:
    """Extract mean and std from a cell like '34.1 $\\pm\\,\\scriptstyle{1.7}$'.

    Returns (mean, std). std is None if no ± pattern is found.
    """
    clean = _strip_latex(cell_text)
    numbers = _NUM_RE.findall(clean)
    if not numbers:
        return None, None
    mean = float(numbers[0])
    # If there's a ± pattern, the second number is the std
    if len(numbers) >= 2 and ("pm" in cell_text or "±" in clean):
        return mean, float(numbers[1])
    return mean, None


def _parse_tex_table(text: str) -> list[list[str]]:
    """Parse a LaTeX table into a list of rows, each a list of cell strings.

    Handles commented-out lines (% prefix) and multiline cells.
    Returns only data rows (between \\midrule or \\toprule and \\bottomrule).
    """
    lines = text.splitlines()
    # Un-comment lines that start with %
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("%"):
            stripped = stripped[1:].strip()
        cleaned.append(stripped)

    full = "\n".join(cleaned)

    # Find everything between \toprule and \bottomrule
    match = re.search(r"\\toprule(.*?)\\bottomrule", full, re.DOTALL)
    if not match:
        return []

    body = match.group(1)

    # Split on \\ (row delimiter)
    raw_rows = re.split(r"\\\\", body)

    rows = []
    for raw in raw_rows:
        raw = raw.strip()
        if not raw:
            continue
        # Strip structural commands (\midrule, \cmidrule) from within the
        # chunk instead of skipping the whole chunk — the actual row data
        # often follows these commands in the same chunk.
        raw = re.sub(r"\\cmidrule\([^)]*\)\{[^}]*\}", "", raw)
        raw = re.sub(r"\\(midrule|toprule|bottomrule|hline)\b", "", raw)
        raw = raw.strip()
        if not raw:
            continue

        cells = [c.strip() for c in raw.split("&")]
        # Skip rows that are purely structural (all empty or all commands)
        text_content = "".join(_strip_latex(c) for c in cells)
        if text_content.strip():
            rows.append(cells)

    return rows


def _cell_delta_html(old_cell: str, new_cell: str) -> str:
    """Render a single cell comparison as HTML with % change and within-std annotation."""
    old_val = _extract_number(old_cell)
    new_val = _extract_number(new_cell)
    old_display = _strip_latex(old_cell) or "-"
    new_display = _strip_latex(new_cell) or "-"

    if old_display == new_display:
        # Identical
        return f'<td class="cell-same">{html_mod.escape(new_display)}</td>'

    if old_val is not None and new_val is not None:
        abs_diff = new_val - old_val
        if old_val != 0:
            pct = (abs_diff / abs(old_val)) * 100
            pct_str = f"{pct:+.1f}%"
        else:
            pct_str = f"{abs_diff:+.2f}"

        # Check if change is within std
        old_mean, old_std = _extract_mean_std(old_cell)
        new_mean, new_std = _extract_mean_std(new_cell)
        within_std_tag = ""
        if old_mean is not None and new_mean is not None:
            # Use the larger std of the two as the reference
            stds = [s for s in (old_std, new_std) if s is not None]
            if stds:
                max_std = max(stds)
                diff = abs(new_mean - old_mean)
                if max_std > 0 and diff <= max_std:
                    within_std_tag = '<span class="within-std" title="Change is within 1 std">&#x2714; within std</span>'
                elif max_std > 0:
                    n_std = diff / max_std
                    within_std_tag = f'<span class="outside-std" title="Change is {n_std:.1f}x std">{n_std:.1f}x std</span>'

        # Color by magnitude of change
        abs_pct = abs(pct) if old_val != 0 else abs(abs_diff)
        if abs_pct < 2:
            cls = "cell-minor"
        elif abs_pct < 10:
            cls = "cell-moderate"
        else:
            cls = "cell-major"

        return (
            f'<td class="{cls}">'
            f'<span class="val-old">{html_mod.escape(old_display)}</span>'
            f'<span class="val-arrow">→</span>'
            f'<span class="val-new">{html_mod.escape(new_display)}</span>'
            f'<span class="val-delta">({pct_str})</span>'
            f"{within_std_tag}"
            f"</td>"
        )
    else:
        # Non-numeric change (e.g. label changed)
        if old_display.strip() == "-" or new_display.strip() == "-":
            cls = "cell-moderate"
        else:
            cls = "cell-text-change"
        return (
            f'<td class="{cls}">'
            f'<span class="val-old">{html_mod.escape(old_display)}</span>'
            f'<span class="val-arrow">→</span>'
            f'<span class="val-new">{html_mod.escape(new_display)}</span>'
            f"</td>"
        )


_COLUMN_RENAMES = {
    "pgs": "pgd",
    "pgs-concat.": "pgd-concat.",
    "pgs-concat": "pgd-concat",
    "pgs-gklr": "pgd-gklr",
    "tv-pgs": "tv-pgd",
    "spec.": "eig.",
    "spec. rbf": "eig. rbf",
    "spec. pgs": "eig.",
    "validity": "vun",
    "val.": "vun",
    "rbf mmd^2 deg.": "rbf deg.",
    "rbf mmd^2 clust.": "rbf clust.",
    "rbf mmd^2 orb.": "rbf orb.",
    "rbf mmd^2 eig.": "rbf eig.",
    "gtv mmd^2 deg.": "gtv deg.",
    "gtv mmd^2 clust.": "gtv clust.",
    "gtv mmd^2 orb.": "gtv orb.",
    "gtv mmd^2 eig.": "gtv eig.",
    "steps": "steps",
    "# steps": "steps",
    "orbit rbf": "orb. rbf",
    "orbit pgs": "orb.",
    "orbit5 pgs": "orb5.",
    "orbit4 pgs": "orb4.",
    "orbit pgs5": "orb5.",
    "clust. pgs": "clust.",
    "deg. pgs": "deg.",
    "gin pgs": "gin",
}


def _normalize_header(h: str) -> str:
    """Normalize a header string for matching across old/new tables."""
    h = _strip_latex(h).strip().lower()
    # Remove direction indicators
    h = re.sub(r"\s*\(?\s*[↓↑]\s*\)?\s*", "", h)
    h = re.sub(r"\s*\(\s*\)\s*", "", h)
    h = h.strip(" ()")
    # Apply known renames
    return _COLUMN_RENAMES.get(h, h)


def _is_header_row(row: list[str]) -> bool:
    """Detect if a row is a header row (not data).

    A row is a header if:
    - It contains multicolumn commands, or
    - It uses \\textbf formatting on non-numeric cells, or
    - Most of its cells (beyond the first) are non-numeric text labels.
    """
    if not row:
        return False
    first = row[0].strip()
    if "multicolumn" in first.lower():
        return True
    if "\\textbf" in first:
        has_number = _NUM_RE.search(_strip_latex(first))
        if not has_number:
            # Before declaring this a header, check if other cells have numbers.
            # Data rows often have bolded labels in the first column (e.g.
            # \textbf{PGD} & 0.6 & ...) — those are NOT headers.
            if len(row) > 1 and any(
                _NUM_RE.search(_strip_latex(c)) for c in row[1:]
            ):
                pass  # data row with bolded label
            else:
                return True
    # Fallback: check if cells are primarily non-numeric (data rows have mostly numbers).
    # A cell is "numeric" if its stripped content is predominantly a number (e.g. "0.1879"),
    # not just containing a digit somewhere (e.g. "Orbit5 PGS" contains "5" but is a label).
    if len(row) > 1:

        def _is_numeric_cell(c: str) -> bool:
            clean = _strip_latex(c).strip()
            if not clean or clean == "-":
                return False
            # A data cell contains at least one number and is mostly numeric
            nums = _NUM_RE.findall(clean)
            if not nums:
                return False
            num_chars = sum(len(n) for n in nums)
            return num_chars > len(clean) * 0.3

        numeric_count = sum(1 for c in row[1:] if _is_numeric_cell(c))
        if numeric_count == 0:
            return True
    return False


def _build_column_mapping(
    old_headers: list[str], new_headers: list[str]
) -> list[tuple[int, int]]:
    """Build a mapping of (old_col_idx, new_col_idx) pairs for columns that match.

    Returns a list of (old_idx, new_idx) tuples for matched columns,
    preserving order of the new table.
    """
    old_norm = [_normalize_header(h) for h in old_headers]
    new_norm = [_normalize_header(h) for h in new_headers]

    mapping = []
    used_old = set()

    for new_idx, new_name in enumerate(new_norm):
        for old_idx, old_name in enumerate(old_norm):
            if old_idx in used_old:
                continue
            if old_name == new_name:
                mapping.append((old_idx, new_idx))
                used_old.add(old_idx)
                break

    return mapping


def render_table_diff(old_path: Path, new_path: Path) -> tuple[str, dict]:
    """Parse two LaTeX tables and render an HTML diff table.

    Uses column-header matching to align columns correctly even when
    columns are added, removed, or reordered between versions.

    Returns (html_string, stats_dict).
    """
    old_rows = _parse_tex_table(old_path.read_text())
    new_rows = _parse_tex_table(new_path.read_text())

    if not old_rows and not new_rows:
        return "<p>Could not parse table.</p>", {
            "cells_changed": 0,
            "cells_total": 0,
        }

    stats = {
        "cells_changed": 0,
        "cells_total": 0,
        "max_pct": 0.0,
        "changes": [],
        "top_changes": [],
    }

    # Detect header rows — find ALL consecutive header rows at the top.
    # Use the one with the most cells for column mapping (e.g. the subheader
    # row "VUN & PGD & Clust ..." rather than a multicolumn row).
    def _find_headers(rows):
        """Return (best_header_idx, data_start_idx) or (None, 0)."""
        header_indices = []
        for i, row in enumerate(rows):
            if _is_header_row(row):
                header_indices.append(i)
            else:
                break  # stop at first non-header row
        if not header_indices:
            return None, 0
        # Pick the header with the most cells (most detailed for column mapping)
        best = max(header_indices, key=lambda i: len(rows[i]))
        data_start = header_indices[-1] + 1
        return best, data_start

    old_header_idx, old_data_start = _find_headers(old_rows)
    new_header_idx, new_data_start = _find_headers(new_rows)

    # Build column mapping from headers
    col_mapping = None  # (old_idx, new_idx) pairs
    if old_header_idx is not None and new_header_idx is not None:
        old_headers = old_rows[old_header_idx]
        new_headers = new_rows[new_header_idx]
        col_mapping = _build_column_mapping(old_headers, new_headers)

    # Determine data rows (skip headers)
    old_data = old_rows[old_data_start:]
    new_data = new_rows[new_data_start:]

    html_rows = []

    # Render header row using new table's headers
    if new_header_idx is not None:
        header_cells = []
        for h in new_rows[new_header_idx]:
            display = _strip_latex(h) or "-"
            header_cells.append(f"<th>{html_mod.escape(display)}</th>")
        html_rows.append(f'<tr class="header-row">{"".join(header_cells)}</tr>')

    # Resolve header labels for change tracking
    new_header_labels = None
    if new_header_idx is not None:
        new_header_labels = [_strip_latex(h) for h in new_rows[new_header_idx]]

    def _record_change(old_c, new_c, row_idx, col_idx):
        """Track a cell change for the top-changes summary."""
        ov = _extract_number(old_c)
        nv = _extract_number(new_c)
        old_d = _strip_latex(old_c)
        new_d = _strip_latex(new_c)
        if old_d == new_d:
            return
        stats["cells_changed"] += 1
        pct = None
        if ov is not None and nv is not None and ov != 0:
            pct = abs((nv - ov) / ov * 100)
            stats["max_pct"] = max(stats["max_pct"], pct)
        # Determine row label (first cell of the data row)
        data_row = new_data[row_idx] if row_idx < len(new_data) else []
        row_label = _strip_latex(data_row[0]) if data_row else f"row {row_idx}"
        col_label = (
            new_header_labels[col_idx]
            if new_header_labels and col_idx < len(new_header_labels)
            else f"col {col_idx}"
        )
        old_mean, old_std = _extract_mean_std(old_c)
        new_mean, new_std = _extract_mean_std(new_c)
        within_std = False
        if old_mean is not None and new_mean is not None:
            stds = [s for s in (old_std, new_std) if s is not None]
            if stds and max(stds) > 0:
                within_std = abs(new_mean - old_mean) <= max(stds)
        stats["top_changes"].append(
            {
                "row": row_label,
                "col": col_label,
                "old": old_d,
                "new": new_d,
                "pct": pct,
                "within_std": within_std,
            }
        )

    # Render data rows
    max_data_rows = max(len(old_data), len(new_data))
    for i in range(max_data_rows):
        old_r = old_data[i] if i < len(old_data) else []
        new_r = new_data[i] if i < len(new_data) else []

        if col_mapping is not None:
            # Use column mapping: iterate over new columns, find matching old column
            new_cols = (
                len(new_rows[new_header_idx])
                if new_header_idx is not None
                else len(new_r)
            )
            # Build a quick lookup: new_idx -> old_idx
            new_to_old = {new_idx: old_idx for old_idx, new_idx in col_mapping}

            cells_html = []
            for j in range(max(new_cols, len(new_r))):
                new_c = new_r[j] if j < len(new_r) else ""
                old_idx = new_to_old.get(j)
                if old_idx is not None and old_idx < len(old_r):
                    old_c = old_r[old_idx]
                else:
                    old_c = new_c  # No old column match; treat as unchanged

                stats["cells_total"] += 1
                _record_change(old_c, new_c, i, j)
                cells_html.append(_cell_delta_html(old_c, new_c))
        else:
            # No header mapping available; fall back to positional comparison
            max_cols = max(len(old_r), len(new_r))
            cells_html = []
            for j in range(max_cols):
                old_c = old_r[j] if j < len(old_r) else ""
                new_c = new_r[j] if j < len(new_r) else ""

                stats["cells_total"] += 1
                _record_change(old_c, new_c, i, j)
                cells_html.append(_cell_delta_html(old_c, new_c))

        html_rows.append(f"<tr>{''.join(cells_html)}</tr>")

    table_html = f'<table class="diff-table">{"".join(html_rows)}</table>'
    return table_html, stats


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)

    fig_pairs = find_figure_pairs()
    tbl_pairs = find_table_pairs()
    print(f"Found {len(fig_pairs)} figure pairs, {len(tbl_pairs)} table pairs")

    # ── Figures ──
    fig_sections = {}
    for old_path, new_path, label in fig_pairs:
        section = label.split("/")[0]
        try:
            old_img = pdf_to_image(old_path)
            new_img = pdf_to_image(new_path)
            diff_img, pct = compute_diff_image(old_img, new_img)
            old_uri = img_to_uri(old_img)
            new_uri = img_to_uri(new_img)
            diff_uri = img_to_uri(diff_img)

            badge_class = (
                "badge-ok"
                if pct < 0.5
                else ("badge-minor" if pct < 5 else "badge-major")
            )
            badge_label = "identical" if pct < 0.1 else f"{pct:.1f}% px changed"

            block = f"""
            <div class="item">
                <div class="item-header">
                    <h3>{html_mod.escape(label)}</h3>
                    <span class="badge {badge_class}">{badge_label}</span>
                </div>
                <div class="fig-grid">
                    <div class="cell"><h4>Rebuttal</h4><img src="{old_uri}" /></div>
                    <div class="cell"><h4>Camera-ready (v2.5)</h4><img src="{new_uri}" /></div>
                    <div class="cell diff-cell"><h4>Diff (red = changed)</h4><img src="{diff_uri}" /></div>
                </div>
            </div>"""
            fig_sections.setdefault(section, []).append((pct, block))
            print(f"  Fig: {label} → {pct:.1f}%")
        except Exception as e:
            print(f"  SKIP fig {label}: {e}")

    # ── Tables ──
    tbl_blocks = []
    all_table_changes = []  # (label, pct, change_info) for biggest-changes summary
    for old_path, new_path, label in tbl_pairs:
        try:
            table_html, stats = render_table_diff(old_path, new_path)
            ct = stats["cells_total"]
            cc = stats["cells_changed"]
            mp = stats["max_pct"]

            # Collect top changes across all tables
            for ch in stats.get("top_changes", []):
                if ch["pct"] is not None:
                    all_table_changes.append((label, ch))

            if cc == 0:
                badge_class = "badge-ok"
                badge_label = "identical"
                summary = "All values identical."
            else:
                badge_class = "badge-major" if mp > 10 else "badge-minor"
                badge_label = f"{cc}/{ct} cells changed"
                summary = f"{cc} of {ct} cells differ. Max relative change: {mp:.1f}%."

            block = f"""
            <div class="item">
                <div class="item-header">
                    <h3>{html_mod.escape(label)}</h3>
                    <span class="badge {badge_class}">{badge_label}</span>
                </div>
                <p class="table-summary">{summary}</p>
                <div class="table-scroll">{table_html}</div>
                <div class="legend-bar">
                    <span class="legend-item"><span class="swatch swatch-same"></span> Unchanged</span>
                    <span class="legend-item"><span class="swatch swatch-minor"></span> &lt;2% change</span>
                    <span class="legend-item"><span class="swatch swatch-moderate"></span> 2–10% change</span>
                    <span class="legend-item"><span class="swatch swatch-major"></span> &gt;10% change</span>
                </div>
            </div>"""
            tbl_blocks.append(block)
            print(f"  Tbl: {label} → {cc}/{ct} cells, max Δ {mp:.1f}%")
        except Exception as e:
            print(f"  SKIP tbl {label}: {e}")

    # ── Build biggest-changes summary ──
    all_table_changes.sort(key=lambda x: x[1]["pct"], reverse=True)
    top_n = min(30, len(all_table_changes))
    biggest_rows = []
    for table_label, ch in all_table_changes[:top_n]:
        pct = ch["pct"]
        std_badge = (
            '<span class="within-std">within std</span>'
            if ch["within_std"]
            else '<span class="outside-std">outside std</span>'
        )
        biggest_rows.append(
            f"<tr>"
            f"<td>{html_mod.escape(table_label)}</td>"
            f"<td>{html_mod.escape(ch['row'])}</td>"
            f"<td>{html_mod.escape(ch['col'])}</td>"
            f'<td class="val-old">{html_mod.escape(ch["old"])}</td>'
            f'<td class="val-new">{html_mod.escape(ch["new"])}</td>'
            f'<td class="{"cell-major" if pct > 10 else "cell-moderate" if pct > 2 else "cell-minor"}">'
            f"{pct:.1f}%</td>"
            f"<td>{std_badge}</td>"
            f"</tr>"
        )
    biggest_changes_html = f"""
    <div class="item">
        <div class="item-header">
            <h3>Top {top_n} Biggest Relative Changes Across All Tables</h3>
        </div>
        <p class="table-summary">
            Sorted by absolute relative change. "Within std" means the difference
            between old and new mean is within the larger standard deviation of the two.
        </p>
        <div class="table-scroll">
            <table class="diff-table">
                <tr class="header-row">
                    <th>Table</th><th>Row</th><th>Column</th>
                    <th>Old</th><th>New</th><th>Rel. Change</th><th>Std Check</th>
                </tr>
                {"".join(biggest_rows)}
            </table>
        </div>
    </div>"""

    # ── Assemble HTML ──
    fig_html_parts = []
    for section in sorted(fig_sections.keys()):
        items = fig_sections[section]
        items.sort(key=lambda x: -x[0])
        total = len(items)
        changed = sum(1 for pct, _ in items if pct >= 0.1)
        fig_html_parts.append(
            f'<h2 class="section-header">{html_mod.escape(section)} '
            f'<span class="section-count">({changed}/{total} changed)</span></h2>'
        )
        for _, block in items:
            fig_html_parts.append(block)

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Rebuttal vs Camera-Ready (v2.5) Diff Report</title>
<style>
    * {{ box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: #0d1117; color: #c9d1d9;
        margin: 0 auto; padding: 20px; max-width: 1800px;
    }}
    h1 {{ text-align: center; color: #fff; border-bottom: 2px solid #e94560; padding-bottom: 12px; }}
    .summary {{ text-align: center; color: #8b949e; margin-bottom: 30px; }}
    .summary strong {{ color: #e94560; }}

    /* TOC */
    .toc {{
        background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 16px 24px; margin-bottom: 30px;
    }}
    .toc h2 {{ margin-top: 0; color: #58a6ff; font-size: 16px; }}
    .toc a {{ color: #58a6ff; text-decoration: none; }}
    .toc a:hover {{ text-decoration: underline; }}
    .toc ul {{ columns: 2; list-style: none; padding: 0; }}
    .toc li {{ padding: 2px 0; }}

    .section-header {{
        color: #58a6ff; border-bottom: 1px solid #30363d;
        padding-bottom: 6px; margin-top: 40px;
    }}
    .section-count {{ color: #8b949e; font-size: 0.7em; font-weight: normal; }}

    .item {{
        background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 16px; margin: 16px 0;
    }}
    .item-header {{
        display: flex; align-items: center; gap: 12px; margin-bottom: 12px;
    }}
    .item-header h3 {{
        margin: 0; color: #c9d1d9; font-size: 14px; font-family: monospace;
    }}
    .badge {{
        font-size: 11px; padding: 2px 8px; border-radius: 12px;
        font-weight: 600; white-space: nowrap;
    }}
    .badge-ok {{ background: #238636; color: #fff; }}
    .badge-minor {{ background: #d29922; color: #000; }}
    .badge-major {{ background: #da3633; color: #fff; }}

    /* Figure grid */
    .fig-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }}
    .cell {{
        background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
        padding: 8px; text-align: center;
    }}
    .cell h4 {{ margin: 0 0 6px 0; color: #8b949e; font-size: 12px; }}
    .cell img {{ width: 100%; border-radius: 4px; background: white; }}
    .diff-cell {{ border-color: #da3633; }}

    /* Table diff */
    .table-summary {{ color: #8b949e; font-size: 13px; margin: 0 0 10px 0; }}
    .table-scroll {{ overflow-x: auto; }}
    .diff-table {{
        border-collapse: collapse; width: 100%; font-size: 12px;
        font-family: "SF Mono", "Fira Code", monospace;
    }}
    .diff-table th, .diff-table td {{
        padding: 6px 10px; border: 1px solid #30363d; text-align: center;
        white-space: nowrap;
    }}
    .diff-table .header-row th {{
        background: #21262d; color: #58a6ff; font-weight: 600;
        font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px;
    }}
    .cell-same {{ color: #8b949e; }}
    .cell-minor {{
        background: rgba(210, 153, 34, 0.08);
    }}
    .cell-moderate {{
        background: rgba(210, 153, 34, 0.18);
    }}
    .cell-major {{
        background: rgba(218, 54, 51, 0.18);
    }}
    .cell-text-change {{
        background: rgba(88, 166, 255, 0.12);
    }}
    .val-old {{
        color: #f85149; text-decoration: line-through; font-size: 11px;
    }}
    .val-arrow {{
        color: #484f58; margin: 0 3px; font-size: 10px;
    }}
    .val-new {{
        color: #3fb950; font-weight: 600;
    }}
    .val-delta {{
        display: block; font-size: 10px; color: #d29922; margin-top: 1px;
    }}
    .within-std {{
        display: block; font-size: 9px; color: #3fb950; margin-top: 1px;
    }}
    .outside-std {{
        display: block; font-size: 9px; color: #f85149; margin-top: 1px;
    }}

    /* Legend */
    .legend-bar {{
        display: flex; gap: 16px; margin-top: 10px; padding: 6px 0;
        font-size: 11px; color: #8b949e;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 4px; }}
    .swatch {{
        display: inline-block; width: 14px; height: 14px;
        border-radius: 3px; border: 1px solid #30363d;
    }}
    .swatch-same {{ background: #161b22; }}
    .swatch-minor {{ background: rgba(210, 153, 34, 0.25); }}
    .swatch-moderate {{ background: rgba(210, 153, 34, 0.45); }}
    .swatch-major {{ background: rgba(218, 54, 51, 0.4); }}

    /* Nav */
    .nav {{ position: fixed; bottom: 20px; right: 20px; z-index: 999; }}
    .nav a {{
        display: block; background: #30363d; color: #58a6ff;
        padding: 8px 14px; border-radius: 6px; text-decoration: none;
        font-size: 13px; margin-top: 4px;
    }}
    .nav a:hover {{ background: #484f58; }}
</style>
</head>
<body>
<h1>Rebuttal vs Camera-Ready (v2.5) — Full Diff Report</h1>
<p class="summary">
    Comparing <strong>{len(fig_pairs)} figures</strong> and <strong>{len(tbl_pairs)} tables</strong>
    between the ICLR rebuttal and camera-ready (TabPFN weights v2.5).
</p>
<div class="toc" id="top">
    <h2>Table of Contents</h2>
    <ul>
        <li><a href="#biggest-changes">Biggest Changes (Top {top_n})</a></li>
        <li><a href="#tables">Tables ({len(tbl_pairs)})</a></li>
        <li><a href="#figures">Figures ({len(fig_pairs)})</a></li>
    </ul>
</div>

<h1 id="biggest-changes">Biggest Changes</h1>
{biggest_changes_html}

<h1 id="tables">Tables</h1>
{"".join(tbl_blocks)}

<h1 id="figures">Figures</h1>
{"".join(fig_html_parts)}

<div class="nav">
    <a href="#top">Top</a>
    <a href="#biggest-changes">Biggest</a>
    <a href="#tables">Tables</a>
    <a href="#figures">Figures</a>
</div>
</body>
</html>"""

    OUT_HTML.write_text(page)
    size_mb = OUT_HTML.stat().st_size / 1024 / 1024
    print(f"\nHTML report: {OUT_HTML} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
