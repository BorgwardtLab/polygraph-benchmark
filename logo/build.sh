#!/usr/bin/env bash
set -euo pipefail

# Build TikZ logos to PDF, SVG, and PNG
# - PDF via pdflatex
# - SVG via dvisvgm (from PDF)
# - PNG via rsvg-convert/inkscape/ImageMagick (from SVG)
# Also cleans LaTeX intermediate files (.aux, .log, etc.) in relevant folders
#
# Requires: pdflatex, dvisvgm
# Optional for PNG: rsvg-convert OR inkscape OR ImageMagick (magick/convert)
#
# Usage: ./build.sh
# After build, PNGs are moved into this logo directory.

cd "$(dirname "$0")"

files=(
  "logo_icon"
  "logo_full"
  "logo"
)

THEME=${THEME:-}
PALETTE=${PALETTE:-}
PNG_DPI=${PNG_DPI:-1200}
KEEP_INTERMEDIATE=${KEEP_INTERMEDIATE:-0}

# Output directories
OUT_DIR_ROOT="build"
OUT_PDF_DIR="$OUT_DIR_ROOT/pdf"
OUT_SVG_DIR="$OUT_DIR_ROOT/svg"
OUT_PNG_DIR="$OUT_DIR_ROOT/png"
mkdir -p "$OUT_PDF_DIR" "$OUT_SVG_DIR" "$OUT_PNG_DIR"

# Known palettes (case-sensitive)
valid_palettes=(NordLight NordDark)

is_valid_palette() {
  local name="$1"
  for p in "${valid_palettes[@]}"; do
    if [[ "$name" == "$p" ]]; then
      return 0
    fi
  done
  return 1
}

# Determine which themes to build. If THEME is unset, build both.
if [[ -z "${THEME}" ]]; then
  themes=("Light" "Dark")
else
  themes=("${THEME}")
fi

to_png() {
  local svg_in="$1"
  local png_out="$2"
  # Prefer librsvg for faithful rendering and transparency
  if command -v rsvg-convert >/dev/null 2>&1; then
    rsvg-convert -a -d "$PNG_DPI" -p "$PNG_DPI" -o "$png_out" "$svg_in"
    return 0
  fi
  # Fallback to Inkscape
  if command -v inkscape >/dev/null 2>&1; then
    inkscape "$svg_in" --export-type=png -o "$png_out" --export-dpi="$PNG_DPI" --export-background-opacity=0
    return 0
  fi
  # Fallback to ImageMagick (magick or convert). Transparency may depend on build.
  if command -v magick >/dev/null 2>&1; then
    magick -background none -density "$PNG_DPI" "$svg_in" -alpha on -colorspace sRGB "$png_out"
    return 0
  fi
  if command -v convert >/dev/null 2>&1; then
    convert -background none -density "$PNG_DPI" "$svg_in" -alpha on -colorspace sRGB "$png_out"
    return 0
  fi
  echo "Warning: No SVG->PNG converter found (rsvg-convert/inkscape/magick/convert). Skipping PNG for $svg_in" 1>&2
  return 1
}

cleanup_intermediates() {
  # Remove LaTeX intermediates in the logo directory and build/pdf
  # Patterns: aux, log, synctex.gz, fdb_latexmk, fls, out, toc, nav, snm
  find . \
    -type f \
    \( -name '*.aux' -o -name '*.log' -o -name '*.synctex.gz' -o -name '*.fdb_latexmk' -o -name '*.fls' -o -name '*.out' -o -name '*.toc' -o -name '*.nav' -o -name '*.snm' \) \
    -print -delete || true
}

for theme in "${themes[@]}"; do
  for name in "${files[@]}"; do
    [[ -f "$name.tex" ]] || continue
    # Default palette follows current theme if not provided explicitly
    if [[ -z "$PALETTE" ]]; then
      palette="Nord${theme}"
    else
      palette="$PALETTE"
    fi
    # Validate palette
    if ! is_valid_palette "$palette"; then
      echo "Error: Unknown palette '$palette'. Valid palettes: ${valid_palettes[*]}" 1>&2
      exit 2
    fi
    job="${name}_${theme}_${palette}"
    echo "==> Building $job.pdf (pdflatex)"
    pdflatex -interaction=nonstopmode -halt-on-error -jobname "$job" -output-directory "$OUT_PDF_DIR" "\\def\\THEME{$theme} \\def\\PALETTE{$palette} \\input{$name.tex}" >/dev/null
    echo "==> Converting $job.pdf -> $job.svg (dvisvgm)"
    dvisvgm --pdf --no-fonts --exact -o "$OUT_SVG_DIR/$job.svg" "$OUT_PDF_DIR/$job.pdf" >/dev/null
    echo "==> Converting $job.svg -> $job.png"
    to_png "$OUT_SVG_DIR/$job.svg" "$OUT_PNG_DIR/$job.png" || true
  done
done

if [[ "$KEEP_INTERMEDIATE" -eq 0 ]]; then
  echo "==> Cleaning LaTeX intermediate files"
  cleanup_intermediates
fi

echo "==> Moving PNGs to $(pwd)"
if compgen -G "$OUT_PNG_DIR/*.png" >/dev/null; then
  mv -f "$OUT_PNG_DIR"/*.png .
else
  echo "No PNGs to move from $OUT_PNG_DIR"
fi

echo "Done. PDFs: $OUT_PDF_DIR, SVGs: $OUT_SVG_DIR, PNGs moved to $(pwd)."
