import os
import re
import pathlib
import textwrap
from typing import Iterator, List, Tuple

# ----------------------------
# Configuration
# ----------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"

# Regex to match triple-backticked code blocks:
# - PYTHON_ONLY matches blocks explicitly labeled as Python (python|py|python3|pycon|ipython)
# - ANY_LABEL_OR_UNLABELED matches python-labeled OR unlabeled blocks
PYTHON_CODE_BLOCK_PATTERN = re.compile(
    r"```\s*(?:python|py|python3|pycon|ipython)(?:[^\n]*)?\s*\n(.*?)```",
    re.DOTALL,
)
ANY_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:\s*(?:python|py|python3|pycon|ipython)(?:[^\n]*)?)?\s*\n(.*?)```",
    re.DOTALL,
)

# Directories to skip when walking the docs tree (common build outputs)
SKIP_DIRS = {"_build", ".git", ".venv", "build", "site", "dist", "node_modules"}


def _extract_code_blocks_from_markdown(
    markdown_text: str, only_python: bool = True
) -> List[str]:
    """Extract triple-backtick code blocks from a markdown string and fix indentation.

    If only_python is True, only code blocks explicitly labeled as Python are captured.
    Otherwise, unlabeled code blocks are captured as well.
    """
    pattern = (
        PYTHON_CODE_BLOCK_PATTERN if only_python else ANY_CODE_BLOCK_PATTERN
    )
    code_blocks = pattern.findall(markdown_text or "")

    fixed_blocks: List[str] = []
    for code in code_blocks:
        dedented_code = textwrap.dedent(code)
        fixed_blocks.append(dedented_code)

    return fixed_blocks


def _iter_markdown_files(root: pathlib.Path) -> Iterator[pathlib.Path]:
    """Yield markdown files (.md, .mdx) under root, skipping common build dirs."""
    if not root.exists():
        return

    for dirpath, dirnames, filenames in os.walk(root):
        # In-place prune dirs to skip
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for filename in filenames:
            if filename.lower().endswith((".md", ".mdx")):
                yield pathlib.Path(dirpath) / filename


def gather_markdown_snippets(only_python: bool = True) -> List[Tuple[str, str]]:
    """Gather all triple-backtick code snippets from markdown docs in `docs/`.

    For each markdown file, concatenate all discovered code blocks into a single
    Python snippet. Returns a list of (file_identifier, concatenated_code) tuples.
    The identifier is the path to the markdown file relative to the repo root.

    Parameters
    ----------
    only_python: bool
        If True (default), only capture code fences labeled as Python. If False,
        also capture unlabeled fences.
    """
    print("Gathering markdown snippets from docs/ ...")

    snippets: List[Tuple[str, str]] = []

    if not DOCS_DIR.exists():
        print(f"docs/ directory not found at {DOCS_DIR}")
        return snippets

    files = list(_iter_markdown_files(DOCS_DIR))
    print(f"Found {len(files)} markdown files to process")

    for index, md_path in enumerate(sorted(files)):
        rel_path = md_path.relative_to(BASE_DIR)
        print(f"[{index + 1}/{len(files)}] Processing: {rel_path}")
        try:
            text = md_path.read_text(encoding="utf-8")
        except Exception as exc:
            print(
                f"  -> Failed to read {rel_path}: {type(exc).__name__}: {exc}"
            )
            # Treat read errors as test cases too for visibility
            snippets.append(
                (
                    f"{rel_path} (read error)",
                    f"# read error\nraise IOError({repr(str(exc))})",
                )
            )
            continue

        code_blocks = _extract_code_blocks_from_markdown(
            text, only_python=only_python
        )
        print(f"  -> Found {len(code_blocks)} code blocks")

        if code_blocks:
            # Concatenate all blocks with spacing to avoid accidental token pasting
            concatenated = (
                "\n\n".join(block.strip("\n") for block in code_blocks) + "\n"
            )
            identifier = f"{rel_path}"
            snippets.append((identifier, concatenated))
            print(f"    -> Added concatenated snippet for: {identifier}")

    print(f"Finished gathering snippets. Total: {len(snippets)} snippets")
    return snippets


__all__ = [
    "gather_markdown_snippets",
]
