import os
import re
import types
import importlib
import pathlib
import textwrap

# ----------------------------
# Configuration
# ----------------------------
ROOT_PACKAGE = "polygraph"  # ← Your top-level package name
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent  # Adjust if needed

# Regex to match triple-backticked python code blocks
CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_code_blocks_from_docstring(docstring):
    """Extract triple-backtick code blocks from a docstring and fix indentation."""
    code_blocks = CODE_BLOCK_PATTERN.findall(docstring or "")

    # Fix indentation for each code block
    fixed_blocks = []
    for code in code_blocks:
        # Use textwrap.dedent to remove common leading whitespace
        dedented_code = textwrap.dedent(code)
        fixed_blocks.append(dedented_code)

    return fixed_blocks


def extract_all_docstrings_from_module(module):
    """Get all docstrings from a module, its classes, and functions."""
    docstrings = []

    if module.__doc__:
        docstrings.append((module.__name__, module.__doc__))

    for name in dir(module):
        try:
            obj = getattr(module, name)
        except Exception:
            continue
        if isinstance(obj, (types.FunctionType, types.MethodType, type)):
            if obj.__doc__:
                qualname = f"{module.__name__}.{name}"
                docstrings.append((qualname, obj.__doc__))
    return docstrings


def discover_modules(package):
    """Recursively discover all modules in a package."""
    package_path = BASE_DIR / package.replace(".", "/")
    for root, _, files in os.walk(package_path):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                rel_path = os.path.relpath(os.path.join(root, file), BASE_DIR)
                module_name = rel_path.replace(os.sep, ".")[:-3]
                yield module_name


def gather_docstring_snippets():
    """Gather all docstring code snippets across the package."""
    print("Gathering docstring snippets...")
    print("Starting module discovery...")

    # Skip modules that are known to cause issues during pytest collection
    SKIP_MODULES = {
        "polygraph.datasets.base.molecules",  # Contains RDKit imports that can hang
        "polygraph.datasets.molecules",  # Also likely contains RDKit
    }

    snippets = []
    modules_list = list(discover_modules(ROOT_PACKAGE))
    print(f"Found {len(modules_list)} modules to process")

    for i, module_name in enumerate(modules_list):
        if module_name in SKIP_MODULES:
            print(
                f"[{i + 1}/{len(modules_list)}] Skipping module: {module_name} (in skip list)"
            )
            continue

        print(f"[{i + 1}/{len(modules_list)}] Processing module: {module_name}")
        try:
            print(f"  -> Importing {module_name}...")
            module = importlib.import_module(module_name)
            print(f"  -> Import successful for {module_name}")
        except Exception as e:
            print(
                f"  -> Import failed for {module_name}: {type(e).__name__}: {e}"
            )
            # Treat import errors as test cases too
            snippets.append(
                (
                    f"{module_name} (import error)",
                    f"# import error\nraise ImportError({repr(str(e))})",
                )
            )
            continue

        print(f"  -> Extracting docstrings from {module_name}...")
        docstring_count = 0
        for qualname, docstring in extract_all_docstrings_from_module(module):
            docstring_count += 1
            print(f"    -> Found docstring: {qualname}")
            code_blocks = extract_code_blocks_from_docstring(docstring)
            print(f"    -> Found {len(code_blocks)} code blocks in {qualname}")
            for i, code in enumerate(code_blocks):
                id_str = f"{qualname} [snippet #{i + 1}]"
                snippets.append((id_str, code))
                print(f"      -> Added snippet: {id_str}")
        print(f"  -> Processed {docstring_count} docstrings from {module_name}")

    print(f"Finished gathering snippets. Total: {len(snippets)} snippets")
    return snippets
