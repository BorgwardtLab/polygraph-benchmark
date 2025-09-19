import os
import sys

sys.path.insert(0, os.path.abspath(".."))


project = "PolyGraph"
copyright = "2025, PolyGraph"
author = "PolyGraph contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

autosummary_generate = True
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 3

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "show_prev_next": False,
}

html_static_path = ["_static"]

