"""Sphinx configuration for Star Gazer."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Star Gazer"
author = "Siddharth"
copyright = "2026, Siddharth"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.mermaid",
]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "tasklist"]
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
}

html_theme = "furo"
html_title = "Star Gazer"
html_static_path = ["_static"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]
