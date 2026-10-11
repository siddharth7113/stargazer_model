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
    "sphinx.ext.mathjax",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "tasklist", "dollarmath", "amsmath"]
myst_dmath_double_inline = True
math_numfig = True
math_eqref_format = "Eq. {number}"
numfig = True
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 3

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

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
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_css_files = ["custom.css"]
html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#2a9d8f",
        "color-brand-content": "#1d6f86",
        "color-background-primary": "#fffdf8",
        "color-background-secondary": "#fbf6ec",
        "color-admonition-background": "#fff7e6",
        "font-stack": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif",
        "font-stack--monospace": "JetBrains Mono, SFMono-Regular, Menlo, Consolas, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5cc8b8",
        "color-brand-content": "#8fd3e8",
        "color-background-primary": "#111722",
        "color-background-secondary": "#161e2c",
    },
    "source_repository": "https://github.com/siddharth7113/stargazer_model",
    "source_branch": "main",
    "source_directory": "docs/",
}
pygments_style = "friendly"
pygments_dark_style = "monokai"
copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True
mermaid_init_js = (
    "mermaid.initialize({startOnLoad:true, theme:'neutral', "
    "themeVariables:{fontFamily:'Inter, sans-serif'}});"
)
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]
