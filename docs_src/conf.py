"""Sphinx configuration for T1D Analytics documentation."""

import os
import sys

# Ensure src is discoverable for autodoc
sys.path.insert(0, os.path.abspath("../src"))

project = "T1D Analytics"
copyright = "2026, Samuel Marks"
author = "Samuel Marks"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

suppress_warnings = [
    "docutils",
    "ref.python",
    "ref.duplicate",
    "autodoc",
    "autodoc.import_object",
    "duplicate_declaration",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = []
