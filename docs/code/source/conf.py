# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import pytorch_sphinx_theme

sys.path.insert(0, os.path.abspath("../../.."))


# -- Project information -----------------------------------------------------

project = "torchdata"
copyright = "2019, Szymon Maszke"
author = "Szymon Maszke"
version = "0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.katex",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

html_theme_options = {
    "related": "https://szymonmaszke.github.io/torchdata/related.html",
    "roadmap": "https://github.com/szymonmaszke/torchdata/blob/master/ROADMAP.md",
    "github_issues": "https://github.com/szymonmaszke/torchdata/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc",
    "home": "https://szymonmaszke.github.io/torchdata",
    "installation": "https://szymonmaszke.github.io/torchdata/#installation",
    "github": "https://github.com/szymonmaszke/torchdata",
    "docs": "https://szymonmaszke.github.io/torchdata/#torchdata",
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": False,
    "canonical_url": "https://szymonmaszke.github.io/torchdata/",
}

# Other settings

default_role = "py:obj"  # Reference to Python by default
