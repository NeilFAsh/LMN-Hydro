# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath('../../')
sys.path.insert(0, PROJECT_ROOT)

# Paths for Sphinx to find code
INCLUDE_DIR = os.path.abspath('../../include')
sys.path.insert(0, INCLUDE_DIR)
SRC_DIR = os.path.abspath('../../src')
sys.path.insert(0, SRC_DIR)
PY_DIR = os.path.abspath('../../py')
sys.path.insert(0, PY_DIR)
EXAMPLES_DIR = os.path.abspath('../../examples')
sys.path.insert(0, EXAMPLES_DIR)


# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LMN-Hydro'
copyright = '2025, Neil Ash, Marbely Micolta, Levi Walls'
author = 'Neil Ash, Marbely Micolta, Levi Walls'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'breathe',             # connects Sphinx with Doxygen
    'sphinx.ext.autodoc',
    # 'sphinx.ext.napoleon',
    # 'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Breathe configuration ---------------------------------------------------

# breathe_projects = {
#     "LMN-Hydro": os.path.join(PROJECT_ROOT, "docs", "doxygen", "xml")
# }
# breathe_default_project = "LMN-Hydro"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
