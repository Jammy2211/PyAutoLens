import datetime

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from pyprojroot import here

workspace_path = str(here())

import os
import sys

os.system("git clone https://github.com/Jammy2211/autolens_workspace --depth 1")
os.system(f"cp -r autolens_workspace/dataset {workspace_path}")
os.system(f"cp -r autolens_workspace/config {workspace_path}")
os.system("rm -rf autolens_workspace")

sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------

year = datetime.date.today().year
project = "PyAutoLens"
copyright = "2020, James Nightingale, Richard Hayes"
author = "James Nightingale, Richard Hayes"

# The full version, including alpha/beta/rc tags
release = "1.0.16"
master_doc = "index"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "numpydoc",
    "nbsphinx",
    "nbsphinx_link",
]

## Generate autodoc stubs with summaries from code
autosummary_generate = True

## Include Python objects as they appear in source files
autodoc_member_order = "bysource"

## Default flags used by autodoc directives
autodoc_default_flags = ["members"]

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

nbsphinx_allow_errors = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


sphinx_gallery_conf = {
    # Remove the "Download all examples" button from the top level gallery
    "download_all_examples": False,
    # directory where function granular galleries are stored
    "backreferences_dir": "api/generated/backreferences",
    # Modules for which function level galleries are created.
    "doc_module": "pyautolens",
    # Insert links to documentation of objects in the examples
    "reference_url": {"pyautolens": None},
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_last_updated_fmt = "%b %d, %Y"
html_title = "PyAutoLens"
html_short_title = "PyAutoLens"
pygments_style = "default"
add_function_parentheses = False
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

html_context = {
    "menu_links_name": "Repository",
    # Custom variables to enable "Improve this page"" and "Download notebook"
    # links
    "doc_path": "docs",
    "github_project": "pyautolens",
    "github_repo": "pyautolens",
    "github_version": "development",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- IMAGES --


from sphinx.builders.html import StandaloneHTMLBuilder

StandaloneHTMLBuilder.supported_image_types = ["image/gif", "image/png", "image/jpeg"]
