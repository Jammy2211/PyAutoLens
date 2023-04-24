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

# /home/docs/checkouts/readthedocs.org/user_builds/pyautolens/checkouts/latest/docs

clone_path = str(here())

import os
import sys

sys.path.insert(0, os.path.abspath("."))


def clone_repo(name: str, url: str):
    clone = f"git clone {url}/{name}"
    os.system(clone)
    os.system(f"pip install -r {name}/requirements.txt")
    os.system(f"rm -rf {name}/docs")
    sys.path.insert(
        0,
        os.path.abspath(f"{clone_path}/{name}"),
    )


clone_repo(name="PyAutoFit", url="https://github.com/rhayes777")
clone_repo(name="PyAutoArray", url="https://github.com/Jammy2211")
clone_repo(name="PyAutoGalaxy", url="https://github.com/Jammy2211")

clone_path = os.path.split(clone_path)[0]

sys.path.insert(
    0,
    os.path.abspath(clone_path),
)

import autolens

# -- Project information -----------------------------------------------------

year = datetime.date.today().year
project = "PyAutoLens"
copyright = "2022, James Nightingale, Richard Hayes"
author = "James Nightingale, Richard Hayes"

# The full version, including alpha/beta/rc tags
release = autolens.__version__
master_doc = "index"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",  # Automatically document param types (less noise in class signature)
    "numpydoc",
    # External stuff
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_inline_tabs",
]

set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
add_module_names = False  # Remove namespaces from class/method signatures

templates_path = ["_templates"]

# -- Options for extlinks ----------------------------------------------------

extlinks = {"pypi": ("https://pypi.org/project/%s/", "")}

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}

# -- Options for TODOs -------------------------------------------------------

todo_include_todos = True

# -- Options for Markdown files ----------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

autosummary_generate = True
autosummary_imported_members = True
autodoc_member_order = "bysource"
autodoc_default_flags = ["members"]
autodoc_class_signature = "separated"
autoclass_content = "init"

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
nnumpydoc_class_members_toctree = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "PyAutoLens"
html_short_title = "PyAutoLens"
html_permalinks_icon = "<span>#</span>"
html_last_updated_fmt = "%b %d, %Y"

html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

pygments_style = "sphinx"
pygments_dark_style = "monokai"
add_function_parentheses = False

html_context = {
    "menu_links_name": "Repository",
    "doc_path": "docs",
    "github_project": "pyautolens",
    "github_repo": "pyautolens",
    "github_version": "master",
}
language = "en"

html_static_path = ["_static"]
html_css_files = ["pied-piper-admonition.css"]

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#7C4DFF",
        "color-brand-content": "#7C4DFF",
    }
}

from sphinx.builders.html import StandaloneHTMLBuilder

StandaloneHTMLBuilder.supported_image_types = ["image/gif", "image/png", "image/jpeg"]
