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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'diffusion_for_multi_scale_molecular_dynamics'
copyright = '2023, amlrt_team'
author = 'amlrt_team'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = []

# enable use of markdown files
extensions.append('myst_parser')

# use the readthedocs theme
extensions.append('sphinx_rtd_theme')
extensions.append('sphinx.ext.napoleon')
extensions.append('sphinxcontrib.katex')

# autoapi extension for doc strings
extensions.append('autoapi.extension')
autoapi_type = 'python'
autoapi_dirs = ['../src/']


# Skip docstrings for loggers and tests
def check_skip_member(app, what, name, obj, skip, options):
    """Skips documentation when the function returns True."""
    SKIP_PATTERNS = ["test_", "logger"]
    for pattern in SKIP_PATTERNS:
        if pattern in name:
            print("Skipping documentation for: ", name)
            return True
    return False


def setup(app):
    """Handler to connect to the autoapi app."""
    app.connect("autoapi-skip-member", check_skip_member)


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
