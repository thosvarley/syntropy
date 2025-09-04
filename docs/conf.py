# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Syntropy'
copyright = '2025, Thomas F. Varley, PhD'
author = 'Thomas F. Varley, PhD'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # auto-import docstrings
    'sphinx.ext.napoleon',      # support Google/NumPy docstring style
    'sphinx.ext.viewcode',      # add links to source code
    'sphinx.ext.mathjax'        # enable MathJax 
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
autodoc_typehints = "description"
autodoc_member_order = 'bysource'

# %%
import os
import shutil

def copy_examples(app, exception):
    if exception is None:
        source = os.path.abspath('../examples')
        target = os.path.join(app.outdir, 'examples')
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.copytree(source, target)

def setup(app):
    app.connect("build-finished", copy_examples)

